import os
import argparse
import torch

from torch.utils.checkpoint import checkpoint as orig_checkpoint
from accelerate import Accelerator
from tqdm import tqdm
import torchvision
from torch.utils.data import DataLoader, Dataset

import wandb
import numpy as np
import inspect
import random
import gc
import math
import copy

from PIL import Image

import diffusers
from diffusers import FluxFillPipeline

def get_parser():
    parser = argparse.ArgumentParser(description="Accelerate Training Loop with wandb")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--wandb_name', type=str, required=True, help='wandb run name')
    parser.add_argument('--wandb_project', type=str, required=True, help='wandb project name (umbrella for runs)')
    parser.add_argument('--validation_epochs', type=int, default=1, help='How often (in epochs) to run validation')
    parser.add_argument('--save_epochs', type=int, default=1, help='How many epochs between checkpoints (default: 1). 0 indicates no intermediate saves')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (optional, if not set a random one will be generated)')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing for transformer')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--offload-heavy-encoders', action='store_true', default=False, help='Offload heavy encoder modules to CPU to save GPU memory when not in use')

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    return parser


class FluxFillDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.prompts_dir = os.path.join(root_dir, 'prompts')

        # Assume all images are numbered and have the same base name in all folders
        self.ids = [f.split('.')[0] for f in os.listdir(self.images_dir) if not f.startswith('.')]
        self.ids.sort()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        # Load image
        img_path = os.path.join(self.images_dir, f"{id_}.png")
        image = Image.open(img_path).convert('RGB')
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0  # (C, H, W), float32
        image = image 

        # Load mask
        mask_path = os.path.join(self.masks_dir, f"{id_}.png")
        mask = Image.open(mask_path).convert('L')
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0  # (1, H, W), float32
        mask = mask

        # Load prompt
        prompt_path = os.path.join(self.prompts_dir, f"{id_}.txt")
        with open(prompt_path, 'r') as f:
            prompt = f.read().strip()

        return image, mask, prompt
    
def get_weight_dtype(accelerator):
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    elif accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    else:
        return torch.float32
    
class DummyTransformer(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self._dtype = dtype
    def forward(self, *args, **kwargs):
        raise RuntimeError("Dummy transformer should not be called.")
    @property
    def dtype(self):
        return self._dtype
    
MAX_VALIDATION_IMAGES = 4
def validate(transformer, val_dataloader, accelerator, pipeline, epoch=-1, offload_heavy=False):
    if not accelerator.is_main_process:
        return
    
    weight_dtype = get_weight_dtype(accelerator)

    print(f"Validating at epoch {epoch}...")

    transformer.eval()
    val_losses = []
    generated_images: list[wandb.Image] = []

    # Generate validation images.
    # This requires setting `pipeline.transformer` to the unwrapped transformer model,
    # and then setting it to `eval`.
    #
    # Then at the end we need to set `pipeline.transformer` back to the dummy transformer.
    pipeline.transformer = accelerator.unwrap_model(transformer)
    pipeline.transformer.eval()
    pipeline.set_progress_bar_config(disable=True)
    
    with torch.no_grad():
        with torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype):
            for image, mask, prompt in val_dataloader:
                # Use the same training_step for validation, but do not backprop
                loss = training_step(transformer, pipeline, image, mask, prompt, weight_dtype, accelerator.device, offload_heavy)
                if loss is not None:
                    val_losses.append(loss.item())

                if len(generated_images) >= MAX_VALIDATION_IMAGES:
                    continue

                height, width = image.shape[2], image.shape[3]
                outputs = pipeline(
                    prompt=prompt,
                    image=image.to(device=accelerator.device, dtype=weight_dtype),
                    mask_image=mask.to(device=accelerator.device, dtype=weight_dtype),
                    height=height,
                    width=width,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    output_type="pil",
                ).images

                for generated_image, single_prompt in zip(outputs, prompt):
                    generated_image = generated_image.convert("RGB")
                    wandb_image = wandb.Image(generated_image, caption=single_prompt)
                    generated_images.append(wandb_image)

    avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('nan')
    log_dict = {"val/loss": avg_val_loss, "val/samples": generated_images if len(generated_images) > 0 else None}
    log_dict["epoch"] = epoch

    # Set the transformer back to the dummy transformer
    pipeline.transformer = DummyTransformer(dtype=getattr(transformer, "dtype", torch.float32))

    if offload_heavy:
        offload_pipeline_heavy(pipeline)

    wandb.log(log_dict)
    accelerator.print(f"Validation{' at epoch ' + str(epoch + 1) if epoch is not None else ''}: avg val loss = {avg_val_loss:.4f}")
    transformer.train()

def load_flux_fill(dtype):
    repo_id = "black-forest-labs/FLUX.1-Fill-dev"
    return FluxFillPipeline.from_pretrained(repo_id, torch_dtype=dtype)

# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps,
    device,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

# Save a full checkpoint for resuming training (model, optimizer, epoch)
def save_checkpoint(transformer, optimizer, epoch, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        "transformer": transformer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }, checkpoint_path)
    return checkpoint_path

# Load a full checkpoint for resuming training
def load_checkpoint(transformer, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    transformer.load_state_dict(checkpoint["transformer"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    return start_epoch

def select_timesteps(batch_size, pipeline):
    u = diffusers.training_utils.compute_density_for_timestep_sampling(
        weighting_scheme="none",
        batch_size=batch_size,
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
    )
    indices = (u * pipeline.training_scheduler.config.num_train_timesteps).long()
    timesteps = pipeline.training_scheduler.timesteps[indices]
    return timesteps

def get_sigmas(scheduler, timesteps, n_dim=4):
    sigmas = scheduler.sigmas
    schedule_timesteps = scheduler.timesteps.to(timesteps.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def prepare_latents_and_target(
    pipeline,
    image,
    timesteps,
    batch_size,
    num_channels_latents,
    height,
    width,
    dtype,
    device,
):
    shape = (batch_size, num_channels_latents, height, width)
    height = 2 * (int(height) // (pipeline.vae_scale_factor * 2))
    width = 2 * (int(width) // (pipeline.vae_scale_factor * 2))
    latent_image_ids = pipeline._prepare_latent_image_ids(
        batch_size, 
        height // 2, 
        width // 2, 
        device, 
        dtype,
    )

    image = image.to(device=device, dtype=dtype)

    if image.shape[1] != pipeline.latent_channels:
        image_latents = pipeline._encode_vae_image(image=image, generator=None)
    else:
        image_latents = image
    if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
        # expand init_latents for batch_size
        additional_image_per_prompt = batch_size // image_latents.shape[0]
        image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
    elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
        raise ValueError(
            f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
        )
    else:
        image_latents = torch.cat([image_latents], dim=0)

    noise = torch.randn_like(image_latents, device=device)


    target = noise - image_latents

    # add noise
    sigmas = get_sigmas(pipeline.training_scheduler, timesteps, n_dim=len(shape)).to(device)
    noisy_latents = (1.0 - sigmas) * image_latents + sigmas * noise

    packed_noisy_latents = pipeline._pack_latents(
        noisy_latents,                    
        batch_size=noisy_latents.shape[0],
        num_channels_latents=noisy_latents.shape[1],
        height=noisy_latents.shape[2],
        width=noisy_latents.shape[3]
    )
    return packed_noisy_latents, latent_image_ids, target

def offload_pipeline_heavy(pipeline):
    """
    Offload heavy pipeline modules (e.g., text_encoder_2 - t5) to CPU to save GPU memory.
    """
    if hasattr(pipeline, "text_encoder_2") and pipeline.text_encoder_2 is not None:
        pipeline.text_encoder_2.to("cpu")

    torch.cuda.empty_cache()

def load_pipeline_heavy(pipeline, device):
    """
    Load heavy pipeline modules (e.g., text_encoder_2 - t5) to the specified device.
    """
    if hasattr(pipeline, "text_encoder_2") and pipeline.text_encoder_2 is not None:
        pipeline.text_encoder_2.to(device)

# Runs a training step. returns the loss.
def training_step(transformer, pipeline, init_image, mask_image, prompt, weight_dtype, device, offload_heavy_encoders):
    """
    Args:
        transformer: The transformer module from the pipeline (pipeline.transformer)
        pipeline: The full FluxFillPipeline
        init_image: Tensor (B, C, H, W) - input images
        mask_image: Tensor (B, 1, H, W) - input masks
        prompt: List[str] - text prompts
        device: torch.device
    Returns:
        loss: torch.Tensor
    """

    gc.collect()
    torch.cuda.empty_cache()

    # disable gradient tracking for preparing inputs.
    with torch.no_grad():
        batch_size = init_image.shape[0]
        height, width = init_image.shape[2], init_image.shape[3]

        init_image = init_image.to(device, dtype=weight_dtype)
        mask_image = mask_image.to(device, dtype=weight_dtype)

        init_image = pipeline.image_processor.preprocess(init_image, height=height, width=width)
        
        # 1. Choose random timesteps for the entire batch.
        timesteps = select_timesteps(batch_size, pipeline).to(device=device)

        if offload_heavy_encoders:
            load_pipeline_heavy(pipeline, device)

        # 2. Encode prompts, then offload heavy pipeline modules to CPU to save memory
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            lora_scale=None,
        )

        # Offload heavy pipeline modules to CPU to save memory
        # This is important to avoid OOM errors during training.
        if offload_heavy_encoders:
            offload_pipeline_heavy(pipeline)

        prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)

        # 3. Prepare latents
        num_channels_latents = pipeline.vae.config.latent_channels
        height, width = init_image.shape[2], init_image.shape[3]

        latents, latent_image_ids, target = prepare_latents_and_target(
            pipeline,
            init_image,
            timesteps,
            batch_size,
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
        )

        # 4. Prepare masked latents
        mask_image = pipeline.mask_processor.preprocess(mask_image, height=height, width=width)
        mask_image = mask_image

        masked_image = init_image * (1 - mask_image)
        masked_image = masked_image.to(device=device, dtype=prompt_embeds.dtype)

        height, width = init_image.shape[-2:]
        mask, masked_image_latents = pipeline.prepare_mask_latents(
            mask_image,
            masked_image,
            batch_size,
            num_channels_latents,
            1,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator=None,
        )
        masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)

        guidance_scale = 30.0

        # 5. handle guidance
        if pipeline.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=weight_dtype)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

    # 6. Forward pass through the transformer. Everything from here on will be gradient-tracked.
    noise_pred = transformer(
        hidden_states=torch.cat((latents, masked_image_latents), dim=2),
        timestep=timesteps / 1000,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]

    noise_pred = pipeline._unpack_latents(
        noise_pred,
        height=target.shape[2] * pipeline.vae_scale_factor,
        width=target.shape[3] * pipeline.vae_scale_factor,
        vae_scale_factor=pipeline.vae_scale_factor,
    )

    # 7. Compute loss
    # For simplicity, we use MSE loss here, but you can use any other loss function as needed.
    loss = torch.mean(
        ((noise_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        1,
    )
    loss = loss.mean()
    return loss

def collate_fn(batch):
    images, masks, prompts = zip(*batch)
    images = torch.stack(images)
    # Ensure masks are single channel: (B, 1, H, W)
    # If mask has more than 1 channel, take only the first channel
    masks = torch.stack(masks)
    if masks.ndim == 4 and masks.shape[1] > 1:
        masks = masks[:, :1, ...]
    prompts = list(prompts)  # Ensures prompts is a list of strings
    return images, masks, prompts

def main():
    torch.utils.checkpoint.set_checkpoint_debug_enabled(True)

    parser = get_parser()
    args = parser.parse_args()

    # Set or generate random seed
    if args.seed is not None:
        seed = args.seed
    else:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    print(f"Using random seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # DataLoader drop_last explanation:
    # drop_last=True will drop the last batch if it's smaller than batch_size. This is useful if your model or pipeline requires fixed batch sizes.
    # drop_last=False (default) will include the last batch even if it's smaller. This is usually fine for most training, but can cause issues if your model expects fixed batch sizes.
    # You can set drop_last=True below if needed.

    # Initialize Weights & Biases with additional config
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "validation_epochs": args.validation_epochs,
            "save_epochs": args.save_epochs,
            "seed": seed,
            "drop_last": False,  # default, see above
        }
    )


    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    weight_dtype = get_weight_dtype(accelerator)
    print(f"Using weight dtype: {weight_dtype}")

    pipeline = load_flux_fill(weight_dtype)

    # Only train the transformer component
    transformer = pipeline.transformer

    pipeline.guidance_embeds = transformer.config.guidance_embeds
    # Take the transformer out of the pipeline for training and then send everything else to
    # the accelerator device.
    pipeline.transformer = DummyTransformer(dtype=getattr(transformer, "dtype", torch.float32))
    pipeline.to(accelerator.device)

    # explicitly disable gradient tracking for all non-transformer components
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)

    pipeline.training_scheduler = copy.deepcopy(pipeline.scheduler)

    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing for transformer...")
        if hasattr(transformer, 'enable_gradient_checkpointing'):
            transformer.enable_gradient_checkpointing()
        else:
            print("Warning: transformer does not support gradient checkpointing.")

    dataset = FluxFillDataset(os.path.join('data', 'training'))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)

    # Validation dataset and dataloader
    val_dataset = FluxFillDataset(os.path.join('data', 'validation'))
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        transformer.parameters(), 
        lr=args.lr, 
        fused=True,
    )

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    len_train_dataloader_after_sharding = math.ceil(len(dataloader) / accelerator.num_processes)
    num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
    num_training_steps_for_scheduler = (
        args.epochs * accelerator.num_processes * num_update_steps_per_epoch
    )
    lr_scheduler = diffusers.optimization.get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(transformer, optimizer, dataloader, lr_scheduler)

    print(f"Optimizer dtype: {next(iter(optimizer.param_groups[0]['params'])).dtype}")

    validate(transformer, val_dataloader, accelerator, pipeline, offload_heavy=args.offload_heavy_encoders)

    transformer.train()
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            epoch_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            epoch_bar = enumerate(dataloader)
        for step, batch in epoch_bar:
            images, masks, prompts = batch
            # Use the modular training_step function for per-batch training logic
            with accelerator.accumulate(transformer):
                with accelerator.autocast():
                    loss = training_step(
                        transformer, pipeline, images, masks, prompts, weight_dtype, accelerator.device, args.offload_heavy_encoders
                    )
                if loss is None:
                    raise RuntimeError("training_step returned None. Check implementation.")
                
                accelerator.backward(loss)
                # Only step optimizer and zero grad when gradients are synced (i.e., after accumulation)
                if accelerator.sync_gradients:
                    torch.cuda.empty_cache()  # Clear cache to avoid OOM errors
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Log loss and learning rate to wandb (log only on main process and after accumulation step)
            if accelerator.is_main_process and accelerator.sync_gradients:
                lr = lr_scheduler.get_last_lr()[0]
                wandb.log({"train/loss": loss.item(), "train/lr": lr, "epoch": epoch, "step": step})

        # Validation logic
        is_last_epoch = (epoch + 1) == args.epochs
        if (epoch + 1) % args.validation_epochs == 0 and not is_last_epoch:
            validate(transformer, val_dataloader, accelerator, pipeline, epoch=epoch, offload_heavy=args.offload_heavy_encoders)

        # Checkpoint saving logic (avoid duplicate save at end, and allow save_epochs==0 to mean 'never except end')

        if args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0 and not is_last_epoch and accelerator.is_main_process:
            save_checkpoint(transformer, optimizer, epoch + 1)

    # Final validation at the end
    validate(transformer, val_dataloader, accelerator, pipeline, epoch=args.epochs-1, offload_heavy=args.offload_heavy_encoders)

    # Always save a final checkpoint at the end
    if accelerator.is_main_process:
        save_checkpoint(transformer, optimizer, args.epochs)
    wandb.finish()

if __name__ == "__main__":
    main()
