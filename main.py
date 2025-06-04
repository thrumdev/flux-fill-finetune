import os
import argparse
import torch
import re
import yaml

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
import lpips

from PIL import Image

import diffusers
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline

def get_parser():
    parser = argparse.ArgumentParser(description="Accelerate Training Loop with wandb")
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--wandb_name', type=str, help='wandb run name')
    parser.add_argument('--wandb_project', type=str, help='wandb project name (umbrella for runs)')
    parser.add_argument('--validation_epochs', type=int, default=1, help='How often (in epochs) to run validation')
    parser.add_argument('--save_epochs', type=int, default=1, help='How many epochs between checkpoints (default: 1). 0 indicates no intermediate saves')
    parser.add_argument('--max_checkpoints', type=int, default=0, help='Maximum number of checkpoints to keep (default: 0, unlimited). Older checkpoints will be deleted if exceeded.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (optional, if not set a random one will be generated)')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing for transformer')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--offload_heavy_encoders', action='store_true', default=False, help='Offload heavy encoder modules to CPU to save GPU memory when not in use')

    parser.add_argument('--mask_loss_weight', type=float, default=5.0, help='Weight multiplier for the masked area in the loss (default: 5.0)')
    parser.add_argument('--mse_loss_weight', type=float, default=0.8, help='Weighting of the MSE loss in the total loss (default: 0.8)')
    parser.add_argument('--pixel_loss_weight', type=float, default=0.2, help='Weighting of the LPIPS (pixel) loss in the total loss (default: 0.2)')

    parser.add_argument('--pixel_loss_noise_threshold', type=float, default=0.5, help='Noise threshold (sigma) below which pixel/LPIPS loss is applied (default: 0.5). Currently unused.')

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
    parser.add_argument(
        '--trainable_params',
        type=str,
        nargs='*',
        default=[r'.*'],
        help='Regex or list of regexes to select trainable parameters by name. Default: all parameters.'
    )

    parser.add_argument(
        '--restore_from_checkpoint',
        type=str,
        default=None,
        help='Path to a .pt checkpoint file to restore training from'
    )
    parser.add_argument(
        '--restore_optimizer',
        action='store_true',
        default=False,
        help='Whether to restore the optimizer state from the checkpoint (default: False). If True, the optimizer state will be restored from the checkpoint.'
    )
    
    parser.add_argument(
        '--use_mask_ratio_weight',
        action='store_true',
        default=False,
        help='If set, use mask-to-image size ratio as a weighting factor in loss calculation.'
    )
    parser.add_argument(
        '--flux_redux_rate',
        type=float,
        default=0.0,
        help='The rate (fraction of training steps) to use Flux-Redux to condition. Default: 0.0.'
    )
    parser.add_argument(
        '--flux_redux_scale',
        type=float,
        default=1.0,
        help='Scale factor (between 0 and 1, default 1) for Flux-Redux conditioning.'
    )
    return parser

def parse_args_with_config():
    parser = get_parser()
    # Parse only --config first
    args, remaining_argv = parser.parse_known_args()
    defaults = {}
    if getattr(args, 'config', None):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            if config:
                defaults.update(config)
    # Rebuild parser with defaults from config
    parser = get_parser()
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    return args

def enable_lpips_gradient_checkpointing(lpips_model):
    """
    Monkey-patch LPIPS to enable gradient checkpointing for its feature network.
    This wraps each feature block in torch.utils.checkpoint.checkpoint.
    """

    def monkey_patch_block(block):
        if isinstance(block, torch.nn.Module):
            if not hasattr(block, "_original_forward"):
                block._original_forward = block.forward
                def checkpointed_forward(*inputs, **kwargs):
                    return torch.utils.checkpoint.checkpoint(
                        block._original_forward, *inputs, use_reentrant=True, **kwargs
                    )
                block.forward = checkpointed_forward
    
    # patch underlying net
    monkey_patch_block(lpips_model.net.slice1)
    monkey_patch_block(lpips_model.net.slice2)
    monkey_patch_block(lpips_model.net.slice3)
    monkey_patch_block(lpips_model.net.slice4)
    monkey_patch_block(lpips_model.net.slice5)

    # patch lpips linear interpolation layers.
    monkey_patch_block(lpips_model.lin0)
    monkey_patch_block(lpips_model.lin1)
    monkey_patch_block(lpips_model.lin2)
    monkey_patch_block(lpips_model.lin3)
    monkey_patch_block(lpips_model.lin4)

    return lpips_model

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
    
def get_trainable_params(model, patterns):
    """
    Given a model and a list of regex patterns, return a list of parameters whose names match any pattern.
    """
    if isinstance(patterns, str):
        patterns = [patterns]

    param_name_to_param = dict(model.named_parameters())
    trainable_param_names = set()
    for pattern in patterns:
        regex = re.compile(pattern)
        for name in param_name_to_param:
            if regex.fullmatch(name) or regex.search(name):
                trainable_param_names.add(name)
    trainable_params = [param for name, param in param_name_to_param.items() if name in trainable_param_names]
    if not trainable_params:
        raise ValueError(f"No trainable parameters matched the provided regex(es): {patterns}")
    return trainable_params

def set_trainable_params(model, trainable_params):
    """
    Set the requires_grad attribute of parameters matching any of the provided regex patterns to True.
    """
    for param in model.parameters():
        param.requires_grad = False  # Disable all by default
    for param in trainable_params:
        param.requires_grad = True  # Enable only the selected ones

def print_trainable_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Training parameter: {name}")
    
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
def validate(transformer, val_dataloader, accelerator, pipeline, redux_pipeline, config, epoch=-1):
    if not accelerator.is_main_process:
        return
    
    weight_dtype = get_weight_dtype(accelerator)

    print(f"Validating at epoch {epoch + 1}...")

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
                loss = training_step(transformer, pipeline, redux_pipeline, image, mask, prompt, weight_dtype, accelerator.device, config)
                if loss is not None:
                    val_losses.append(loss)

                if len(generated_images) >= MAX_VALIDATION_IMAGES:
                    continue

                height, width = image.shape[2], image.shape[3]
                if redux_pipeline is not None:
                    (prompt_embeds, pooled_prompt_embeds) = redux_pipeline(
                        image.to(device=accelerator.device, dtype=weight_dtype),
                        prompt_embeds_scale=config.flux_redux_scale,
                        pooled_prompt_embeds_scale=config.flux_redux_scale,
                        return_dict=False,
                    )

                    outputs = pipeline(
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        image=image.to(device=accelerator.device, dtype=weight_dtype),
                        mask_image=mask.to(device=accelerator.device, dtype=weight_dtype),
                        height=height,
                        width=width,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        output_type="pil",
                    ).images
                else:
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

    n = len(val_losses)
    avg_val_loss = sum([v["loss"] for v in val_losses]) / n if n > 0 else float('nan')
    avg_mse_loss = sum([v["mse_loss"] for v in val_losses]) / n if n > 0 else float('nan')
    avg_pixel_loss = sum([v["pixel_loss"] for v in val_losses]) / n if n > 0 else float('nan')
    log_dict = {
        "val/loss": avg_val_loss, 
        "val/mse_loss": avg_mse_loss,
        "val/pixel_loss": avg_pixel_loss,
        "val/samples": generated_images if len(generated_images) > 0 else None,
    }
    log_dict["epoch"] = epoch

    # Set the transformer back to the dummy transformer
    pipeline.transformer = DummyTransformer(dtype=getattr(transformer, "dtype", torch.float32))

    if config.offload_heavy_encoders:
        offload_pipeline_heavy(pipeline)

    if wandb.run is not None:
        wandb.log(log_dict)
    accelerator.print(f"Validation{' at epoch ' + str(epoch + 1) if epoch is not None else ''}:")
    accelerator.print(f"\tavg. loss total= {avg_val_loss:.4f} mse= {avg_mse_loss:.4f} pixel= {avg_pixel_loss:.4f}")
    transformer.train()

def load_flux_fill(dtype, ignore_text_encoders=False):
    repo_id = "black-forest-labs/FLUX.1-Fill-dev"
    if ignore_text_encoders:
        # If we want to ignore text encoders, we can set them to None
        return FluxFillPipeline.from_pretrained(
            repo_id, 
            text_encoder=None,
            text_encoder_2=None,
            torch_dtype=dtype,
        )
    else:
        return FluxFillPipeline.from_pretrained(repo_id, torch_dtype=dtype)

def load_flux_redux(dtype):
    repo_id = "black-forest-labs/FLUX.1-Redux-dev"
    return FluxPriorReduxPipeline.from_pretrained(repo_id, torch_dtype=dtype)

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

def manage_max_checkpoints(checkpoint_dir, max_checkpoints):
    """
    Deletes oldest checkpoints if there are more than max_checkpoints-1 in the directory.
    This is intended to be called BEFORE saving a new checkpoint, so that after saving, the count is at most max_checkpoints.
    """
    if max_checkpoints == 0:
        return
    # List all checkpoint files matching the pattern
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
    # Extract epoch numbers and sort by epoch
    def extract_epoch(filename):
        import re
        m = re.search(r"checkpoint_epoch_(\d+)\.pt", filename)
        return int(m.group(1)) if m else -1
    files = sorted(files, key=extract_epoch)
    # If more than max_checkpoints-1, delete the oldest (so after saving, count is at most max_checkpoints)
    while len(files) > max_checkpoints - 1:
        to_delete = files.pop(0)
        try:
            print(f"Checkpoint limit reached, deleting old checkpoint: {to_delete}")
            os.remove(os.path.join(checkpoint_dir, to_delete))
        except Exception as e:
            print(f"Warning: failed to delete old checkpoint {to_delete}: {e}")

# Save a full checkpoint for resuming training (model, optimizer, epoch)
def save_checkpoint(transformer, optimizer, epoch, checkpoint_dir="checkpoints", max_checkpoints=0):
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Delete old checkpoints BEFORE saving the new one, to limit peak disk usage
    manage_max_checkpoints(checkpoint_dir, max_checkpoints)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        "transformer": transformer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }, checkpoint_path)
    return checkpoint_path


# Load a full checkpoint for resuming training
def load_checkpoint(transformer, optimizer, checkpoint_path, restore_optimizer):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    transformer.load_state_dict(checkpoint["transformer"])
    if restore_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0
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
    image_latents,
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
    return noisy_latents, packed_noisy_latents, latent_image_ids, target

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
def training_step(transformer, pipeline, redux_pipeline, init_image, mask_image, prompt, weight_dtype, device, config):
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

        if config.offload_heavy_encoders:
            load_pipeline_heavy(pipeline, device)
            load_pipeline_heavy(redux_pipeline, device)

        # 2. Encode prompts, then offload heavy pipeline modules to CPU to save memory

        use_flux_redux = random.random() < config.flux_redux_rate
        if use_flux_redux:
            (prompt_embeds, pooled_prompt_embeds) = redux_pipeline(
                image=init_image,
                prompt_embeds_scale=config.flux_redux_scale,
                pooled_prompt_embeds_scale=config.flux_redux_scale,
                return_dict=False,
            )
            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=weight_dtype)
        else:
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
        if config.offload_heavy_encoders:
            offload_pipeline_heavy(pipeline)
            offload_pipeline_heavy(redux_pipeline)

        prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)

        # 3. Prepare latents
        num_channels_latents = pipeline.vae.config.latent_channels
        height, width = init_image.shape[2], init_image.shape[3]

        clean_latents = pipeline._encode_vae_image(image=init_image, generator=None)
        noisy_latents, packed_noisy_latents, latent_image_ids, target = prepare_latents_and_target(
            pipeline,
            clean_latents,
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
            guidance = guidance.expand(packed_noisy_latents.shape[0])
        else:
            guidance = None

    # 6. Forward pass through the transformer. Everything from here on will be gradient-tracked.
    noise_pred = transformer(
        hidden_states=torch.cat((packed_noisy_latents, masked_image_latents), dim=2),
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
    return compute_loss(
        config, 
        pipeline, 
        noise_pred, 
        target, 
        timesteps, 
        mask_image, 
        noisy_latents, 
        clean_latents,
    )

def vae_scale_mask(mask, pipeline):
    height, width = mask.shape[2], mask.shape[3]
    height = 2 * (int(height) // (pipeline.vae_scale_factor * 2))
    width = 2 * (int(width) // (pipeline.vae_scale_factor * 2))

    # Scale the mask to the VAE latent size. Input is (B, 1, H, W), output is (B, 1, H', W')
    mask = torchvision.transforms.functional.resize(
        mask, (height, width), interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )

    return mask

def get_mask_loss_weight_by_ratio(mask_image):
    # input mask_image is a tensor of shape (B, 1, H, W)
    # returns a weight tensor that has shape (B,) where each item is the reciprocal of the ratio
    # of the number of masked pixels to the total number of pixels in the mask

    h, w = mask_image.shape[2], mask_image.shape[3]
    # first combine dims 1, 2, 3 into a single dim
    mask_image_flat = mask_image.reshape(mask_image.shape[0], -1)  # (B, H*W)
    # count the number of masked pixels (where mask_image_flat > 0)
    num_masked_pixels = mask_image_flat.sum(dim=1)  # (B,)
    # set num_masked_pixels to minimum of 1 to avoid division by zero, though empty masks aren't 
    # expected in practice.
    num_masked_pixels = torch.clamp(num_masked_pixels, min=1.0)  # (B,)
    # compute the weight as the reciprocal of the ratio of masked pixels to total pixels
    mask_weight = (h * w) / num_masked_pixels
    return mask_weight

def compute_loss(
    config, 
    pipeline, 
    noise_pred, 
    target,
    timesteps, 
    mask_image, 
    noisy_latents,
    clean_latents,
):
    with torch.no_grad():
        # Reshape mask_latent_sized to have the same number of channels as noise_pred and target
        # (B, 1, H', W') -> (B, latent_channels, H', W')
        mask_latent_sized = vae_scale_mask(mask_image, pipeline).expand_as(noise_pred)

        # Compute mask loss weight based on the ratio of masked pixels
        if config.use_mask_ratio_weight:
            mask_loss_weight = get_mask_loss_weight_by_ratio(mask_image) * config.mask_loss_weight
        else:
            # make (B,) shaped tensor where each item has value `config.mask_loss_weight`
            mask_loss_weight = torch.full((mask_latent_sized.shape[0],), config.mask_loss_weight, device=noisy_latents.device, dtype=noisy_latents.dtype)
            
        # Reshape to (B, 1, 1, 1) for broadcasting across the batch dimension
        mask_loss_weight = mask_loss_weight.reshape(-1, 1, 1, 1) 

        # weight the masked portions higher in the loss.
        weighted_latent_mask = (mask_latent_sized * mask_loss_weight) + (1 - mask_latent_sized)
        mask_weighted_target = target * weighted_latent_mask

    # apply the mask weight to the noise prediction and target
    mask_weighted_noise_pred = noise_pred * weighted_latent_mask
    mask_weighted_mse_loss = torch.mean(
        ((mask_weighted_noise_pred.float() - mask_weighted_target.float()) ** 2).reshape(target.shape[0], -1),
        1,
    )
    mask_weighted_mse_loss = mask_weighted_mse_loss.mean()

    if config.pixel_loss_weight > 0:
        with torch.no_grad():
            sigmas = get_sigmas(
                pipeline.training_scheduler, 
                timesteps, 
                n_dim=len(noisy_latents.shape)
            ).to(noisy_latents.device)

        # Decode the noise prediction to get the pixel space prediction
        model_noised_latents = (1.0 - sigmas) * clean_latents + sigmas * noise_pred
        predicted_pixels = pipeline.vae.decode(model_noised_latents, return_dict=False)[0]

        with torch.no_grad():
            noisy_image = pipeline.vae.decode(noisy_latents, return_dict=False)[0]

            # get the batch indices of sigmas where the sigma is below the pixel loss noise threshold.
            # note that sigmas is a (B, 1, 1, 1) tensor,
            # as a single-dim tensor, this will be a (B,) tensor.
            active_batch_indices = (sigmas <= config.pixel_loss_noise_threshold).nonzero(as_tuple=False).reshape(-1)    
            noisy_image = noisy_image[active_batch_indices]
            mask_weights = mask_loss_weight[active_batch_indices]

        predicted_pixels = predicted_pixels[active_batch_indices]

        if active_batch_indices.numel() > 0:
            # note: shape here is (B,)
            pixel_loss = pipeline.lpips_loss(predicted_pixels, noisy_image)
            pixel_loss = pixel_loss * mask_weights.squeeze()
            
            pixel_loss = pixel_loss.mean()
        else:
            # If no active batch indices, return 0 loss
            pixel_loss = 0

    else:
        pixel_loss = 0

    total_loss = config.mse_loss_weight * mask_weighted_mse_loss + config.pixel_loss_weight * pixel_loss
    return {
        "loss": total_loss,
        "mse_loss": mask_weighted_mse_loss,
        "pixel_loss": pixel_loss,
    }

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

    args = parse_args_with_config()

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
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    weight_dtype = get_weight_dtype(accelerator)
    print(f"Using weight dtype: {weight_dtype}")

    # Initialize Weights & Biases with additional config
    if args.wandb_name is not None and args.wandb_project is not None:
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
                "dtype": weight_dtype,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "mask_loss_weight": args.mask_loss_weight,
                "mse_loss_weight": args.mse_loss_weight,
                "pixel_loss_weight": args.pixel_loss_weight,
                "pixel_loss_noise_threshold": args.pixel_loss_noise_threshold,
                "lr_scheduler": args.lr_scheduler,
                "trainable_params": args.trainable_params,
                "use_mask_ratio_weight": args.use_mask_ratio_weight,
                "drop_last": False,  # default, see above
            }
        )

    pipeline = load_flux_fill(weight_dtype, args.flux_redux_rate > 0 and args.flux_redux_rate < 1.0)
    if args.flux_redux_rate > 0:
        redux_pipeline = load_flux_redux(weight_dtype).to(accelerator.device)

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

    # Spatial means that the LPIPS loss will be computed per-pixel
    pipeline.lpips_loss = lpips.LPIPS(net='vgg', spatial=False).to(accelerator.device)

    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing for transformer...")
        if hasattr(transformer, 'enable_gradient_checkpointing'):
            transformer.enable_gradient_checkpointing()
            if args.pixel_loss_weight > 0:
                pipeline.vae.enable_gradient_checkpointing()
                enable_lpips_gradient_checkpointing(pipeline.lpips_loss)
        else:
            print("Warning: transformer does not support gradient checkpointing.")

    dataset = FluxFillDataset(os.path.join('data', 'training'))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)

    # Validation dataset and dataloader
    val_dataset = FluxFillDataset(os.path.join('data', 'validation'))
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn)

    trainable_params = get_trainable_params(transformer, args.trainable_params)
    set_trainable_params(transformer, trainable_params)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        fused=True,
    )

    start_epoch = 0
    if args.restore_from_checkpoint is not None:
        print(f"Restoring from checkpoint: {args.restore_from_checkpoint}")
        start_epoch = load_checkpoint(transformer, optimizer, args.restore_from_checkpoint, args.restore_optimizer)
        print(f"Resuming training from epoch {start_epoch}")

    print_trainable_params(transformer)
        
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
        last_epoch=start_epoch-1,
    )

    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(transformer, optimizer, dataloader, lr_scheduler)

    print(f"Optimizer dtype: {next(iter(optimizer.param_groups[0]['params'])).dtype}")

    validate(transformer, val_dataloader, accelerator, pipeline, redux_pipeline, config=args)

    transformer.train()
    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch, end_epoch):
        if accelerator.is_main_process:
            epoch_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{end_epoch}")
        else:
            epoch_bar = enumerate(dataloader)

        total_steps = len(dataloader)
        losses = []
        for step, batch in epoch_bar:
            images, masks, prompts = batch
            # Use the modular training_step function for per-batch training logic
            with accelerator.accumulate(transformer):
                with accelerator.autocast():
                    loss = training_step(
                        transformer, 
                        pipeline, 
                        redux_pipeline, 
                        images, 
                        masks, 
                        prompts, 
                        weight_dtype, 
                        accelerator.device, 
                        args
                    )
                if loss is None:
                    raise RuntimeError("training_step returned None. Check implementation.")
                
                accelerator.backward(loss["loss"])
                # Only step optimizer and zero grad when gradients are synced (i.e., after accumulation)
                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Log loss and learning rate to wandb
            if accelerator.is_main_process:
                losses.append(loss)
                lr = lr_scheduler.get_last_lr()[0]
                if wandb.run is not None:
                    wandb.log({
                        "train/loss": loss["loss"], 
                        "train/mse_loss": loss["mse_loss"],
                        "train/pixel_loss": loss["pixel_loss"],
                        "train/lr": lr, 
                        "train/step": step,
                        "epoch": epoch, 
                    })

        avg_loss = sum([v["loss"] for v in losses]) / len(losses) if losses else float('nan')
        avg_mse_loss = sum([v["mse_loss"] for v in losses]) / len(losses) if losses else float('nan')
        avg_pixel_loss = sum([v["pixel_loss"] for v in losses]) / len(losses) if losses else float('nan')

        print(f"Epoch {epoch + 1}/{end_epoch} - Avg Loss: {avg_loss:.4f}, MSE Loss: {avg_mse_loss:.4f}, Pixel Loss: {avg_pixel_loss:.4f}")

        # Validation logic
        is_last_epoch = (epoch + 1) == end_epoch
        if (epoch + 1) % args.validation_epochs == 0 and not is_last_epoch:
            validate(transformer, val_dataloader, accelerator, pipeline, redux_pipeline, epoch=epoch, config=args)

        # Checkpoint saving logic (avoid duplicate save at end, and allow save_epochs==0 to mean 'never except end')

        if args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0 and not is_last_epoch and accelerator.is_main_process:
            save_checkpoint(
                accelerator.unwrap_model(transformer), 
                optimizer, 
                epoch + 1, 
                checkpoint_dir="checkpoints", 
                max_checkpoints=args.max_checkpoints,
            )

    # Final validation at the end
    validate(transformer, val_dataloader, accelerator, pipeline, redux_pipeline, epoch=end_epoch-1, config=args)

    # Always save a final checkpoint at the end
    if accelerator.is_main_process:
        save_checkpoint(
            accelerator.unwrap_model(transformer), 
            optimizer, 
            end_epoch, 
            checkpoint_dir="checkpoints", 
            max_checkpoints=args.max_checkpoints,
        )
    wandb.finish()

if __name__ == "__main__":
    main()
