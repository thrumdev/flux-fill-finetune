import argparse
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader, Dataset
import wandb
import numpy as np
import inspect

from diffusers import FluxFillPipeline

def get_parser():
    parser = argparse.ArgumentParser(description="Accelerate Training Loop with wandb")
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--wandb_name', type=str, default='dummy-run', help='wandb run name')
    parser.add_argument('--validation_epochs', type=int, default=1, help='How often (in epochs) to run validation')
    return parser


import os
from PIL import Image


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

        # Load mask
        mask_path = os.path.join(self.masks_dir, f"{id_}.png")
        mask = Image.open(mask_path).convert('L')

        # Load prompt
        prompt_path = os.path.join(self.prompts_dir, f"{id_}.txt")
        with open(prompt_path, 'r') as f:
            prompt = f.read().strip()

        return image, mask, prompt

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)
    
def validate(model, val_dataloader, accelerator, epoch=None):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x_val, y_val in val_dataloader:
            outputs = model(x_val)
            val_loss = torch.nn.functional.mse_loss(outputs.squeeze(), y_val.float())
            val_losses.append(val_loss.item())
    avg_val_loss = sum(val_losses) / len(val_losses)
    log_dict = {"val_loss": avg_val_loss}
    if epoch is not None:
        log_dict["epoch"] = epoch
    wandb.log(log_dict)
    accelerator.print(f"Validation{' at epoch ' + str(epoch) if epoch is not None else ''}: avg val loss = {avg_val_loss:.4f}")
    model.train()

def load_flux_fill():
    repo_id = "black-forest-labs/FLUX.1-Fill-dev"
    return FluxFillPipeline.from_pretrained(repo_id, torch_dtype=torch.bfloat16).to("cuda")

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

def prepare_latents_and_noise(
    pipeline,
    image,
    timestep,
    batch_size,
    num_channels_latents,
    height,
    width,
    dtype,
    device,
):
    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (pipeline.vae_scale_factor * 2))
    width = 2 * (int(width) // (pipeline.vae_scale_factor * 2))
    shape = (batch_size, num_channels_latents, height, width)
    latent_image_ids = pipeline._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

    image = image.to(device=device, dtype=dtype)
    if image.shape[1] != pipeline.latent_channels:
        image_latents = pipeline._encode_vae_image(image=image)
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

    noise = torch.randn(shape, dtype=dtype, device=device)
    latents = pipeline.scheduler.scale_noise(image_latents, timestep, noise)
    latents = pipeline._pack_latents(latents, batch_size, num_channels_latents, height, width)
    return latents, latent_image_ids, noise

# Runs a training step. returns the loss.
def training_step(transformer, pipeline, init_image, mask_image, prompt, device):
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

    batch_size = init_image.shape[0]
    height, width = init_image.shape[2], init_image.shape[3]

    init_image = pipeline.image_processor.preprocess(init_image, height=height, width=width)
    init_image = init_image.to(dtype=torch.float32)
    
    # 1. Choose a random timestep for the entire batch.
    num_inference_steps = torch.randint(20, 50);
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = (int(height) // pipeline.vae_scale_factor // 2) * (int(width) // pipeline.vae_scale_factor // 2)
    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    timesteps, num_inference_steps = pipeline.get_timesteps(num_inference_steps, 1.0, device)

    # 2. Encode prompts
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

    # 3. Prepare latents
    num_channels_latents = pipeline.vae.config.latent_channels
    latent_timestep = timesteps[:1].repeat(batch_size)
    latents, latent_image_ids, noise = prepare_latents_and_noise(
        pipeline,
        init_image,
        latent_timestep,
        batch_size,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
    )

    # 4. Prepare masked latents
    mask_image = pipeline.mask_processor.preprocess(mask_image, height=height, width=width)

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
    if pipeline.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    # 6. Forward pass through the transformer
    t = torch.randint(0, len(timesteps), (1,), device=device)
    timestep = t.expand(latents.shape[0]).to(latents.dtype)

    noise_pred = transformer(
        hidden_states=torch.cat((latents, masked_image_latents), dim=2),
        timestep=timestep / 1000,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]

    # compute the previous noisy sample x_t -> x_t-1
    latents_dtype = latents.dtype
    output_latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # 7. Compute loss
    # For simplicity, we use MSE loss here, but you can use any other loss function as needed.
    target = noise - latents
    loss = torch.mean(
        ((noise_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        1,
    )
    loss = loss.mean()

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Initialize Weights & Biases
    wandb.init(project="flux-fill-finetuning", name=args.wandb_name, config={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr})

    accelerator = Accelerator()
    pipeline = load_flux_fill()

    # Only train the transformer component
    transformer = pipeline.transformer
    optimizer = torch.optim.Adam(transformer.parameters(), lr=args.lr)

    dataset = FluxFillDataset(os.path.join('data', 'training'))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Validation dataset and dataloader
    val_dataset = FluxFillDataset(os.path.join('data', 'validation'))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    transformer, optimizer, dataloader = accelerator.prepare(transformer, optimizer, dataloader)

    transformer.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            images, masks, prompts = batch
            optimizer.zero_grad()
            # TODO: Add diffusion training logic here, including noise addition, timestep sampling, and forward pass
            # Example placeholder loss:
            loss = torch.tensor(0.0, device=images.device, requires_grad=True)
            accelerator.backward(loss)
            optimizer.step()
            # Log loss to wandb
            wandb.log({"loss": loss.item(), "epoch": epoch, "step": step})
            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")

        # Validation logic
        if (epoch + 1) % args.validation_epochs == 0:
            validate(transformer, val_dataloader, accelerator, epoch=epoch)

    # Final validation at the end
    validate(transformer, val_dataloader, accelerator)
    wandb.finish()

if __name__ == "__main__":
    main()
