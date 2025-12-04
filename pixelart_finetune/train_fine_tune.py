"""
Fine-tune Stable Diffusion (runwayml/stable-diffusion-v1-5) on a custom
pixel art dataset using image+caption pairs stored in two folders:

    train_image_dir/
        image1.png
        image2.png
        ...
    train_captions_dir/
        image1.txt
        image2.txt
        ...

Each .txt file contains the caption for the image with the same base name.

This script:
- loads a pre-trained Stable Diffusion pipeline
- freezes the VAE and text encoder
- fine-tunes the UNet on your pixel art data
- saves the fine-tuned model
- generates a before/after comparison image
"""

import os
import math
import argparse
from glob import glob

# Avoid potential PyTorch dynamo issues
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler


# -------------------------
# Helpers
# -------------------------

def get_device() -> str:
    """Return the best available device: cuda, mps (Apple), or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def image_to_latents(vae, images: torch.Tensor) -> torch.Tensor:
    """
    Encode images (scaled to [-1,1]) into latent space.
    """
    latents = vae.encode(images).latent_dist.sample()
    # Scale factor used in Stable Diffusion
    latents = latents * 0.18215
    return latents


# -------------------------
# Dataset
# -------------------------

class PixelArtDataset(Dataset):
    """
    Dataset for pixel art with separate image and caption directories.

    image_dir: folder containing images (png/jpg/jpeg/webp/bmp)
    captions_dir: folder containing .txt files with same basename as images
    """

    def __init__(self, image_dir: str, captions_dir: str, tokenizer, resolution: int = 512):
        self.image_dir = image_dir
        self.captions_dir = captions_dir
        self.tokenizer = tokenizer
        self.resolution = resolution

        exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(glob(os.path.join(image_dir, ext)))

        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

        self.image_paths = sorted(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path: str) -> Image.Image:
        image = Image.open(path).convert("RGB")
        # Center-crop to square then resize
        w, h = image.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        image = image.crop((left, top, left + s, top + s))
        image = image.resize((self.resolution, self.resolution), Image.BICUBIC)
        return image

    def _load_caption(self, img_path: str) -> str:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(self.captions_dir, base_name + ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            if caption:
                return caption
        # Fallback generic caption if no .txt or empty
        return "pixel art"

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = self._load_image(img_path)
        caption = self._load_caption(img_path)

        # Convert to tensor in [-1, 1]
        np_image = np.array(image).astype(np.float32) / 255.0
        np_image = (np_image * 2.0) - 1.0
        np_image = np.transpose(np_image, (2, 0, 1))  # HWC -> CHW
        pixel_values = torch.from_numpy(np_image)

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids[0],
        }


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    input_ids = torch.stack([e["input_ids"] for e in examples])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
    }


# -------------------------
# Training
# -------------------------

def train(
    train_image_dir: str,
    train_captions_dir: str,
    output_dir: str,
    resolution: int = 512,
    batch_size: int = 2,
    lr: float = 1e-5,
    num_epochs: int = 5,
    num_inference_steps: int = 30,
    comparison_prompt: str = "pixel art of a knight holding a sword, 32-bit style, vibrant colors",
):
    device = get_device()
    print(f"Using device: {device}")

    # Load pretrained Stable Diffusion pipeline
    print("Loading base Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
    )

    # Disable safety checker to avoid unexpected black/blurred images
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    pipe = pipe.to(device)

    # Extract components
    vae = pipe.vae
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # Scheduler for training
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(True)

    vae.eval()
    text_encoder.eval()
    unet.train()

    # Dataset & DataLoader
    dataset = PixelArtDataset(train_image_dir, train_captions_dir, tokenizer, resolution=resolution)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    # Training loop
    global_step = 0
    total_steps = num_epochs * math.ceil(len(train_dataloader))

    print(f"Start training for {num_epochs} epochs, {total_steps} steps total.")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            with torch.no_grad():
                # Move images to device
                pixel_values = batch["pixel_values"].to(device)

                # Encode images to latents
                latents = image_to_latents(vae, pixel_values)

                # Sample random timesteps
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=device,
                    dtype=torch.long,
                )

                # Sample noise and add to latents
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                input_ids = batch["input_ids"].to(device)
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Predict noise with UNet
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Loss is MSE between predicted noise and true noise
            loss = nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "step": global_step,
            })

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} done. Avg loss: {avg_loss:.4f}")

    # Save fine-tuned UNet
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving fine-tuned model to {output_dir} ...")

    # Save full pipeline with updated UNet
    pipe.unet = unet
    pipe.save_pretrained(output_dir)

    print("Model saved successfully.")

    # -------------------------
    # Comparison: before vs after
    # -------------------------
    print("Generating before/after comparison image...")

    # Base model
    base_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
    )
    base_pipe.safety_checker = None
    base_pipe.requires_safety_checker = False
    base_pipe = base_pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(42)

    # Before fine-tune
    with torch.no_grad():
        base_image = base_pipe(
            comparison_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

    # After fine-tune (reload fine-tuned)
    ft_pipe = StableDiffusionPipeline.from_pretrained(
        output_dir,
        torch_dtype=torch.float32,
    )
    ft_pipe.safety_checker = None
    ft_pipe.requires_safety_checker = False
    ft_pipe = ft_pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(42)
    with torch.no_grad():
        ft_image = ft_pipe(
            comparison_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

    # Make side-by-side comparison
    w, h = base_image.size
    comp = Image.new("RGB", (w * 2, h))
    comp.paste(base_image, (0, 0))
    comp.paste(ft_image, (w, 0))

    comp_path = os.path.join(output_dir, "comparison_before_after.png")
    comp.save(comp_path)
    print(f"Comparison image saved to: {comp_path}")
    print("Training and comparison complete!")


# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion on pixel art dataset.")
    parser.add_argument("--train_image_dir", type=str, required=True, help="Path to folder with training images.")
    parser.add_argument("--train_captions_dir", type=str, required=True, help="Path to folder with captions (.txt).")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the fine-tuned model.")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution (square).")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Steps for comparison sampling.")
    parser.add_argument(
        "--comparison_prompt",
        type=str,
        default="pixel art of a knight holding a sword, 32-bit style, vibrant colors",
        help="Prompt used for before/after comparison.",
    )

    args = parser.parse_args()

    train(
        train_image_dir=args.train_image_dir,
        train_captions_dir=args.train_captions_dir,
        output_dir=args.output_dir,
        resolution=args.resolution,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        num_inference_steps=args.num_inference_steps,
        comparison_prompt=args.comparison_prompt,
    )
