"""
Generate pixel art images using Stable Diffusion from Hugging Face.
Uses the diffusers library with PyTorch.
"""

import os

# Disable torch._dynamo to avoid potential compatibility issues
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def get_device() -> str:
    """Return the best available device: cuda, mps (Apple), or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon / Metal
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_image(prompt: str, output_path: str, seed: int = 42):
    """
    Generate a pixel art image from a text prompt using Stable Diffusion.
    
    Args:
        prompt: Text description of the image to generate.
        output_path: Path where the generated image will be saved.
        seed: Random seed for reproducibility.
    """
    print("Loading Stable Diffusion model...")

    device = get_device()
    print(f"Using device: {device}")

    # Choose dtype depending on device
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Load the Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch_dtype,    # <-- important: torch_dtype, not dtype
    )

    # (Optional) disable safety checker to avoid unexpected black/blurred images
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    pipe = pipe.to(device)

    # Optimize for inference
    if device == "cuda":
        pipe.enable_attention_slicing()

    # Set a deterministic generator
    generator = torch.Generator(device=device).manual_seed(seed)

    print(f"Generating image with prompt: '{prompt}'")
    print("This may take a little while...")

    # Run the pipeline
    if device in ("cuda", "mps"):
        # autocast for performance on GPU / MPS
        autocast_device = "cuda" if device == "cuda" else "mps"
        with torch.autocast(autocast_device):
            image = pipe(
                prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]
    else:
        # CPU – no autocast
        image = pipe(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

    # Save the image
    image.save(output_path)
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    # Example: Generate pixel art of a knight holding a sword
    prompt = "pixel art of a knight holding a sword, 32-bit style, vibrant colors"

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "output.png")

    try:
        generate_image(prompt, output_path)
        print("\n✓ Generation completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        print("\nMake sure you have installed the required dependencies:")
        print("pip install diffusers transformers accelerate torch torchvision")
