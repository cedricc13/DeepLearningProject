import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5" 

OUTPUT_DIR = "outputs_knight"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def pixelize(image: Image.Image, sprite_size=(32, 32)) -> Image.Image:
    """
    1) Downscale to the sprite size.  
    2) Upscale to the original size using NEAREST to preserve "big pixels".
    """
    w, h = image.size
    small = image.resize(sprite_size, Image.NEAREST)
    big = small.resize((w, h), Image.NEAREST)
    return big


def quantize_to_palette(image: Image.Image, n_colors: int = 16) -> Image.Image:
    """
    Reduce the image to n_colors using PIL quantization,  
    then convert back to RGB for compatibility with most tools.
    """
    pal_img = image.convert("P", palette=Image.ADAPTIVE, colors=n_colors)
    return pal_img.convert("RGB")


def load_base_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    pipe = pipe.to(DEVICE)
    return pipe


def generate_high_res(pipe, prompt: str, height=768, width=768, steps=30, cfg=7.5, generator=None):
    """
    Generate the high-resolution "continuous" image from the prompt.
    """
    out = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        height=height,
        width=width,
        generator=generator,
    )
    return out.images[0]


def hi_res_to_pixel_art(image: Image.Image,
                        sprite_size=(32, 32),
                        palette_colors=16) -> Image.Image:
    """
    Convert the high-resolution image to pixel art:
    - enforce a sprite resolution,
    - pixelize,
    - reduce the color palette.
    """
    pix = pixelize(image, sprite_size=sprite_size)
    pix = quantize_to_palette(pix, n_colors=palette_colors)
    return pix

# Create 20 images to keep only the best one

def main():
    prompt = (
        "single tiny knight character, centered, full body, front view, takes almost all the frame, simple armor, clear silhouette, no other characters, plain solid background color, flat lighting, minimal details, high contrast between character and background, illustration style"
    )

    pipe = load_base_pipeline()

    sprite_size = (32, 32)   
    palette_colors = 25    
    n_images = 20

    for i in range(n_images):
        gen = torch.Generator(device=DEVICE)
        gen = gen.manual_seed(i)  

        hi_res = generate_high_res(
            pipe,
            prompt,
            height=768,
            width=768,
            steps=30,
            cfg=7.5,
            generator=gen,
        )

        hi_name = os.path.join(OUTPUT_DIR, f"knight_hi_res_{i:02d}.png")
        hi_res.save(hi_name)
        print(f"[{i+1}/{n_images}] Image haute résolution : {hi_name}")

        pixel_art = hi_res_to_pixel_art(
            hi_res,
            sprite_size=sprite_size,
            palette_colors=palette_colors,
        )

        pix_name = os.path.join(OUTPUT_DIR, f"knight_pixel_art_{i:02d}.png")
        pixel_art.save(pix_name)
        print(f"[{i+1}/{n_images}] Image pixelisée : {pix_name}")

if __name__ == "__main__":
    main()
