import os
import argparse

# 1) GPU setup MUST happen before importing TensorFlow
from gpu_setup import setup_tensorflow_gpu
setup_tensorflow_gpu()

import cv2
import tensorflow as tf
from tensorflow import keras
import sys
import types

# -------------------------------------------------------------------
# Shim de compatibilité pour 'keras.src.engine.functional.Functional'
# -------------------------------------------------------------------
try:
    # 1) s'assurer qu'un package 'keras.src' existe dans sys.modules
    if "keras.src" not in sys.modules:
        keras_src = types.ModuleType("keras.src")
        sys.modules["keras.src"] = keras_src
    else:
        keras_src = sys.modules["keras.src"]

    # 2) créer un sous-module 'keras.src.engine' s'il n'existe pas
    if "keras.src.engine" not in sys.modules:
        engine_mod = types.ModuleType("keras.src.engine")
        sys.modules["keras.src.engine"] = engine_mod
    else:
        engine_mod = sys.modules["keras.src.engine"]

    # 3) créer le sous-module 'keras.src.engine.functional'
    functional_mod = types.ModuleType("keras.src.engine.functional")

    # Essayer de récupérer une vraie classe Functional depuis keras.models
    FunctionalClass = None
    try:
        # si Keras expose vraiment Functional
        from keras.models import Functional as RealFunctional
        FunctionalClass = RealFunctional
        print("[KERAS] Using keras.models.Functional for shim.")
    except Exception:
        # fallback : utiliser keras.Model comme classe de base
        from keras import Model as RealFunctional
        FunctionalClass = RealFunctional
        print("[KERAS] Using keras.Model as Functional shim.")

    # Injecter la classe dans le module fictif
    functional_mod.Functional = FunctionalClass

    # Enregistrer le module fictif dans sys.modules
    sys.modules["keras.src.engine.functional"] = functional_mod

    print("[KERAS] Shim installed for keras.src.engine.functional")
except Exception as e:
    print("[KERAS] Warning: could not create keras.src shim:", e)

from config import CONFIG
from sampling import generate_single_image, visualize_diffusion_process

# -------------------------------------------------------------------
# CLIP text encoder backend selection (ONLY transformers + PyTorch)
# -------------------------------------------------------------------
USE_SENTENCE_TRANSFORMERS = False
USE_TRANSFORMERS = True

_CLIP_TOKENIZER = None
_CLIP_TEXT_MODEL = None

from transformers import CLIPTokenizer, CLIPTextModel
import torch

print("[IMPORT] Loaded transformers (PyTorch CLIP backend).")


def load_clip_text_model(model_name: str = "openai/clip-vit-base-patch32"):
    """
    Lazily load the CLIP text model / encoder.

    Returns:
        (tokenizer, text_model) tuple for transformers backend.
    """
    global _CLIP_TOKENIZER, _CLIP_TEXT_MODEL

    if _CLIP_TOKENIZER is not None and _CLIP_TEXT_MODEL is not None:
        return _CLIP_TOKENIZER, _CLIP_TEXT_MODEL

    print("[CLIP] Loading transformers CLIP text model:", model_name)
    _CLIP_TOKENIZER = CLIPTokenizer.from_pretrained(model_name)
    _CLIP_TEXT_MODEL = CLIPTextModel.from_pretrained(model_name)
    return _CLIP_TOKENIZER, _CLIP_TEXT_MODEL


def encode_prompt_to_tf(prompt: str, model_name: str = "openai/clip-vit-base-patch32") -> tf.Tensor:
    """
    Encode a text prompt into a CLIP embedding and return it as a TensorFlow tensor.

    Returns:
        tf.Tensor of shape (1, text_dim), typically (1, 512) for CLIP ViT-B/32.
    """
    tokenizer, text_model = load_clip_text_model(model_name)
    text_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_model = text_model.to(device)

    with torch.no_grad():
        inputs = tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = text_model(**inputs)
        pooled = outputs.pooler_output  # shape (1, 512)
        emb_np = pooled.detach().cpu().numpy().astype("float32")
        text_emb_tf = tf.convert_to_tensor(emb_np, dtype=tf.float32)
        return text_emb_tf


def generate_image_from_prompt(
    prompt: str,
    output_path: str = "sample.png",
    visualize_process: bool = False,
    viz_steps: int = 10,
):
    """
    High-level function to:
        - load the trained UNet+CLIP diffusion model,
        - encode the text prompt,
        - generate an image,
        - optionally save intermediate diffusion snapshots.

    Args:
        prompt: Text prompt used to condition the diffusion model.
        output_path: Path where the final image will be saved.
        visualize_process: If True, also save intermediate diffusion steps.
        viz_steps: Number of steps to visualize if visualize_process is True.
    """
    # 1) Check model path
    model_path = CONFIG["best_model_name"]
    img_size = CONFIG["img_size"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Make sure training has saved the model and that "
            "CONFIG['best_model_name'] points to the correct .keras file."
        )

    print(f"[GEN] Loading trained UNet_clip model from: {model_path}")
    unet_model = keras.saving.load_model(
    model_path,
    compile=False,
    safe_mode=False,  # on autorise le chargement même si les chemins de modules sont un peu exotiques
)

    print("[GEN] Model loaded.")

    # 2) Encode the text prompt
    print(f"[GEN] Encoding prompt: \"{prompt}\"")
    text_emb = encode_prompt_to_tf(prompt)  # shape (1, 512)

    # 3) Run diffusion sampling
    print("[GEN] Running diffusion sampling...")
    img = generate_single_image(
        unet_model,
        img_size=img_size,
        text_emb=text_emb
    )

    # 4) Save the final image
    cv2.imwrite(output_path, img)
    print(f"[GEN] Image saved to: {output_path}")

    # 5) Optionally: visualize the diffusion process
    if visualize_process:
        print(f"[GEN] Visualizing diffusion process with {viz_steps} snapshots...")
        viz_images = visualize_diffusion_process(
            unet_model,
            img_size=img_size,
            text_emb=text_emb,
            steps_to_show=viz_steps
        )
        viz_dir = "diffusion_viz"
        os.makedirs(viz_dir, exist_ok=True)
        for i, im in enumerate(viz_images):
            path = os.path.join(viz_dir, f"step_{i:02d}.png")
            cv2.imwrite(path, im)
        print(f"[GEN] Saved {len(viz_images)} diffusion steps in folder: {viz_dir}")


def parse_args():
    """
    Parse command-line arguments for the CLI.
    """
    parser = argparse.ArgumentParser(
        description="Generate an image from a text prompt using a trained UNet+CLIP diffusion model."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt used to generate the image."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sample.png",
        help="Path where the generated image will be saved (default: sample.png)."
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="If set, also save intermediate diffusion images (reverse process visualization)."
    )
    parser.add_argument(
        "--viz_steps",
        type=int,
        default=10,
        help="Number of diffusion steps to visualize if --viz is used (default: 10)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    generate_image_from_prompt(
        prompt=args.prompt,
        output_path=args.output,
        visualize_process=args.viz,
        viz_steps=args.viz_steps,
    )
