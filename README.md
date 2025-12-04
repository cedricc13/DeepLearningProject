# Pixel Art Generation – Stable Diffusion, Fine-Tuning, CLIP Encoders & Custom Model

This repository contains multiple sub-projects exploring **pixel-art generation** using:
- Stable Diffusion  
- Fine-tuning approaches  
- Custom UNet + CLIP architectures  
- Text-embedding preprocessing (CLIP encoder)

Before running anything, install the dependencies:

```bash
pip install -r requirements.txt
```

---

# 📁 Repository Structure

---

## **1. `already_trained_model/` – Pixel Art Generation From a Pre-Trained Stable Diffusion Model**

Use an **already trained Stable Diffusion** model to generate pixel-art–style images.

Run:

```bash
python already_trained_model/already_trained_test/generate_pixel_art.py
```

Output is saved in:

```
already_trained_model/already_trained_test/output.png
```

---

## **2. `pixelart_finetune/` – Fine-Tuning Stable Diffusion on Pixel-Art Data**

Fine-tune Stable Diffusion on a custom pixel-art dataset.

Run training:

```bash
python pixelart_finetune/train_fine_tune.py
```

Comparison image:

```
pixelart_finetune/comparison_before_after.png
```

---

## **3. `source_code_clip/` – Custom UNet + CLIP Text Encoder (Training From Scratch)**

Train your own diffusion pipeline with a UNet + CLIP text embeddings.

Run:

```bash
python source_code_clip/main.py
```

CLI options:

```
--dataset_path       
--epochs             
--batch_size         
--lr                 
--img_size           
--save_every         
--output_dir         
--use_gpu            
```

---

## **4. `clipEncoder/` – Generate CLIP Text Embeddings for Captions**

This folder contains the script:

```
clipEncoder/clip_encode_labels.py
```

### What this script does

It:

1. Loads images from:
   ```
   dataset/train_images/
   ```
2. Loads matching captions from:
   ```
   dataset/train_images_captions/
   ```
3. Generates CLIP text embeddings  
4. Saves:
   - `clip_text_embs.npy`  
   - `clip_image_paths.txt`

### How to run it

```bash
python clipEncoder/clip_encode_labels.py
```

---

# Dataset

Direct download:

https://huggingface.co/datasets/ayhantasyurt/pixel-art-2d-game-character-sprites-idle/resolve/main/pixel-art-2dgame-charecter-sprites-idle.zip

---

# Installation

```bash
pip install -r requirements.txt
```

