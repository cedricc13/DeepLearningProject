# Pixel Art Generation – Stable Diffusion, Fine-Tuning & Custom Model

This repository contains multiple sub-projects exploring pixel-art generation using different approaches with Stable Diffusion and custom architectures.  
Please make sure to **install all dependencies first**:

```bash
pip install -r requirements.txt
```

## 📁 Repository Structure

### **1. `already_trained_model/` – Pixel Art with a Pre-Trained Stable Diffusion Model**

In this folder, we take an **already trained Stable Diffusion model** and use it to generate **pixel-art-style images**.

Run:

```bash
python already_trained_model/already_trained_test/generate_pixel_art.py
```

### **2. `pixelart_finetune/` – Fine-Tuning Stable Diffusion on a Pixel-Art Dataset**

Fine-tune a Stable Diffusion model on pixel-art dataset.

Run:

```bash
python pixelart_finetune/train_fine_tune.py
```

### **3. `source_code_clip/` – Training a Model From Scratch (UNet + CLIP Encoder)**

Run:

```bash
python source_code_clip/main.py
```

CLI options:

```
parser.add_argument("--dataset_path", type=str, default="data/")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--img_size", type=int, default=64)
parser.add_argument("--save_every", type=int, default=5)
parser.add_argument("--output_dir", type=str, default="outputs/")
parser.add_argument("--use_gpu", action="store_true")
```

### Dataset

https://www.kaggle.com/datasets/ayhantasyurt/pixel-art-2dgame-charecter-sprites-idle
