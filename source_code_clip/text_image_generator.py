import numpy as np
import cv2
import os
from tensorflow.keras.utils import Sequence

class TextImageGenerator(Sequence):
    def __init__(self, img_list_file, emb_file, batch_size=16, img_size=128):
        # Get the base directory (project root)
        # Assuming we're running from source_code_clip, go up one level
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Resolve paths relative to project root
        if not os.path.isabs(img_list_file):
            img_list_file = os.path.join(base_dir, img_list_file)
        if not os.path.isabs(emb_file):
            emb_file = os.path.join(base_dir, emb_file)
        
        # Charger la liste des images, dans l'ordre généré par clip_encode_labels.py
        with open(img_list_file, "r") as f:
            img_paths = [l.strip() for l in f.readlines()]
        
        # Resolve image paths relative to project root
        self.img_paths = []
        for p in img_paths:
            if not os.path.isabs(p):
                # If path is relative, it should be relative to project root
                full_path = os.path.join(base_dir, p)
                # Handle legacy paths that might be "train_images/..." instead of "dataset/train_images/..."
                if not os.path.exists(full_path) and p.startswith("train_images/"):
                    # Try with dataset/ prefix
                    alt_path = os.path.join(base_dir, "dataset", p)
                    if os.path.exists(alt_path):
                        full_path = alt_path
            else:
                full_path = p
            if os.path.exists(full_path):
                self.img_paths.append(full_path)
            else:
                print(f"[WARN] Image not found: {full_path}")

        # Charger les embeddings CLIP
        self.embs = np.load(emb_file)  # shape (N, 512)

        self.batch_size = batch_size
        self.img_size = img_size
        self.N = len(self.img_paths)
        
        # Ensure embeddings match number of images
        if len(self.embs) != self.N:
            print(f"[WARN] Mismatch: {len(self.embs)} embeddings but {self.N} images. Using min.")
            min_len = min(len(self.embs), self.N)
            self.embs = self.embs[:min_len]
            self.img_paths = self.img_paths[:min_len]
            self.N = min_len

    def __len__(self):
        return (self.N + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.N)
        
        # Récupère les chemins d'images pour ce batch
        batch_img_paths = self.img_paths[start_idx:end_idx]

        # Embeddings texte correspondants
        batch_embs = self.embs[start_idx:end_idx]

        images = []
        for p in batch_img_paths:
            if not os.path.exists(p):
                print(f"[ERROR] Image not found: {p}")
                # Create a black image as fallback
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            else:
                img = cv2.imread(p)
                if img is None:
                    print(f"[ERROR] Failed to load image: {p}")
                    img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    img = img.astype("float32") / 127.5 - 1.0  # normalisation [-1,1]
            images.append(img)

        images = np.array(images)
        text_embs = batch_embs.astype("float32")

        return images, text_embs
    
    def on_epoch_end(self):
        # Shuffle the data at the end of each epoch
        indices = np.arange(self.N)
        np.random.shuffle(indices)
        self.img_paths = [self.img_paths[i] for i in indices]
        self.embs = self.embs[indices]
