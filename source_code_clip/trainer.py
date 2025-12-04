import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm

from config import CONFIG
from text_image_generator import TextImageGenerator  

# -----------------------------
# Data generator (texte + images)
# -----------------------------
train_generator = TextImageGenerator(
    img_list_file="clipEncoder/clip_image_paths.txt",
    emb_file="clipEncoder/clip_text_embs.npy",
    batch_size=CONFIG.get("batch_size", 16),
    img_size=CONFIG["img_size"]
)

# -----------------------------
# Diffusion schedule
# -----------------------------
betas = tf.constant(
    np.linspace(CONFIG["beta_start"], CONFIG["beta_end"], CONFIG["T"], dtype=np.float32)
)
alphas = tf.constant(1.0 - betas.numpy())  
alpha_bars = tf.constant(np.cumprod(alphas.numpy(), axis=0))


def q_sample(x0, t, noise=None):
    """
    q(x_t | x_0) = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps
    """
    if noise is None:
        noise = tf.random.normal(shape=tf.shape(x0), dtype=x0.dtype)

    sqrt_alpha_bar = tf.gather(tf.sqrt(alpha_bars), t)
    sqrt_one_minus = tf.gather(tf.sqrt(1.0 - alpha_bars), t)

    sqrt_alpha_bar = tf.reshape(sqrt_alpha_bar, (-1, 1, 1, 1))
    sqrt_one_minus = tf.reshape(sqrt_one_minus, (-1, 1, 1, 1))

    sqrt_alpha_bar = tf.cast(sqrt_alpha_bar, x0.dtype)
    sqrt_one_minus = tf.cast(sqrt_one_minus, x0.dtype)
    noise = tf.cast(noise, x0.dtype)

    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise


def timestep_embedding(t, dim):
    """
    (Optionnel) Embedding sinusoidal de t, si tu veux l'utiliser ailleurs.
    Ici on ne s'en sert pas directement car le UNet_clip a déjà son Embedding.
    """
    half = dim // 2
    freqs = tf.exp(-np.log(10000.0) * tf.range(0, half, dtype=tf.float32) / half)
    args = tf.cast(tf.expand_dims(t, 1), tf.float32) * tf.expand_dims(freqs, 0)
    emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
    return emb


# -----------------------------
# Difusion loss
# -----------------------------
def diffusion_loss(model, x0, text_emb):
    """
    x0       : (B, H, W, C) images propres
    text_emb : (B, text_dim) embeddings CLIP alignés avec x0
    """
    batch_size = tf.shape(x0)[0]

    t = tf.random.uniform(
        (batch_size,),
        minval=0,
        maxval=CONFIG["T"],
        dtype=tf.int32
    )

    noise = tf.random.normal(shape=tf.shape(x0), dtype=tf.float32)
    x_t = q_sample(x0, t, noise)

    pred_noise = model([x_t, t, text_emb], training=True)
    pred_noise = tf.cast(pred_noise, tf.float32)

    loss = tf.reduce_mean(tf.square(noise - pred_noise))
    return loss


# -----------------------------
# Optimizer
# -----------------------------
optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-5)


# -----------------------------
# Early stopping custom
# -----------------------------
class CustomEarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float("inf")
        self.wait = 0
        self.best_weights = None
    
    def on_epoch_end(self, epoch, loss, model):
        """
        Retourne True si on doit arrêter l'entraînement.
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
            self.best_weights = model.get_weights()
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                if self.best_weights is not None:
                    model.set_weights(self.best_weights)
                return True
        return False


early_stopping = CustomEarlyStopping(patience=5)


# -----------------------------
# train_step 
# -----------------------------
@tf.function
def train_step(x, text_emb, model):
    with tf.GradientTape() as tape:
        loss = diffusion_loss(model, x, text_emb)

    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_norm(g, 1.0) for g in grads if g is not None]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# -----------------------------
# Training loop
# -----------------------------
def train_model(model):
    losses_history = []
    
    # Create checkpoints directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    best_loss = float("inf")

    for epoch in range(CONFIG["epochs"]):
        epoch_losses = []

        progress_bar = tqdm(
            range(len(train_generator)),
            desc=f"Epoch {epoch+1}"
        )

        for batch_idx in progress_bar:
            x_batch, text_emb_batch = train_generator[batch_idx]

            x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
            text_emb_batch = tf.convert_to_tensor(text_emb_batch, dtype=tf.float32)

            loss = train_step(x_batch, text_emb_batch, model)

            epoch_losses.append(loss.numpy())
            progress_bar.set_postfix({
                "loss": f"{loss.numpy():.4f}",
                "lr": f"{optimizer.learning_rate.numpy():.2e}"
            })

        avg_loss = np.mean(epoch_losses)
        losses_history.append(avg_loss)
        print(f"\n Epoch {epoch+1} - Loss: {avg_loss:.4f}")

        # Save model at each epoch
        epoch_model_path = os.path.join(checkpoints_dir, f"model_epoch_{epoch+1:03d}_loss_{avg_loss:.4f}.keras")
        model.save(epoch_model_path)
        print(f"Model saved to: {epoch_model_path}")
        
        # Save best model if this is the best loss so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(base_dir, CONFIG["best_model_name"])
            model.save(best_model_path)
            print(f"Best model updated (loss: {best_loss:.4f})")

        train_generator.on_epoch_end()
        should_stop = early_stopping.on_epoch_end(epoch, avg_loss, model)
        if should_stop:
            print(f"\nTraining stopped early. Final model saved.")
            break

    # Final save (in case training completes normally)
    final_model_path = os.path.join(base_dir, CONFIG["best_model_name"])
    if not os.path.exists(final_model_path):
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
    
    return losses_history
