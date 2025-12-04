import numpy as np
import tensorflow as tf
import cv2

from config import CONFIG

# -----------------------------------------------------
# Diffusion schedule (must match training exactly)
# -----------------------------------------------------
betas = tf.constant(
    np.linspace(CONFIG["beta_start"], CONFIG["beta_end"], CONFIG["T"], dtype=np.float32)
)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas.numpy(), axis=0)

betas = tf.constant(betas, dtype=tf.float32)
alphas = tf.constant(alphas, dtype=tf.float32)
alpha_bars = tf.constant(alpha_bars, dtype=tf.float32)


# -----------------------------------------------------
# Reverse diffusion: one step x_t --> x_(t-1)
# -----------------------------------------------------
def p_sample(model, x_t, t_scalar, text_emb):
    """
    Reverse diffusion step:
        p(x_{t-1} | x_t)

    model      : your UNet conditioned on text
    x_t        : noisy image at timestep t
    t_scalar   : integer timestep (0 ... T-1)
    text_emb   : CLIP embedding (1, text_dim)
    """

    # Convert scalar t → tensor batch of shape (1,)
    t_tensor = tf.constant([t_scalar], dtype=tf.int32)

    # Predict noise using the trained UNet
    pred_noise = model([x_t, t_tensor, text_emb], training=False)

    beta_t      = betas[t_scalar]
    alpha_t     = alphas[t_scalar]
    alpha_bar_t = alpha_bars[t_scalar]

    # Predict x0 from model output
    x0_pred = (x_t - tf.sqrt(1 - alpha_bar_t) * pred_noise) / tf.sqrt(alpha_bar_t)

    # Compute the mean of posterior distribution q(x_{t-1} | x_t, x0)
    # Using DDPM formula: mean = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * pred_noise)
    if t_scalar > 0:
        alpha_bar_prev = alpha_bars[t_scalar - 1]
        # Standard DDPM sampling formula
        mean = (1.0 / tf.sqrt(alpha_t)) * (x_t - (beta_t / tf.sqrt(1 - alpha_bar_t)) * pred_noise)
        # Add noise
        noise = tf.random.normal(shape=tf.shape(x_t))
        # Variance: beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        x_prev = mean + tf.sqrt(variance) * noise
    else:
        # Final step: no noise
        x_prev = x0_pred

    return x_prev


# -----------------------------------------------------
# Sampling full diffusion trajectory
# -----------------------------------------------------
def sample(model, text_emb, img_size):
    """
    Generates one image given a CLIP text embedding.
    
    model     : trained UNet
    text_emb  : CLIP embedding (1, text_dim)
    img_size  : final image size (H = W = img_size)
    """

    # Start from pure Gaussian noise
    x = tf.random.normal((1, img_size, img_size, 3), dtype=tf.float32)

    # Go backward through the diffusion steps
    for t in range(CONFIG["T"] - 1, -1, -1):
        x = p_sample(model, x, t, text_emb)

    return x


# -----------------------------------------------------
# Utility: convert [-1, 1] tensor → OpenCV image
# -----------------------------------------------------
def postprocess_img(x):
    """
    Convert tensor in [-1, 1] to uint8 BGR for cv2.imwrite.
    Input shape: (1, H, W, 3)
    """

    x = tf.clip_by_value(x, -1.0, 1.0)
    x = (x + 1.0) * 127.5
    img = tf.cast(x[0], tf.uint8).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# -----------------------------------------------------
# Main helpers called from main.py
# -----------------------------------------------------
def generate_single_image(model, img_size, text_emb=None):
    """
    Convenience function:
        - sample an image with a CLIP embedding
        - return OpenCV-ready uint8 image
    
    If text_emb is None → use a dummy zero embedding.
    """

    if text_emb is None:
        # dummy embedding (works only for debugging)
        text_emb = tf.zeros((1, CONFIG["text_dim"]), dtype=tf.float32)

    x = sample(model, text_emb, img_size)
    return postprocess_img(x)


def visualize_diffusion_process(model, img_size, text_emb=None, steps_to_show=10):
    """
    Generates multiple intermediate diffusion states
    to visualize how noise gradually becomes an image.

    Returns a list of OpenCV images.
    """

    if text_emb is None:
        text_emb = tf.zeros((1, CONFIG["text_dim"]), dtype=tf.float32)

    # Start from initial noise
    x = tf.random.normal((1, img_size, img_size, 3), dtype=tf.float32)

    images = []

    # Select evenly spaced timesteps
    step_indices = np.linspace(CONFIG["T"] - 1, 0, steps_to_show).astype(int)

    for t in step_indices:
        x = p_sample(model, x, int(t), text_emb)
        images.append(postprocess_img(x))

    return images
