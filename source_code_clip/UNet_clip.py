"""
UNet_clip.py

U-Net de diffusion conditionné par :
  - le timestep (comme dans ton modèle actuel)
  - un embedding texte (par ex. venant de CLIP)

Usage typique :

    from UNet_clip import build_unet_clip

    model = build_unet_clip(
        img_size=128,
        channels=3,
        time_dim=256,
        text_dim=512,       # dimension de ton embedding CLIP
        max_timesteps=1000  # T dans ta diffusion
    )

    # Appel :
    # images : (B, 128, 128, 3)   dans [-1, 1]
    # t      : (B,) int32         timesteps aléatoires
    # text_emb : (B, 512)         embeddings CLIP
    #
    # pred_noise = model([images_noisy, t, text_emb])
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# -------------------------------------------------------------------
# Blocs de base : conv + GroupNorm + injection de condition (temps+texte)
# -------------------------------------------------------------------

def conv_block_cond(x, filters, cond_emb, kernel_size=3):
    """
    Bloc convolutionnel avec :
      - Conv2D + GroupNorm
      - Injection de l'embedding cond_emb (temps + texte) via une Dense
      - Activation swish

    x        : (B, H, W, C)
    cond_emb : (B, cond_dim)  (déjà fusion temps + texte)
    """
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.GroupNormalization(groups=8)(x)

    # cond_emb -> proj vers "filters", puis broadcast spatialement
    t = layers.Dense(filters)(cond_emb)          # (B, filters)
    t = layers.Reshape((1, 1, filters))(t)       # (B, 1, 1, filters)
    x = x + t

    x = layers.Activation('swish')(x)
    return x


def encoder_block_cond(x, filters, cond_emb):
    """
    Deux conv_block_cond + MaxPool (downsampling)
    Retourne :
      - features pour le skip connection
      - features downsamplés
    """
    x = conv_block_cond(x, filters, cond_emb)
    x = conv_block_cond(x, filters, cond_emb)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p


def decoder_block_cond(x, skip_features, filters, cond_emb):
    """
    Up-convolution + concat avec les features de l'encodeur + conv_block_cond
    """
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block_cond(x, filters, cond_emb)
    return x


# -------------------------------------------------------------------
# UNet complet conditionné par :
#   - un embedding de temps (timestep)
#   - un embedding texte (par ex. CLIP)
# -------------------------------------------------------------------

def build_unet_clip(
    img_size,
    channels=3,
    time_dim=256,
    text_dim=512,
    max_timesteps=1000,
):
    """
    Construit un U-Net de diffusion conditionné par :
      - t (timestep, int32)
      - text_emb (embedding texte, ex CLIP)

    Paramètres
    ----------
    img_size : int
        Taille des images (img_size x img_size).
    channels : int
        Nombre de canaux (3 pour RGB).
    time_dim : int
        Dimension de l'embedding de temps (et de la fusion temps+texte).
    text_dim : int
        Dimension de l'embedding texte d'entrée (CLIP, etc.).
    max_timesteps : int
        Nombre maximum de timesteps (T) utilisé dans la diffusion.
        Doit être >= au max des t que tu échantillonnes pendant l'entraînement.

    Retour
    ------
    model : tf.keras.Model
        Modèle Keras qui prend :
          [img, t, text_emb] -> pred_noise
    """

    # -------------------------
    # Inputs
    # -------------------------
    img_input = layers.Input(
        shape=(img_size, img_size, channels),
        name='image'
    )  # (B, H, W, C)

    time_input = layers.Input(
        shape=(),
        dtype=tf.int32,
        name='timestep'
    )  # (B,)

    text_input = layers.Input(
        shape=(text_dim,),
        dtype=tf.float32,
        name='text_emb'
    )  # (B, text_dim)

    # -------------------------
    # Embedding de temps
    # -------------------------
    # Embedding discrète des timesteps (0 .. max_timesteps-1)
    # Input shape: (B,) -> Output shape: (B, time_dim)
    time_emb = layers.Embedding(
        input_dim=max_timesteps,
        output_dim=time_dim
    )(time_input)              # (B, time_dim)

    time_emb = layers.Dense(time_dim, activation='swish')(time_emb)
    time_emb = layers.Dense(time_dim, activation='swish')(time_emb)
    # No squeeze needed - Embedding already produces (B, time_dim) from (B,) input

    # -------------------------
    # Embedding texte -> projection dans le même espace
    # -------------------------
    text_proj = layers.Dense(time_dim, activation='swish')(text_input)   # (B, time_dim)

    # -------------------------
    # Fusion temps + texte
    # -------------------------
    cond_emb = layers.Add()([time_emb, text_proj])                       # (B, time_dim)

    # -------------------------
    # U-Net Encoder
    # -------------------------
    c1, p1 = encoder_block_cond(img_input,  64, cond_emb)   # 128 -> 64
    c2, p2 = encoder_block_cond(p1,        128, cond_emb)   # 64  -> 32
    c3, p3 = encoder_block_cond(p2,        256, cond_emb)   # 32  -> 16
    c4, p4 = encoder_block_cond(p3,        512, cond_emb)   # 16  -> 8

    # -------------------------
    # Bottleneck
    # -------------------------
    b = conv_block_cond(p4, 1024, cond_emb)
    b = conv_block_cond(b,  1024, cond_emb)

    # -------------------------
    # U-Net Decoder
    # -------------------------
    d4 = decoder_block_cond(b,  c4, 512, cond_emb)   # 8  -> 16
    d3 = decoder_block_cond(d4, c3, 256, cond_emb)   # 16 -> 32
    d2 = decoder_block_cond(d3, c2, 128, cond_emb)   # 32 -> 64
    d1 = decoder_block_cond(d2, c1,  64, cond_emb)   # 64 -> 128

    # -------------------------
    # Sortie : prédiction du bruit (mêmes channels que l'image)
    # -------------------------
    output = layers.Conv2D(
        channels,
        (1, 1),
        activation='linear',
        name='pred_noise'
    )(d1)

    model = Model(
        inputs=[img_input, time_input, text_input],
        outputs=output,
        name='UNet_Diffusion_ClipCond'
    )

    return model


# -------------------------------------------------------------------
# Petit test rapide (optionnel)
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Exemple de construction du modèle
    IMG_SIZE = 128
    TIME_DIM = 256
    TEXT_DIM = 512
    T = 1000

    model = build_unet_clip(
        img_size=IMG_SIZE,
        channels=3,
        time_dim=TIME_DIM,
        text_dim=TEXT_DIM,
        max_timesteps=T
    )

    model.summary()

    # Test d'un forward pass dummy
    import numpy as np

    batch_size = 2
    dummy_imgs = np.random.randn(batch_size, IMG_SIZE, IMG_SIZE, 3).astype("float32")
    dummy_t = np.random.randint(0, T, size=(batch_size,), dtype="int32")
    dummy_text = np.random.randn(batch_size, TEXT_DIM).astype("float32")

    out = model([dummy_imgs, dummy_t, dummy_text])
    print("Output shape:", out.shape)
