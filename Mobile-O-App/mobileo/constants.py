DEFAULT_IMAGE_TOKEN = "<image>"
UND_IMAGE_TOKEN_IDX = -200

# DiT / SANA diffusion constants
DIT_LATENT_CHANNELS = 32       # DC-AE latent channels (not 4 — SANA uses 32)
DIT_LATENT_SIZE = 16           # 512px / 32 = 16 spatial size
DIT_TEXT_HIDDEN_SIZE = 2304    # Connector output dim = DiT cross-attention dim

# VAE constants
VAE_IMAGE_SIZE = 512           # Output image resolution
VAE_LATENT_SCALE = 32          # VAE downscale factor (512 / 32 = 16 latent size)

# Sequence length constants (from iOS ConditioningConnectorConfiguration)
SEQ_LEN_DEFAULT = 77           # Default / minimum sequence length (pad target)
SEQ_LEN_MIN = 77
SEQ_LEN_MAX = 512
