import math
import torch
import torch.nn as nn

# ----------- sinusoidal embedding & MLP epsilon net ----------------

def sinusoidal_t_embed(t, dim=128):
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=device) * (-math.log(10000.0) / (half-1)))
    angles = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return emb



# ---------------- Residual FiLM-MLP epsilon predictor ----------------


class _FiLMBlock(nn.Module):
    """Single residual block with FiLM modulation.

    Structure: LayerNorm → Linear → SiLU → (scale* + shift) → Linear → residual add.
    Scale/shift are produced outside the block (FiLM)."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.lin1 = nn.Linear(width, width)
        self.lin2 = nn.Linear(width, width)
        self.act = nn.SiLU()

    def forward(self, h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
        residual = h
        h = self.norm(h)
        h = self.lin1(h)
        h = self.act(h)
        # FiLM modulation
        h = h * (1 + gamma) + beta
        h = self.lin2(h)
        return h + residual


class MLPDiffusionEps(nn.Module):
    """Residual MLP ε-network with FiLM time conditioning.

    This follows the recommended design:
      • sinusoidal time embedding → small MLP to `width` dim
      • initial projection of x_t to `width`
      • stack of residual FiLM blocks (4-10)
      • per-block scale/shift produced from time embedding.
    """

    def __init__(self, d_in: int, width: int = 512, n_layers: int = 8, t_dim: int = 128,
                 dropout: float = 0.0):
        if n_layers < 2:
            raise ValueError("n_layers must be at least 2")

        super().__init__()

        # ---- Time embedding → conditioning vector ----
        self.t_proj = nn.Sequential(
            nn.Linear(t_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width)
        )

        # ---- Input projection for x_t ----
        self.x_proj = nn.Linear(d_in, width)

        # ---- Residual FiLM blocks ----
        self.blocks = nn.ModuleList([_FiLMBlock(width) for _ in range(n_layers)])
        # Each block has its own FiLM generator
        self.film_generators = nn.ModuleList([
            nn.Linear(width, 2 * width) for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ---- Output projection ----
        self.out_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, d_in)

    # -----------------------------------------------------------------
    def forward(self, x_t: torch.Tensor, t_frac: torch.Tensor):
        """Predict ε given noisy sample x_t and scaled timestep t_frac∈[0,1]."""

        # Time conditioning vector (B,width)
        t_raw = sinusoidal_t_embed(t_frac, self.t_proj[0].in_features)
        cond = self.t_proj(t_raw)

        # Project input
        h = self.x_proj(x_t)

        # Stacked FiLM residual blocks
        for block, film_gen in zip(self.blocks, self.film_generators):
            gamma_beta = film_gen(cond)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            h = block(h, gamma, beta)
            h = self.dropout(h)

        h = self.out_norm(h)
        return self.out_proj(h)


# ------------------------------------------------------------------
#                           Transformer ε-net
# ------------------------------------------------------------------


class TransformerDiffusionEps(nn.Module):
    """Transformer-based epsilon predictor (more expressive than simple MLP).

    Parameters
    ----------
    d_in : int
        Input latent dimension (θ feature dim).
    width : int, default 256
        Model/attention width.
    n_layers : int, default 4
        Number of TransformerEncoder layers.
    n_head : int, default 8
        Attention heads per layer (width must be divisible by n_head).
    t_dim : int, default 128
        Raw sinusoidal embedding dimension for time-step.
    """

    def __init__(self, d_in: int, width: int = 256, n_layers: int = 4, n_head: int = 8, t_dim: int = 128):
        super().__init__()

        if width % n_head != 0:
            raise ValueError("width must be divisible by n_head for multihead attention")

        # --- Input projections ---
        self.x_proj = nn.Linear(d_in, width)
        self.t_proj = nn.Linear(t_dim, width)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=n_head,
            dim_feedforward=width * 4,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Output projection ---
        self.out_proj = nn.Linear(width, d_in)

    def forward(self, x_t: torch.Tensor, t_frac: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        x_t : (B, D) tensor
            Noisy latent vector at diffusion step t.
        t_frac : (B,) tensor
            Time fraction t/T in [0,1]. Same dtype/device as x_t.
        """
        # Embed time and project inputs
        t_emb = sinusoidal_t_embed(t_frac, self.t_proj.in_features)  # (B, t_dim)
        h = self.x_proj(x_t) + self.t_proj(t_emb)  # (B, width)

        # The transformer expects sequence; use length-1 sequence
        h = self.encoder(h.unsqueeze(1)).squeeze(1)  # (B, width)

        return self.out_proj(h)
