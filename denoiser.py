import torch
import torch.nn as nn
import math


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, frequency_embedding_size=256, max_period=10000):
        super().__init__()
        assert frequency_embedding_size % 2 == 0
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True)
        )  
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        args = t[:, None].float() * self.freqs
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        t_emb = self.mlp(t_freq)
        return t_emb


class OneDimCNN(nn.Module):
    def __init__(self, layer_channels: list, model_dim: int, kernel_size: int,
                 proj_dim: int | None = None, proj_layers: int = 1):
        """1-D U-Net-style denoiser with optional linear dimensionality reduction.

        Parameters
        ----------
        layer_channels : list[int]
            Channel sizes for encoder+decoder path (first and last must be 1).
        model_dim : int
            Hidden dimension for timestep embedding.
        kernel_size : int
            Convolution kernel size (odd value recommended).
        proj_dim : int or None
            If set (< input dimension), apply Linear(in_dim → proj_dim) before CNN and
            Linear(proj_dim → in_dim) after CNN (weights learned jointly).
        proj_layers : int, default 1
            If >1, stack this many Linear/ReLU layers in projection blocks.
        """

        super().__init__()

        self.proj_dim = proj_dim
        self.proj_layers = proj_layers

        # Will initialize projection lazily when first input arrives (knowing D)
        self.in_proj = None
        self.out_proj = None

        self.time_embedder = TimestepEmbedder(hidden_dim=model_dim)

        self.encoder_list = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(layer_channels[i], layer_channels[i+1], kernel_size, 1, kernel_size // 2),
                nn.Sequential(nn.BatchNorm1d(layer_channels[i+1]), nn.ELU())
            ]) for i in range(len(layer_channels) // 2 + 1)
        ])

        self.decoder_list = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(layer_channels[i], layer_channels[i+1], kernel_size, 1, kernel_size // 2),
                nn.Sequential(nn.BatchNorm1d(layer_channels[i+1]), nn.ELU()) if layer_channels[i+1] != 1 else nn.Identity()
            ]) for i in range(len(layer_channels)//2 +1 , len(layer_channels)-1)
        ])

    # ------------------------------------------------------------------
    def _build_proj(self, in_dim: int, device):
        """Create projection blocks once we know input dimension."""
        if self.proj_dim is None or self.proj_dim >= in_dim:
            return  # no projection needed

        layers_in = []
        dim_curr = in_dim
        for l in range(self.proj_layers):
            dim_next = self.proj_dim if l == self.proj_layers-1 else max(self.proj_dim*2, 128)
            layers_in.append(nn.Linear(dim_curr, dim_next))
            if l != self.proj_layers-1:
                layers_in.append(nn.ReLU())
            dim_curr = dim_next
        self.in_proj = nn.Sequential(*layers_in).to(device)

        layers_out = []
        dim_curr = self.proj_dim
        for l in range(self.proj_layers):
            dim_next = in_dim if l == self.proj_layers-1 else max(self.proj_dim*2, 128)
            layers_out.append(nn.Linear(dim_curr, dim_next))
            if l != self.proj_layers-1:
                layers_out.append(nn.ReLU())
            dim_curr = dim_next
        self.out_proj = nn.Sequential(*layers_out).to(device)

    # ------------------------------------------------------------------
    def forward(self, x, t, c=0.):
        # x shape (B, D)
        B, D = x.shape

        if self.in_proj is None and self.proj_dim is not None and self.proj_dim < D:
            self._build_proj(D, x.device)

        if self.in_proj is not None:
            x = self.in_proj(x)
        D_red = x.shape[1]

        # CNN expects (B,1,L)
        x = x.unsqueeze(1)  # (B,1,D_red)
        t_emb = self.time_embedder(t).unsqueeze(1)  # (B,1,model_dim)

        x_list = []
        for i, (conv, act) in enumerate(self.encoder_list):
            x = conv(x + t_emb)
            x = act(x)
            if i < len(self.encoder_list) - 2:
                x_list.append(x)
        for i, (conv, act) in enumerate(self.decoder_list):
            x = x + x_list[-i-1]
            x = conv(x + t_emb)
            x = act(x)

        x = x.squeeze(1)  # (B,D_red)

        if self.out_proj is not None:
            x = self.out_proj(x)
        return x