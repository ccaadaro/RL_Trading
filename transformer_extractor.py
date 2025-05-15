# transformer_extractor.py

import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TransformerFeatures(BaseFeaturesExtractor):
    """
    Extrae secuencias de shape (seq_len, n_feats) con un TransformerEncoder
    y devuelve un feature vector de tamaño embed_dim.
    """
    def __init__(
        self,
        observation_space,
        embed_dim: int = 128,
        n_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        # observation_space.shape = (seq_len, n_feats)
        super().__init__(observation_space, features_dim=embed_dim)
        seq_len, n_feats = observation_space.shape

        # Un embedding lineal sobre cada paso de la secuencia
        self.input_proj = nn.Linear(n_feats, embed_dim)
        # Positional encodings fijas
        pe = th.zeros(seq_len, embed_dim)
        pos = th.arange(0, seq_len).unsqueeze(1).float()
        div = th.exp(th.arange(0, embed_dim, 2).float() * -(th.log(th.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = th.sin(pos * div)
        pe[:, 1::2] = th.cos(pos * div)
        self.register_buffer("pos_encoding", pe)

        # Capas del Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,   # nuestra secuencia es batch x seq_len x embed_dim
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling: toma la última posición o haz un mean-pool
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations: (batch, seq_len, n_feats)
        x = self.input_proj(observations)              # → (batch, seq_len, embed_dim)
        x = x + self.pos_encoding.unsqueeze(0)         # añade encoding → mismo shape
        x = self.transformer(x)                        # → (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)                         # → (batch, embed_dim, seq_len)
        x = self.pool(x).squeeze(-1)                   # → (batch, embed_dim)
        return x
