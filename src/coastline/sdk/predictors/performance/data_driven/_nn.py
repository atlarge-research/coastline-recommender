"""Neural-net architecture for the deep-learning performance predictor.

It lives in the installed package so inference can rebuild the trained net without the
training code. The module imports ``torch``, so it is loaded only when the deep-learning
predictor is selected.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Pre-activation residual block: BatchNorm, SiLU, Dropout and Linear, twice, plus a skip."""

    def __init__(self, dim, dropout_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class EmbeddingNN(nn.Module):
    """Embedding net with SiLU activations, residual blocks, optional training-time Gaussian
    noise, and separate throughput and runtime heads; outputs columns [throughput, runtime] in
    log space. Layer names and order set the checkpoint's state_dict keys, so keep them stable."""

    def __init__(self, embedding_dims, num_numerical_features, hidden_dims, dropout_rate, noise_std=0.0):
        super().__init__()
        self.noise_std = noise_std
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
                for name, (vocab_size, emb_dim) in embedding_dims.items()
            }
        )
        total_emb_dim = sum(emb_dim for _, emb_dim in embedding_dims.values())
        input_dim = total_emb_dim + num_numerical_features

        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        ]
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            if hidden_dim == prev_dim:  # equal widths: residual block
                layers.append(ResidualBlock(hidden_dim, dropout_rate))
            else:
                layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.SiLU(),
                        nn.Dropout(dropout_rate),
                    ]
                )
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.throughput_head = nn.Sequential(nn.Linear(prev_dim, prev_dim // 2), nn.SiLU(), nn.Linear(prev_dim // 2, 1))
        self.runtime_head = nn.Sequential(nn.Linear(prev_dim, prev_dim // 2), nn.SiLU(), nn.Linear(prev_dim // 2, 1))
        self.cat_feature_names = list(embedding_dims.keys())
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.05)

    def forward(self, x_cat, x_num):
        embeddings = [self.embeddings[name](x_cat[:, i]) for i, name in enumerate(self.cat_feature_names)]
        x_emb = torch.cat(embeddings, dim=1)
        if self.training and self.noise_std > 0:
            x_num = x_num + torch.randn_like(x_num) * self.noise_std
        x = torch.cat([x_emb, x_num], dim=1)
        h = self.backbone(x)
        return torch.cat([self.throughput_head(h), self.runtime_head(h)], dim=1)

    def predict(self, x_cat, x_num):
        """sklearn-style alias for forward()."""
        return self.forward(x_cat, x_num)
