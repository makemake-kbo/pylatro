"""Transformer backbone — 8-layer pre-LN encoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class PreNormTransformerLayer(nn.Module):
    """Pre-LayerNorm transformer encoder layer."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-LN → MHA → residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)
        # Pre-LN → FFN → residual
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerBackbone(nn.Module):
    """Stack of PreNormTransformerLayers with final LayerNorm."""

    def __init__(
        self,
        n_layers: int = 8,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            PreNormTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            padding_mask: (batch, seq_len) bool — True for PAD positions
        Returns:
            (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, key_padding_mask=padding_mask)
        return self.final_norm(x)
