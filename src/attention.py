import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.rope import apply_rotary_emb


class CausalSelfAttention(nn.Module):
    """Multi-Head Self-Attention manual com máscara causal e RoPE."""

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        block_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_embd % n_heads != 0:
            raise ValueError(f"n_embd ({n_embd}) deve ser divisível por n_heads ({n_heads}).")

        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.block_size = block_size

        # Projeção linear combinada para Q, K e V
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Projeção de saída
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Regularização
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Máscara causal: triangular inferior
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, t, c = x.shape  # batch_size, seq_len, n_embd

        # 1. Projeção Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape para multi-head: [B, n_heads, T, head_dim]
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. Aplicar Rotary Position Embeddings (RoPE) se fornecido
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        # 4. Cálculo manual dos scores: Q @ K^T / sqrt(head_dim)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 5. Aplicação da máscara causal (impede atenção ao futuro)
        scores = scores.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))

        # 6. Softmax e Dropout nas probabilidades de atenção
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 7. Multiplicação pelos valores: weights @ V
        y = attn_weights @ v  # [B, n_heads, T, head_dim]

        # 8. Reagrupamento das cabeças (concatenação)
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # [B, T, n_embd]

        # 9. Projeção final de saída e dropout residual
        return self.resid_dropout(self.c_proj(y))
