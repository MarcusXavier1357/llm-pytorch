from typing import Tuple
import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Pré-calcula a grade de frequências angulares (freqs) para RoPE."""
    if dim % 2 != 0:
        raise ValueError(f"A dimensão ({dim}) deve ser par para aplicar RoPE.")

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs_table = torch.outer(t, freqs)  # [end, dim // 2]
    return freqs_table


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_table: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aplica Rotary Position Embedding (RoPE) aos tensores de Query e Key.
    
    Shapes de entrada:
        xq, xk: [batch, n_heads, seq_len, head_dim]
        freqs_table: [seq_len, head_dim // 2]
    """
    seq_len = xq.shape[2]
    # Truncar freqs_table para o comprimento da sequência atual se necessário
    freqs = freqs_table[:seq_len].to(xq.device)
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim // 2]
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(1)

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).flatten(-2)

    return _rotate(xq), _rotate(xk)
