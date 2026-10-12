import math
import torch
import pytest
from src.rms_norm import RMSNorm
from src.rope import precompute_freqs_cis, apply_rotary_emb
from src.attention import CausalSelfAttention


def test_rms_norm_shape_and_numerical_stability():
    dim = 64
    norm = RMSNorm(dim=dim)
    x = torch.randn(2, 10, dim)
    y = norm(x)

    assert y.shape == x.shape
    # O valor médio quadrático normalizado de cada vetor deve ser aproximadamente 1
    rms = torch.sqrt(torch.mean(y**2, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_rope_shapes_and_invariance():
    head_dim = 32
    seq_len = 16
    batch = 2
    n_heads = 4

    freqs_cis = precompute_freqs_cis(dim=head_dim, end=seq_len)
    assert freqs_cis.shape == (seq_len, head_dim // 2)

    # Tensores Q e K com shape [B, n_heads, T, head_dim]
    q = torch.randn(batch, n_heads, seq_len, head_dim)
    k = torch.randn(batch, n_heads, seq_len, head_dim)

    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    # Valores não devem ser todos idênticos aos originais após rotação para pos > 0
    assert not torch.allclose(q_rot[:, :, 1:, :], q[:, :, 1:, :])


def test_causal_self_attention_shape_and_masking():
    n_embd = 64
    n_heads = 4
    head_dim = n_embd // n_heads
    block_size = 16
    batch = 2
    seq_len = 8

    attn = CausalSelfAttention(
        n_embd=n_embd,
        n_heads=n_heads,
        block_size=block_size,
        dropout=0.0,
    )
    freqs_cis = precompute_freqs_cis(dim=head_dim, end=seq_len)

    x = torch.randn(batch, seq_len, n_embd)
    out = attn(x, freqs_cis=freqs_cis)

    assert out.shape == (batch, seq_len, n_embd)

    # Teste de não-vazamento temporal (causalidade):
    # Alterar o token na posição t=5 NÃO deve alterar o output na posição t=2
    x_perturbed = x.clone()
    x_perturbed[:, 5:, :] = torch.randn_like(x[:, 5:, :])

    out_original = attn(x, freqs_cis=freqs_cis)
    out_perturbed = attn(x_perturbed, freqs_cis=freqs_cis)

    # Posições até t=4 devem ser exatamente iguais
    assert torch.allclose(out_original[:, :5, :], out_perturbed[:, :5, :], atol=1e-5)
