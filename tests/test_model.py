import torch
import pytest
from src.config import GPTConfig
from src.model import MiniGPT, TransformerBlock


def test_transformer_block_forward_shape():
    config = GPTConfig(n_embd=64, n_heads=4, block_size=16, dropout=0.0)
    block = TransformerBlock(config)
    from src.rope import precompute_freqs_cis

    freqs_cis = precompute_freqs_cis(dim=config.n_embd // config.n_heads, end=8)
    x = torch.randn(2, 8, config.n_embd)
    out = block(x, freqs_cis=freqs_cis)

    assert out.shape == x.shape


def test_mini_gpt_forward_and_loss_shapes():
    config = GPTConfig(
        vocab_size=50,
        block_size=16,
        n_embd=64,
        n_layers=2,
        n_heads=2,
        dropout=0.0,
    )
    model = MiniGPT(config)

    # 1. Forward sem targets (modo inferência)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(idx)

    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is None

    # 2. Forward com targets (modo treino)
    targets = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(idx, targets=targets)

    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is not None
    assert loss.item() > 0.0


def test_mini_gpt_overfits_single_sequence_sanity_check():
    """Teste de sanidade: um modelo Transformer correto deve memorizar 1 sequência até loss < 0.15."""
    torch.manual_seed(42)
    config = GPTConfig(
        vocab_size=30,
        block_size=16,
        n_embd=64,
        n_layers=2,
        n_heads=2,
        dropout=0.0,
    )
    model = MiniGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # Sequência sintética fixa
    seq = torch.tensor([[1, 5, 12, 18, 4, 9, 22, 15, 3]], dtype=torch.long)
    x = seq[:, :-1]
    y = seq[:, 1:]

    initial_loss = None
    final_loss = None

    for step in range(60):
        optimizer.zero_grad()
        _, loss = model(x, targets=y)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert initial_loss > 2.0
    assert final_loss < 0.15, f"Esperado loss < 0.15, mas obteve {final_loss:.4f}"
