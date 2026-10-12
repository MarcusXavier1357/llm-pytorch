import os
import torch
import pytest
from src.config import GPTConfig
from src.model import MiniGPT
from src.tokenizer import CharTokenizer
from src.trainer import get_lr, save_checkpoint
from generate import load_model_from_checkpoint


def test_learning_rate_schedule():
    config = GPTConfig(
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_steps=100,
        max_steps=1000,
    )

    # Step 0: próximo de 0
    lr_0 = get_lr(0, config)
    assert lr_0 < 2e-4

    # Step 99: final do warmup (atinge lr máximo)
    lr_warmup = get_lr(99, config)
    assert abs(lr_warmup - 1e-3) < 1e-5

    # Step 500: decaindo
    lr_mid = get_lr(500, config)
    assert 1e-4 < lr_mid < 1e-3

    # Step 1001: min_lr
    lr_end = get_lr(1001, config)
    assert lr_end == 1e-4


def test_checkpoint_save_and_load_roundtrip(tmp_path):
    tokenizer = CharTokenizer()
    tokenizer.train("abcdef")

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=8,
        n_embd=16,
        n_layers=1,
        n_heads=2,
    )
    model = MiniGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_file = str(tmp_path / "test_ckpt.pt")
    save_checkpoint(
        filepath=ckpt_file,
        model=model,
        optimizer=optimizer,
        step=42,
        val_loss=1.234,
        config=config,
        tokenizer=tokenizer,
    )

    loaded_model, loaded_tokenizer, checkpoint = load_model_from_checkpoint(
        ckpt_file, device="cpu"
    )

    assert checkpoint["step"] == 42
    assert abs(checkpoint["val_loss"] - 1.234) < 1e-5
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size

    # Verificar que os parâmetros carregados são idênticos aos originais
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2)
