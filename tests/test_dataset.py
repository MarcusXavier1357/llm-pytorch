import torch
import pytest
from src.dataset import clean_whatsapp_chat, create_train_val_split, get_batch
from src.tokenizer import CharTokenizer


def test_clean_whatsapp_chat_removes_timestamps_and_system_messages():
    raw_text = (
        "As mensagens e as ligações são protegidas com a criptografia de ponta a ponta.\n"
        "[8/27/26, 7:56:14 PM] - [Ligação]\n"
        "[8/28/26, 6:42:34 PM] - [Ligação]\n"
        "[9/21/26, 9:19:28 AM] Você: eu disse q n sabia\n"
        "[9/21/26, 9:27:32 AM] Amigo: certo\n"
        "[9/21/26, 10:35:51 AM] Amigo: <figurinha omitida>\n"
        "[9/21/26, 10:35:53 AM] Você: oq é isso?\n"
    )

    cleaned = clean_whatsapp_chat(raw_text)

    # Não deve conter timestamps
    assert "[8/27/26" not in cleaned
    assert "[9/21/26" not in cleaned
    # Não deve conter mensagens de sistema
    assert "[Ligação]" not in cleaned
    assert "<figurinha omitida>" not in cleaned
    assert "As mensagens e as ligações são protegidas" not in cleaned
    # Não deve conter nomes literais de interlocutores
    assert "Você:" not in cleaned
    assert "Amigo:" not in cleaned
    # Deve conter marcadores curtos de diálogo (> para você, < para o parceiro)
    assert "> eu disse q n sabia" in cleaned
    assert "< certo" in cleaned
    assert "> oq é isso?" in cleaned


def test_create_train_val_split_shapes_and_ratio():
    tokens = list(range(1000))
    train_data, val_data = create_train_val_split(tokens, split_ratio=0.9)

    assert isinstance(train_data, torch.Tensor)
    assert isinstance(val_data, torch.Tensor)
    assert len(train_data) == 900
    assert len(val_data) == 100
    assert train_data.dtype == torch.long
    assert val_data.dtype == torch.long


def test_get_batch_shapes_and_target_alignment():
    # Tensor de 0 a 999
    data = torch.arange(1000, dtype=torch.long)
    batch_size = 4
    block_size = 8

    x, y = get_batch(data, batch_size=batch_size, block_size=block_size, device="cpu")

    # Shapes esperados: [B, T]
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)

    # Cada y deve ser exatamente o próximo token de x (shift causal)
    for b in range(batch_size):
        for t in range(block_size - 1):
            assert y[b, t].item() == x[b, t + 1].item()
