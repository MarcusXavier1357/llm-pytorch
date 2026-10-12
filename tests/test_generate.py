import torch
import pytest
from src.config import GPTConfig
from src.model import MiniGPT
from src.tokenizer import CharTokenizer
from src.generate import sample_next_token, generate_text


def test_sample_next_token_greedy_at_low_temperature():
    # Logits onde o índice 3 tem a probabilidade esmagadoramente mais alta
    logits = torch.tensor([-10.0, -5.0, 0.0, 15.0, 2.0])

    # Com temperatura muito baixa, deve escolher deterministicamente o índice 3
    token = sample_next_token(logits, temperature=0.01)
    assert token == 3


def test_sample_next_token_with_top_k():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    # Se top_k = 2, apenas índices 3 e 4 devem ser elegíveis
    torch.manual_seed(42)
    tokens = {sample_next_token(logits, temperature=1.0, top_k=2) for _ in range(50)}
    assert tokens.issubset({3, 4})


def test_sample_next_token_with_top_p():
    # Logits com probabilidade concentrada nos 2 primeiros
    logits = torch.tensor([10.0, 9.5, 1.0, 0.0, -5.0])
    torch.manual_seed(42)
    # Com top_p = 0.8, apenas índices 0 e 1 devem ser selecionados
    tokens = {sample_next_token(logits, temperature=1.0, top_p=0.8) for _ in range(50)}
    assert tokens.issubset({0, 1})


def test_generate_text_produces_string_longer_than_prompt():
    corpus = "abcdefghijklmnopqrstuvwxyz .,\n"
    tokenizer = CharTokenizer()
    tokenizer.train(corpus)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=16,
        n_embd=32,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
    )
    model = MiniGPT(config)
    model.eval()

    prompt = "abc"
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=10,
        temperature=1.0,
    )

    assert isinstance(generated, str)
    assert generated.startswith(prompt)
    assert len(generated) == len(prompt) + 10
