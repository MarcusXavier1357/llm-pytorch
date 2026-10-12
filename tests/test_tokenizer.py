import json
import pytest
from src.tokenizer import CharTokenizer, BaseTokenizer


def test_char_tokenizer_inherits_base_tokenizer():
    tokenizer = CharTokenizer()
    assert isinstance(tokenizer, BaseTokenizer)


def test_char_tokenizer_train_and_vocab():
    text = "Olá, mundo! 123"
    tokenizer = CharTokenizer()
    tokenizer.train(text)

    # Vocab size deve ser igual ao número de caracteres únicos
    unique_chars = sorted(list(set(text)))
    assert tokenizer.vocab_size == len(unique_chars)
    assert tokenizer.decode(tokenizer.encode("Olá")) == "Olá"


def test_char_tokenizer_encode_decode_roundtrip():
    text = "O gato subiu no telhado.\nQuem diria?!"
    tokenizer = CharTokenizer()
    tokenizer.train(text)

    tokens = tokenizer.encode(text)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) == len(text)

    decoded = tokenizer.decode(tokens)
    assert decoded == text


def test_char_tokenizer_save_and_load(tmp_path):
    text = "Exemplo de vocabulário com pontuação & acentuação: maçã, café."
    tokenizer = CharTokenizer()
    tokenizer.train(text)

    save_path = tmp_path / "vocab.json"
    tokenizer.save(str(save_path))

    loaded_tokenizer = CharTokenizer.load(str(save_path))
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
    assert loaded_tokenizer.encode("café") == tokenizer.encode("café")
    assert loaded_tokenizer.decode(loaded_tokenizer.encode("maçã")) == "maçã"


def test_char_tokenizer_unknown_char_handling():
    text = "abc"
    tokenizer = CharTokenizer()
    tokenizer.train(text)

    # Caractere fora do vocabulário deve disparar KeyError ou tratar graciosamente
    with pytest.raises(KeyError):
        tokenizer.encode("z")


def test_bpe_tokenizer_inherits_base_and_compresses():
    from src.tokenizer import BPETokenizer

    tokenizer = BPETokenizer()
    assert isinstance(tokenizer, BaseTokenizer)

    # Texto repetitivo deve sofrer merge de pares frequentes
    text = "banana banana banana banana maçã café " * 10
    tokenizer.train(text, target_vocab_size=265)

    assert tokenizer.vocab_size == 265
    tokens = tokenizer.encode(text)
    raw_bytes_len = len(text.encode("utf-8"))
    # Com merges, a quantidade de tokens deve ser significativamente menor que os bytes brutos
    assert len(tokens) < raw_bytes_len


def test_bpe_tokenizer_encode_decode_roundtrip():
    from src.tokenizer import BPETokenizer

    text = "Olá mundo! Teste de diálogo: > oq tu tem? < nada não, só cansaço! 🍕"
    tokenizer = BPETokenizer()
    tokenizer.train(text, target_vocab_size=270)

    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert decoded == text


def test_bpe_tokenizer_save_and_load(tmp_path):
    from src.tokenizer import BPETokenizer

    text = "Treinando BPE para salvar em disco com acentuação: café e coração."
    tokenizer = BPETokenizer()
    tokenizer.train(text, target_vocab_size=275)

    save_path = tmp_path / "vocab_bpe.json"
    tokenizer.save(str(save_path))

    loaded_bpe = BPETokenizer.load(str(save_path))
    assert loaded_bpe.vocab_size == tokenizer.vocab_size
    assert loaded_bpe.encode(text) == tokenizer.encode(text)
    assert loaded_bpe.decode(loaded_bpe.encode(text)) == text
