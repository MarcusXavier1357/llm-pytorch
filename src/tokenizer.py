import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseTokenizer(ABC):
    """Interface base para tokenizadores do Mini-GPT."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Tamanho total do vocabulário."""
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Converte texto em lista de IDs inteiros."""
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Reverte lista de IDs inteiros para string."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Serializa o vocabulário e metadados em arquivo."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseTokenizer":
        """Carrega tokenizador a partir de arquivo serializado."""
        pass


class CharTokenizer(BaseTokenizer):
    """Tokenizador em nível de caracteres didático."""

    def __init__(self, vocab: List[str] | None = None) -> None:
        self.chars: List[str] = vocab or []
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def train(self, text: str) -> None:
        """Constrói o vocabulário a partir dos caracteres únicos presentes no texto."""
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        """Converte string para lista de IDs. Dispara KeyError se caractere desconhecido."""
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: List[int]) -> str:
        """Decodifica lista de inteiros de volta para texto."""
        return "".join([self.itos[i] for i in tokens])

    def save(self, path: str) -> None:
        """Salva o vocabulário em arquivo JSON."""
        data: Dict[str, Any] = {
            "type": "char",
            "vocab": self.chars,
            "vocab_size": len(self.chars),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Carrega vocabulário a partir de JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = data.get("vocab", [])
        return cls(vocab=vocab)


class BPETokenizer(BaseTokenizer):
    """Tokenizador Byte-Pair Encoding (BPE) a nível de bytes, sem tokens desconhecidos."""

    def __init__(
        self,
        merges: Dict[tuple, int] | None = None,
        vocab: Dict[int, bytes] | None = None,
    ) -> None:
        self.merges: Dict[tuple, int] = merges or {}
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = {idx: bytes([idx]) for idx in range(256)}
            for pair, new_id in self.merges.items():
                p0, p1 = pair
                self.vocab[new_id] = self.vocab[p0] + self.vocab[p1]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _get_stats(self, tokens: List[int]) -> Dict[tuple, int]:
        counts: Dict[tuple, int] = {}
        for p0, p1 in zip(tokens, tokens[1:]):
            pair = (p0, p1)
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_tokens(self, tokens: List[int], pair: tuple, new_id: int) -> List[int]:
        new_tokens = []
        i = 0
        p0, p1 = pair
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train(self, text: str, target_vocab_size: int = 512) -> None:
        """Treina merges BPE iterativamente a partir do texto UTF-8."""
        raw_bytes = list(text.encode("utf-8"))
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        tokens = raw_bytes

        num_merges = max(0, target_vocab_size - 256)
        for _ in range(num_merges):
            stats = self._get_stats(tokens)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 2:
                break
            new_id = 256 + len(self.merges)
            tokens = self._merge_tokens(tokens, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

    def encode(self, text: str) -> List[int]:
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            eligible_pairs = {pair: self.merges[pair] for pair in stats if pair in self.merges}
            if not eligible_pairs:
                break
            best_pair = min(eligible_pairs, key=eligible_pairs.get)
            new_id = self.merges[best_pair]
            tokens = self._merge_tokens(tokens, best_pair, new_id)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        byte_chunks = [self.vocab.get(idx, b"") for idx in tokens]
        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        merges_serializable = [
            {"pair": list(pair), "new_id": new_id}
            for pair, new_id in self.merges.items()
        ]
        data = {
            "type": "bpe",
            "vocab_size": self.vocab_size,
            "merges": merges_serializable,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merges = {
            (m["pair"][0], m["pair"][1]): m["new_id"]
            for m in data.get("merges", [])
        }
        return cls(merges=merges)

