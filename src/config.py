from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Configuração e hiperparâmetros do Mini-GPT."""

    vocab_size: int = 128
    block_size: int = 128
    n_embd: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1

    # Parâmetros de Treinamento
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 3000
    warmup_steps: int = 150
    min_lr: float = 3e-5
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 50
    device: str = "cpu"
