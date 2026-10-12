from typing import Optional
import torch
import torch.nn.functional as F
from src.model import MiniGPT
from src.tokenizer import BaseTokenizer


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> int:
    """Amostra o próximo token a partir dos logits usando Temperature, Top-K e Top-P."""
    # Garante 1D
    if logits.dim() > 1:
        logits = logits.squeeze()

    # Tratamento para temperatura muito baixa (amostragem gananciosa / greedy)
    if temperature < 1e-3:
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    # Filtragem Top-K
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        val, _ = torch.topk(logits, k)
        min_val = val[-1]
        logits = torch.where(logits < min_val, torch.tensor(float("-inf"), device=logits.device), logits)

    # Filtragem Top-P (Nucleus Sampling)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Máscara para tokens além do limiar top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Manter sempre pelo menos o primeiro candidato mais provável
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    return int(next_token)


@torch.no_grad()
def generate_text(
    model: MiniGPT,
    tokenizer: BaseTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9,
    device: str = "cpu",
) -> str:
    """Gera texto a partir de um prompt inicial até max_new_tokens."""
    model.eval()
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Truncar o contexto para a janela máxima permitida pelo modelo (block_size)
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        last_logits = logits[0, -1, :]

        next_token = sample_next_token(
            last_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        idx = torch.cat((idx, torch.tensor([[next_token]], device=device)), dim=1)

    return tokenizer.decode(idx[0].tolist())
