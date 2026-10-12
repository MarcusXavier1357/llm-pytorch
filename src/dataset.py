import re
from typing import Tuple, List, Union
import torch


def clean_whatsapp_chat(text: str) -> str:
    """Limpa ruídos de exportação de chat do WhatsApp mantendo os diálogos e interlocutores."""
    lines = text.splitlines()
    cleaned_lines: List[str] = []

    # Padrão de timestamp como: [8/27/26, 7:56:14 PM] ou 8/27/26, 7:56:14 PM -
    timestamp_pattern = re.compile(
        r"^\[?\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\]?\s*(?:-\s*)?"
    )

    # Marcadores de ruído do sistema a serem ignorados completamente
    system_markers = [
        "As mensagens e as ligações são protegidas",
        "[Ligação]",
        "<figurinha omitida>",
        "<mídia ocultada>",
        "<mídia omitida>",
        "<imagem ocultada>",
        "<áudio ocultado>",
        "<áudio omitido>",
        "<arquivo anexado>",
    ]

    # Padrão para capturar interlocutor e conteúdo da mensagem
    speaker_pattern = re.compile(r"^([^:\n]+):\s*(.*)$")
    last_marker = "> "

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # Ignorar mensagens de sistema
        if any(marker in line for marker in system_markers):
            continue

        # Remover prefixo de timestamp
        line = timestamp_pattern.sub("", line).strip()

        # Se após remover o timestamp a linha for ruído ou vazia
        if not line or any(marker in line for marker in system_markers):
            continue

        # Detectar interlocutor e converter para marcador curto
        m = speaker_pattern.match(line)
        if m:
            speaker, content = m.group(1).strip(), m.group(2).strip()
            if not content or any(marker in content for marker in system_markers):
                continue
            if speaker.lower() in ("você", "voce"):
                last_marker = "> "
            else:
                last_marker = "< "
            cleaned_lines.append(f"{last_marker}{content}")
        else:
            # Continuação da mensagem anterior
            cleaned_lines.append(f"{last_marker}{line}")

    return "\n".join(cleaned_lines)


def create_train_val_split(
    tokens: Union[List[int], torch.Tensor], split_ratio: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Divide os tokens em tensores de treino e validação sem data leakage."""
    if not isinstance(tokens, torch.Tensor):
        data = torch.tensor(tokens, dtype=torch.long)
    else:
        data = tokens.to(dtype=torch.long)

    n = int(split_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gera um mini-batch (X, Y) com target causal shifted de 1 posição."""
    if len(data) <= block_size:
        raise ValueError(
            f"O tamanho do dataset ({len(data)}) deve ser maior que o block_size ({block_size})."
        )

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)
