import os
import math
import time
from dataclasses import asdict
from typing import Dict
import torch
from src.config import GPTConfig
from src.model import MiniGPT
from src.dataset import get_batch
from src.tokenizer import BaseTokenizer


def get_lr(step: int, config: GPTConfig) -> float:
    """Calcula a taxa de aprendizado com Warmup linear e Cosine Decay."""
    # 1. Warmup linear
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    # 2. Se ultrapassar max_steps, mantém min_lr
    if step > config.max_steps:
        return config.min_lr
    # 3. Decaimento por Cosseno
    decay_ratio = (step - config.warmup_steps) / max(1, (config.max_steps - config.warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: GPTConfig,
) -> Dict[str, float]:
    """Estima as perdas médias de treino e validação e suas respectivas perplexidades."""
    out = {}
    model.eval()

    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = get_batch(data, config.batch_size, config.block_size, device=config.device)
            _, loss = model(x, targets=y)
            losses[k] = loss.item()

        mean_loss = losses.mean().item()
        out[split] = mean_loss
        # Perplexidade: exp(loss), limitada para evitar overflow numérico
        out[f"{split}_ppl"] = math.exp(min(mean_loss, 20.0))

    model.train()
    return out


def save_checkpoint(
    filepath: str,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    config: GPTConfig,
    tokenizer: BaseTokenizer,
) -> None:
    # Metadados do tokenizador
    if hasattr(tokenizer, "merges"):
        tok_data = {
            "type": "bpe",
            "merges": [
                {"pair": list(pair), "new_id": new_id}
                for pair, new_id in getattr(tokenizer, "merges", {}).items()
            ],
        }
    else:
        tok_data = {
            "type": "char",
            "vocab": getattr(tokenizer, "chars", []),
        }

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "config": asdict(config),
        "tokenizer_data": tok_data,
        "vocab": getattr(tokenizer, "chars", []),
    }
    torch.save(state, filepath)


def print_dashboard_banner(
    config: GPTConfig, param_breakdown: Dict[str, int], device_name: str
) -> None:
    """Imprime cabecalho visual e resumo dos parametros conforme Etapa 30 do plano."""
    total_params = param_breakdown["Total"]
    sep = "=" * 52
    print("\n" + sep)
    print("|                 MINI-GPT TRAINING                 |")
    print(sep)
    print(f"| Dispositivo:      {device_name:<31} |")
    print(f"| Parametros:       {total_params / 1e6:.2f}M ({total_params:,} pesos)       |")
    print(f"| Contexto (Block): {config.block_size:<31} |")
    print(f"| Camadas / Heads:  {config.n_layers} camadas / {config.n_heads} heads              |")
    print(f"| Embedding Dim:    {config.n_embd:<31} |")
    print(f"| Vocabulario:      {config.vocab_size} tokens                       |")
    print(sep + "\n")


def train_model(
    model: MiniGPT,
    tokenizer: BaseTokenizer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: GPTConfig,
    checkpoint_dir: str = "checkpoints",
) -> None:
    """Loop principal de treinamento com AdamW, Warmup, Cosine LR e monitoramento."""
    model.to(config.device)
    param_breakdown = model.get_parameter_breakdown()
    print_dashboard_banner(config, param_breakdown, config.device.upper())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    best_val_loss = float("inf")
    total_tokens_processed = 0
    t0 = time.time()

    latest_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    best_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")

    print(f"Iniciando treinamento ({config.max_steps} steps, batch_size={config.batch_size})...\n")

    for step in range(config.max_steps + 1):
        # 1. Avaliação periódica e salvamento
        if step % config.eval_interval == 0 or step == config.max_steps:
            metrics = estimate_loss(model, train_data, val_data, config)
            elapsed = time.time() - t0
            tokens_per_sec = total_tokens_processed / max(1e-3, elapsed)

            print(
                f"Step {step:5d}/{config.max_steps} | "
                f"Train Loss: {metrics['train']:.4f} | "
                f"Val Loss: {metrics['val']:.4f} | "
                f"Val PPL: {metrics['val_ppl']:.2f} | "
                f"Tokens/s: {tokens_per_sec:.0f}"
            )

            # Salvar checkpoint mais recente
            save_checkpoint(
                latest_ckpt_path,
                model,
                optimizer,
                step,
                metrics["val"],
                config,
                tokenizer,
            )

            # Salvar melhor modelo se validação melhorou
            if metrics["val"] < best_val_loss:
                best_val_loss = metrics["val"]
                save_checkpoint(
                    best_ckpt_path,
                    model,
                    optimizer,
                    step,
                    best_val_loss,
                    config,
                    tokenizer,
                )
                print(f"  [*] Novo melhor checkpoint salvo! (Val Loss: {best_val_loss:.4f})")

        if step == config.max_steps:
            break

        # 2. Atualização de Learning Rate
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 3. Obter batch e Forward Pass
        x, y = get_batch(train_data, config.batch_size, config.block_size, device=config.device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)

        # 4. Backward Pass & Gradient Clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)

        # 5. Otimizador
        optimizer.step()

        total_tokens_processed += config.batch_size * config.block_size

        # Feedback visual frequente a cada 10 passos
        if step % 10 == 0 and step % config.eval_interval != 0:
            print(f"  .. passo {step}/{config.max_steps} em andamento (Loss atual: {loss.item():.4f})", end="\r")

    print("\n[OK] Treinamento concluido com sucesso!")
    print(f"Melhor Val Loss: {best_val_loss:.4f}")
    print(f"Checkpoints gravados em: {os.path.abspath(checkpoint_dir)}")
