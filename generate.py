import argparse
import os
import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

from src.config import GPTConfig
from src.tokenizer import CharTokenizer, BPETokenizer
from src.model import MiniGPT
from src.generate import generate_text


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Reconstitui o modelo e o tokenizador a partir do checkpoint gravado."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' não foi encontrado. Execute o treinamento primeiro (python train.py)."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]

    # Reconstruir config
    config = GPTConfig(**raw_config)
    config.device = device

    # Reconstruir tokenizer (BPE ou Char conforme salvo)
    tok_data = checkpoint.get("tokenizer_data")
    if tok_data and tok_data.get("type") == "bpe":
        merges = {
            (m["pair"][0], m["pair"][1]): m["new_id"]
            for m in tok_data.get("merges", [])
        }
        tokenizer = BPETokenizer(merges=merges)
    else:
        tokenizer = CharTokenizer(vocab=checkpoint.get("vocab", []))

    # Reconstruir modelo
    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer, checkpoint


def run_chat_repl(model, tokenizer, device, temperature, top_k, top_p, max_new_tokens):
    """Loop de conversação interativa no terminal (REPL)."""
    print("\n" + "=" * 50)
    print("  MINI-GPT INTERACTIVE CHAT (Pressione Ctrl+C para sair)")
    print("=" * 50 + "\n")

    history = ""

    while True:
        try:
            user_msg = input("Você: ")
            if not user_msg.strip():
                continue

            # Prompt condicionado causalmente: você entra com '>', modelo responde com '<'
            full_prompt = f"{history}> {user_msg}\n< "

            # Truncar o histórico para caber na janela de contexto do modelo
            encoded_prompt = tokenizer.encode(full_prompt)
            max_ctx = model.config.block_size - 30
            if len(encoded_prompt) > max_ctx:
                encoded_prompt = encoded_prompt[-max_ctx:]
                prompt_input = tokenizer.decode(encoded_prompt)
            else:
                prompt_input = full_prompt

            print("Contato: ", end="", flush=True)

            # Gerar resposta
            response = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device,
            )

            # Extrair apenas os novos tokens gerados após '< '
            new_text = response[len(prompt_input):]
            # Cortar quando o turno do parceiro terminar e voltar para '>'
            clean_reply = new_text.split(">")[0].strip()
            clean_reply = clean_reply.replace("<", "").strip()

            if not clean_reply:
                clean_reply = "..."

            print(clean_reply + "\n")

            # Atualizar histórico para a próxima rodada
            history = f"{history}> {user_msg}\n< {clean_reply}\n"
            if len(tokenizer.encode(history)) > model.config.block_size:
                history = f"> {user_msg}\n< {clean_reply}\n"

        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando sessão de chat. Até logo!")
            break


def main():
    parser = argparse.ArgumentParser(description="Geração de texto com Mini-GPT 2")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_best.pt", help="Caminho do checkpoint")
    parser.add_argument("--prompt", type=str, default="Você: oi", help="Prompt inicial")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperatura de amostragem")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K candidatos")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P (nucleus sampling)")
    parser.add_argument("--max-new-tokens", type=int, default=120, help="Máximo de novos tokens gerados")
    parser.add_argument("--chat", action="store_true", help="Inicia modo de conversação interativa (REPL)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, checkpoint = load_model_from_checkpoint(args.checkpoint, device=device)

    step = checkpoint.get("step", 0)
    val_loss = checkpoint.get("val_loss", 0.0)
    print(f"[Modelo Carregado] Treinado até Step {step} | Val Loss: {val_loss:.4f} | Dispositivo: {device.upper()}")

    if args.chat:
        run_chat_repl(
            model=model,
            tokenizer=tokenizer,
            device=device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        print(f"\nPrompt: {args.prompt}")
        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        print("-" * 50)
        print(output)
        print("-" * 50)


if __name__ == "__main__":
    main()
