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
from src.dataset import clean_whatsapp_chat, create_train_val_split
from src.model import MiniGPT
from src.trainer import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento do Mini-GPT 2 Didático")
    parser.add_argument("--data", type=str, default="chat.txt", help="Caminho do corpus bruto")
    parser.add_argument("--tokenizer", type=str, default="bpe", choices=["bpe", "char"], help="Tipo de tokenizer (bpe ou char)")
    parser.add_argument("--vocab-size", type=int, default=512, help="Tamanho do vocabulário BPE")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do mini-batch")
    parser.add_argument("--block-size", type=int, default=128, help="Comprimento do contexto")
    parser.add_argument("--layers", type=int, default=4, help="Número de camadas Transformer")
    parser.add_argument("--heads", type=int, default=4, help="Número de cabeças de atenção")
    parser.add_argument("--embedding", type=int, default=128, help="Dimensão do embedding")
    parser.add_argument("--steps", type=int, default=1000, help="Total de passos de treinamento")
    parser.add_argument("--lr", type=float, default=3e-4, help="Taxa de aprendizado inicial")
    parser.add_argument("--eval-interval", type=int, default=100, help="Intervalo de passos para avaliação")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Diretório para checkpoints")
    args = parser.parse_args()

    # 1. Detecção e Otimização de Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Hardware] Dispositivo selecionado: {device.upper()}")
    if device == "cuda":
        print(f"[Hardware] GPU: {torch.cuda.get_device_name(0)}")
    else:
        threads = max(1, min(8, (os.cpu_count() or 4) - 1))
        torch.set_num_threads(threads)
        print(f"[Hardware] CPU threads ativas: {threads}")

    # 2. Leitura e Sanitização do Corpus
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Arquivo de corpus '{args.data}' não foi encontrado.")

    print(f"[Dataset] Carregando e sanitizando: {args.data}...")
    with open(args.data, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned_text = clean_whatsapp_chat(raw_text)
    os.makedirs("data", exist_ok=True)
    cleaned_path = os.path.join("data", "chat_cleaned.txt")
    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"[Dataset] Caracteres originais: {len(raw_text):,}")
    print(f"[Dataset] Caracteres limpos:    {len(cleaned_text):,}")
    print(f"[Dataset] Salvo em: {cleaned_path}")

    # 3. Treinamento do Tokenizer (BPE ou Char)
    if args.tokenizer == "bpe":
        print(f"[Tokenizer] Treinando BPE (target_vocab_size={args.vocab_size})...")
        tokenizer = BPETokenizer()
        tokenizer.train(cleaned_text, target_vocab_size=args.vocab_size)
        vocab_path = os.path.join("data", "vocab_bpe.json")
        tokenizer.save(vocab_path)
        encoded_sample = tokenizer.encode(cleaned_text)
        compression = len(cleaned_text) / max(1, len(encoded_sample))
        print(f"[Tokenizer] BPE gerado: {tokenizer.vocab_size} tokens | Compressão: {compression:.1f}x ({len(cleaned_text):,} chars -> {len(encoded_sample):,} tokens)")
    else:
        tokenizer = CharTokenizer()
        tokenizer.train(cleaned_text)
        vocab_path = os.path.join("data", "vocab.json")
        tokenizer.save(vocab_path)
        print(f"[Tokenizer] Char vocabulário: {tokenizer.vocab_size} caracteres únicos.")

    # 4. Tokenização e Split Treino / Validação
    tokens = tokenizer.encode(cleaned_text)
    train_data, val_data = create_train_val_split(tokens, split_ratio=0.9)
    print(f"[Split] Tokens Treino: {len(train_data):,} | Tokens Validação: {len(val_data):,}")

    # 5. Configuração e Inicialização do Modelo
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.embedding,
        n_layers=args.layers,
        n_heads=args.heads,
        ffn_dim=4 * args.embedding,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.steps,
        eval_interval=args.eval_interval,
        device=device,
    )

    model = MiniGPT(config)

    # 6. Iniciar Treinamento
    train_model(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        config=config,
        checkpoint_dir=args.checkpoints_dir,
    )


if __name__ == "__main__":
    main()
