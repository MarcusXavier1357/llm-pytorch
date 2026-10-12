# 🧠 Mini-GPT Modern (From Scratch)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-23%20Passed%20(TDD)-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-RoPE%20%7C%20RMSNorm%20%7C%20BPE-purple.svg)]()

Implementação didática e moderna de um modelo de linguagem autorregressivo baseado na arquitetura **Transformer**, construído **do zero em PyTorch** com convenções arquiteturais contemporâneas (inspiradas em LLaMA 3 e Mistral), tokenizador **Byte-Pair Encoding (BPE)** nativo e desenvolvimento 100% orientado por testes (**TDD**).

---

## ✨ Destaques Arquiteturais

Diferente de implementações simplistas baseadas em GPT-2 clássico (2019), este projeto incorpora o estado da arte moderno:

- **RMSNorm (Root Mean Square Layer Normalization)**: Normalização numericamente estável em arquitetura *Pre-LN*, mais eficiente e sem cálculo de média.
- **RoPE (Rotary Position Embeddings)**: Rotação analítica nos vetores de *Query* e *Key*, eliminando tabelas de embedding de posição fixas e garantindo melhor extrapolação sequencial.
- **Multi-Head Self-Attention Manual**: Projeções explícitas de $W_q, W_k, W_v$, fator de escala $\frac{1}{\sqrt{d_k}}$, máscara causal triangular inferior e dropout.
- **Byte-Pair Encoding (BPE) a nível de Bytes**: Tokenizador nativo treinado do zero sobre bytes UTF-8 (256 base), sem tokens desconhecidos (`<unk>`), garantindo suporte perfeito a emojis, caracteres acentuados e compressão de contexto em ~2.5x a 3x.
- **Causal Role Conditioning**: Sanitização de conversas com marcadores curtos de turno (`>` para usuário, `<` para interlocutor), preservando a dinâmica de diálogo sem desperdiçar tokens de atenção.
- **100% Coberto por Testes (TDD)**: 23 testes unitários verificando desde não-vazamento causal de atenção e invariância do RoPE até memorização matemática de sequências.

---

## 🏛️ Visão da Arquitetura

```text
Entrada de Texto
       │
       ▼
┌───────────────────────────────────────────────┐
│  Byte-Level BPE Tokenizer (Vocab: 256 + N)    │
└───────────────────────────────────────────────┘
       │ [Batch, Seq_Len]
       ▼
┌───────────────────────────────────────────────┐
│  Token Embedding (wte)                        │
└───────────────────────────────────────────────┘
       │ + RoPE aplicado em Q e K
       ▼
┌───────────────────────────────────────────────┐
│  Transformer Blocks (x N Camadas, Pre-LN)    │
│  ├── RMSNorm                                  │
│  ├── Manual Causal Multi-Head Attention       │
│  ├── Residual Connection (+)                  │
│  ├── RMSNorm                                  │
│  ├── MLP (Linear -> GELU -> Linear)           │
│  └── Residual Connection (+)                  │
└───────────────────────────────────────────────┘
       │
       ▼
┌───────────────────────────────────────────────┐
│  Final RMSNorm                                │
└───────────────────────────────────────────────┘
       │
       ▼
┌───────────────────────────────────────────────┐
│  LM Head (Linear -> Logits [Vocab_Size])      │
└───────────────────────────────────────────────┘
       │
       ▼
Cross-Entropy Loss (Treino)  /  Sampling (Inferência)
```

---

## 📁 Estrutura do Projeto

```text
mini-gpt/
│
├── data/
│   ├── sample_chat.txt         # Corpus sintético público para teste
│   └── chat_cleaned.txt        # Corpus sanitizado gerado no treino
│
├── checkpoints/                # Checkpoints salvos (.pt)
│   ├── checkpoint_best.pt      # Modelo com melhor Val Loss
│   └── checkpoint_latest.pt    # Último checkpoint de retomada
│
├── src/
│   ├── __init__.py
│   ├── config.py               # GPTConfig centralizada
│   ├── tokenizer.py            # BaseTokenizer, CharTokenizer e BPETokenizer
│   ├── dataset.py              # Sanitização regex, split 90/10 e batches
│   ├── rms_norm.py             # Implementação do RMSNorm
│   ├── rope.py                 # Rotary Position Embeddings
│   ├── attention.py            # Multi-Head Causal Attention manual
│   ├── model.py                # TransformerBlock e MiniGPT
│   ├── trainer.py              # AdamW, Cosine LR, Métricas e Checkpointing
│   └── generate.py             # Sampling (Temperature, Top-K, Top-P)
│
├── tests/                      # Suíte de testes unitários (TDD)
│   ├── test_tokenizer.py       # Testes de compressão BPE, roundtrip e JSON
│   ├── test_dataset.py         # Testes de sanitização, batches e alinhamento causal
│   ├── test_attention.py       # Testes de RMSNorm, RoPE e causal mask
│   ├── test_model.py           # Forward pass, loss e teste de overfitting
│   ├── test_trainer.py         # Scheduler de LR e persistência de checkpoints
│   └── test_generate.py        # Validação estocástica dos samplers
│
├── train.py                    # CLI principal de treinamento
├── generate.py                 # CLI de geração e chat interativo (REPL)
├── requirements.txt            # Dependências mínimas
├── AGENTS.md                   # Diretrizes para assistentes de IA
└── README.md
```

---

## ⚡ Começando

### 1. Clonar e Instalar Dependências

```bash
git clone https://github.com/SEU-USUARIO/mini-gpt-from-scratch.git
cd mini-gpt-from-scratch

python -m venv .venv
# No Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# No Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Executar a Suíte de Testes Unitários

```bash
python -m pytest tests/ -v
```

Todos os 23 testes unitários devem passar:
```text
tests/test_attention.py::test_rms_norm_shape_and_numerical_stability PASSED
tests/test_attention.py::test_rope_shapes_and_invariance PASSED
tests/test_attention.py::test_causal_self_attention_shape_and_masking PASSED
tests/test_dataset.py::test_clean_whatsapp_chat_removes_timestamps_and_system_messages PASSED
...
tests/test_tokenizer.py::test_bpe_tokenizer_encode_decode_roundtrip PASSED
============================== 23 passed in 3.08s ==============================
```

---

## 🏋️‍♂️ Treinamento

Para treinar usando o dataset de exemplo incluído:

```bash
python train.py --data data/sample_chat.txt --tokenizer bpe --vocab-size 512 --steps 1000 --batch-size 32 --block-size 128
```

Para treinar com seu próprio arquivo de texto ou exportação de conversa:
1. Coloque seu arquivo como `chat.txt` na raiz (ele está protegido no `.gitignore` para não ser commitado).
2. Execute:
```bash
python train.py --data chat.txt --tokenizer bpe --vocab-size 512 --steps 1000 --batch-size 32 --block-size 128
```

### Parâmetros Configuráveis:
- `--tokenizer`: Tipo de tokenizador (`bpe` [recomendado] ou `char`).
- `--vocab-size`: Tamanho do vocabulário BPE (padrão: `512`).
- `--steps`: Total de passos de otimização (padrão: `1000`).
- `--batch-size`: Tamanho do mini-batch (padrão: `32`).
- `--block-size`: Tamanho da janela de contexto (padrão: `128`).
- `--layers`: Quantidade de blocos Transformer (padrão: `4`).
- `--heads`: Cabeças de atenção (padrão: `4`).
- `--embedding`: Dimensão dos vetores de embedding (padrão: `128`).
- `--lr`: Taxa de aprendizado inicial (padrão: `3e-4`).

---

## 💬 Geração e Chat Interativo

### Modo Chat Interativo (Terminal REPL)
Converse diretamente com o modelo no console simulando troca de mensagens:

```bash
python generate.py --chat --temperature 0.7 --top-k 40 --top-p 0.9
```

### Modo Prompt Único
```bash
python generate.py --prompt "> Olá, tudo bem?\n< " --temperature 0.7 --max-new-tokens 80
```

### Controles de Amostragem (*Sampling*):
- `--temperature`: Controla a aleatoriedade (`0.2` a `0.4` para determinístico/coerente, `0.7` a `0.9` para criativo).
- `--top-k`: Filtra apenas os $K$ tokens mais prováveis antes da amostragem (padrão: `40`).
- `--top-p`: Nucleus sampling — mantém o menor grupo de tokens cuja probabilidade acumulada atinja $P$ (padrão: `0.9`).

---

## 🔒 Privacidade e Segurança de Dados

O arquivo `.gitignore` vem pré-configurado de fábrica para **bloquear** qualquer arquivo de conversa pessoal (`chat.txt`, `data/chat_cleaned.txt`, vocabulários gerados e checkpoints binários `.pt`), garantindo que nenhum dado sensível seja exposto acidentalmente no controle de versão.

---

## 📜 Licença

Distribuído sob a licença MIT. Consulte `LICENSE` para mais detalhes.
