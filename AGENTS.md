# 🤖 AGENTS.md — Diretrizes para Assistentes de IA e Contribuidores

Este documento define a arquitetura, convenções de código, restrições e diretrizes de desenvolvimento para qualquer agente ou engenheiro que for atuar neste repositório.

---

## 🎯 Visão Geral do Projeto

Este repositório contém uma implementação moderna e didática de um modelo **Transformer autorregressivo (Mini-GPT)** construído do zero em PyTorch, sem dependências de frameworks de alto nível (como Hugging Face Transformers).

### Tecnologias e Convenções Principais:
- **PyTorch 2.x** (execução otimizada em CPU, compatível com GPU).
- **RoPE (Rotary Position Embeddings)** em `src/rope.py` (aplicado analiticamente em $Q$ e $K$).
- **RMSNorm** em `src/rms_norm.py` (Pre-LN em todos os blocos).
- **Causal Multi-Head Attention** em `src/attention.py` com cálculo manual e máscara triangular causal inferior.
- **Byte-Pair Encoding (BPE)** a nível de bytes em `src/tokenizer.py`.
- **Pytest** para testes unitários com TDD estrito.

---

## 📐 Princípios de Engenharia e Design

1. **Test-Driven Development (TDD) Obrigatório**:
   - **Regra de ouro**: Nunca implemente um novo comportamento ou refatoração sem antes definir como será testado.
   - Sempre adicione testes unitários em `tests/` antes ou junto com alterações em `src/`.
   - Mantenha determinismo nos testes (use seeds em testes estocásticos).
   - Todos os testes devem rodar via:
     ```bash
     python -m pytest tests/ -v
     ```

2. **Fluxo Linear e Baixa Complexidade**:
   - Prefira código com fluxo de cima para baixo (*linear top-to-bottom*).
   - Evite indireções desnecessárias, metaprogramação opaca ou heranças profundas.
   - Use early returns e métodos concisos e coesos.

3. **Anti-Overengineering**:
   - Não adicione abstrações genéricas para necessidades hipotéticas futuras.
   - Prefira clareza matemática e legibilidade em vez de truques "espertos" de código.
   - Cada componente do modelo deve refletir a matemática formal do Transformer de forma intuitiva.

4. **Compatibilidade Cross-Platform (Windows e Unix)**:
   - Toda saída para o console deve usar UTF-8 seguro (`sys.stdout.reconfigure(encoding="utf-8", errors="replace")`).
   - Não utilize caracteres de desenho Unicode complexos que quebrem em terminais com `cp1252` no Windows.

---

## 🔒 Regras Rígidas de Privacidade e Segurança

1. **Proteção de Dados Pessoais**:
   - Arquivos reais de exportação de chat (`chat.txt`, `data/chat_cleaned.txt`, `data/vocab.json`, `data/vocab_bpe.json`) **nunca devem ser commitados no Git**.
   - Verifique sempre o `.gitignore` antes de adicionar arquivos ao stage (`git add`).
   - Use exclusivamente `data/sample_chat.txt` para testes automatizados e exemplos públicos.

2. **Checkpoints e Binários**:
   - Checkpoints (`checkpoints/*.pt`) são arquivos binários pesados e devem permanecer fora do controle de versão.

---

## 🧭 Mapa da Arquitetura de Código

| Caminho | Responsabilidade |
| :--- | :--- |
| `src/config.py` | Dataclass centralizada `GPTConfig` com validação de tipos e hiperparâmetros padrão. |
| `src/tokenizer.py` | `BaseTokenizer`, `CharTokenizer` e `BPETokenizer` (nível de bytes com compressão UTF-8). |
| `src/dataset.py` | Sanitização regex de WhatsApp, marcação de papéis (`>`, `<`), split 90/10 e `get_batch`. |
| `src/rms_norm.py` | Camada `RMSNorm` estável numericamente (sem média). |
| `src/rope.py` | Pré-cálculo de frequências angulares e rotação 2D de tensores $Q$ e $K$. |
| `src/attention.py` | Multi-Head Self-Attention manual com máscara causal triangular e dropout. |
| `src/model.py` | `MLP` com GELU, `TransformerBlock` Pre-LN e classe `MiniGPT` autorregressiva. |
| `src/trainer.py` | Loop de treinamento com AdamW, Warmup + Cosine LR, gradient clipping e checkpoints. |
| `src/generate.py` | Samplers estocásticos (Temperature, Top-K, Top-P) e loop de geração. |
| `train.py` | CLI principal com parsing de argumentos, detecção de threads de CPU e inicialização. |
| `generate.py` | CLI de inferência (suporta modo prompt único e modo REPL de conversa interativa). |

---

## 🧪 Como Rodar e Validar Alterações

Antes de enviar qualquer commit ou PR:
```powershell
# 1. Rodar toda a suíte de testes
python -m pytest tests/ -v

# 2. Executar um dry-run rápido de treino (10 passos) para garantir integração
python train.py --data data/sample_chat.txt --steps 10 --batch-size 4 --block-size 16

# 3. Testar a geração com o checkpoint temporário
python generate.py --prompt "Olá" --max-new-tokens 10
```
