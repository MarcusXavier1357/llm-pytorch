# Mini-GPT 2 — Plano de Implementação

## 1. Objetivo

Evoluir o mini-GPT inicial para uma implementação didática de um modelo de linguagem baseado em Transformer, capaz de:

- receber um corpus `.txt`;
- tokenizar o texto;
- construir automaticamente o dataset;
- treinar em mini-batches;
- separar treinamento e validação;
- utilizar múltiplos Transformer Blocks;
- implementar Multi-Head Self-Attention manualmente;
- calcular e acompanhar `train loss` e `validation loss`;
- salvar e carregar checkpoints;
- gerar texto;
- controlar a geração com `temperature`, `top-k` e `top-p`;
- utilizar GPU quando disponível;
- registrar métricas do treinamento.

O objetivo principal não é criar um LLM competitivo, mas construir uma implementação suficientemente completa para compreender o funcionamento de um GPT moderno.

---

# 2. Estrutura do projeto

Criar a seguinte estrutura:

```text
mini-gpt/
│
├── data/
│   └── input.txt
│
├── checkpoints/
│
├── src/
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── model.py
│   ├── attention.py
│   ├── trainer.py
│   ├── generate.py
│   └── config.py
│
├── train.py
├── generate.py
├── requirements.txt
└── README.md
```

Inicialmente também pode ser mantido tudo em um único arquivo para facilitar o aprendizado. A separação em módulos deve acontecer depois que o funcionamento estiver validado.

---

# 3. Etapa 1 — Preparação do ambiente

## Dependências

Utilizar:

```text
Python 3.11+
PyTorch
NumPy
tqdm
```

Opcionalmente:

```text
matplotlib
```

para visualizar as curvas de treinamento.

## Verificar hardware

O programa deve detectar automaticamente:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

E informar:

```text
Device: CUDA
GPU: ...
VRAM: ...
```

Caso CUDA não esteja disponível:

```text
Device: CPU
```

---

# 4. Etapa 2 — Dataset

Utilizar inicialmente um arquivo:

```text
data/input.txt
```

O usuário poderá colocar qualquer texto nele.

Exemplo:

```text
O gato correu pelo jardim.
O cachorro estava dormindo.
...
```

O programa deverá:

1. carregar o arquivo;
2. verificar encoding;
3. remover caracteres desnecessários apenas se configurado;
4. contabilizar quantidade de caracteres;
5. exibir estatísticas;
6. enviar o texto para o tokenizer.

Informações exibidas:

```text
Corpus carregado

Caracteres: 1.284.392
Linhas: 42.182
```

---

# 5. Etapa 3 — Tokenização

## Versão inicial

Começar com tokenização por caracteres.

Exemplo:

```text
"gato"
```

vira:

```text
["g", "a", "t", "o"]
```

Criar:

```python
encode(text)
decode(tokens)
```

E um vocabulário:

```python
stoi
itos
```

## Versão posterior

Implementar um tokenizer baseado em BPE.

Objetivo:

```text
"computador"
```

poderia ser dividido em:

```text
["comput", "ador"]
```

em vez de:

```text
["c", "o", "m", "p", "u", "t", "a", "d", "o", "r"]
```

O tokenizer deverá permitir:

```python
tokenizer.encode(text)
tokenizer.decode(tokens)
```

Também deverá salvar seu vocabulário para que o modelo treinado possa ser utilizado posteriormente.

---

# 6. Etapa 4 — Construção do dataset

Definir:

```python
block_size
```

Por exemplo:

```text
128 tokens
```

Dado:

```text
A B C D E F G H
```

com `block_size = 4`:

```text
X = A B C D
Y = B C D E

X = B C D E
Y = C D E F

X = C D E F
Y = D E F G
```

O objetivo é sempre:

```text
prever o próximo token.
```

---

# 7. Etapa 5 — Train / Validation Split

Dividir o corpus:

```text
90% → treinamento
10% → validação
```

Exemplo:

```text
Train:
1.800.000 tokens

Validation:
200.000 tokens
```

O conjunto de validação nunca deve ser utilizado para atualizar os pesos.

Ele serve para verificar generalização.

---

# 8. Etapa 6 — Mini-Batches

Em vez de enviar todo o dataset para o modelo, utilizar batches.

Configuração inicial:

```python
batch_size = 32
```

Cada batch terá:

```text
32 sequências
×
128 tokens
```

Formato:

```text
[B, T]
```

onde:

```text
B = batch size
T = sequence length
```

Exemplo:

```text
[32, 128]
```

---

# 9. Etapa 7 — Token Embeddings

Criar:

```python
nn.Embedding(vocab_size, n_embd)
```

Cada token será convertido em um vetor.

Exemplo:

```text
token "gato"
       ↓
[0.13, -0.82, 0.44, ...]
```

Configuração inicial:

```python
n_embd = 128
```

---

# 10. Etapa 8 — Positional Embedding

O Transformer precisa saber a posição dos tokens.

Implementar:

```python
position_embedding
```

Assim:

```text
token embedding
       +
position embedding
       ↓
representação final
```

Exemplo:

```text
"gato"

posição 0
posição 1
posição 2
posição 3
```

O modelo passa a diferenciar:

```text
gato corre
```

de:

```text
corre gato
```

---

# 11. Etapa 9 — Implementar Self-Attention manualmente

Não utilizar inicialmente:

```python
nn.MultiheadAttention
```

Implementar diretamente.

Para cada entrada:

```text
X
```

calcular:

```text
Q = XWq
K = XWk
V = XWv
```

Depois:

```text
scores = QKᵀ / √d
```

Aplicar máscara causal:

```text
         token futuro

[1, 0, 0, 0]
[1, 1, 0, 0]
[1, 1, 1, 0]
[1, 1, 1, 1]
```

Depois:

```text
attention_weights = softmax(scores)
```

E finalmente:

```text
output = attention_weights @ V
```

---

# 12. Etapa 10 — Multi-Head Attention

Criar múltiplas cabeças de attention.

Configuração inicial:

```python
n_heads = 4
```

Com:

```text
n_embd = 128
```

cada cabeça pode trabalhar com:

```text
head_dim = 32
```

Estrutura:

```text
                    Input
                      │
       ┌──────────────┼──────────────┐
       ↓              ↓              ↓
     Head 1          Head 2        Head 3 ... Head 4
       │              │              │
       └──────────────┼──────────────┘
                      ↓
                 Concatenação
                      ↓
                Linear Projection
```

---

# 13. Etapa 11 — Feed Forward Network

Depois da attention:

```text
Linear
   ↓
GELU
   ↓
Linear
```

Configuração:

```python
n_embd = 128
ffn_dim = 4 * n_embd
```

Portanto:

```text
128
 ↓
512
 ↓
128
```

Utilizar `GELU` em vez de `ReLU` para aproximar a arquitetura de GPTs reais.

---

# 14. Etapa 12 — Residual Connections

Cada Transformer Block deverá possuir conexões residuais.

Estrutura:

```text
x
│
├───────────────┐
↓               │
LayerNorm       │
↓               │
Attention       │
↓               │
└──── + x ◄─────┘
        │
        ↓
       x
```

Depois:

```text
x
│
├───────────────┐
↓               │
LayerNorm       │
↓               │
Feed Forward    │
↓               │
└──── + x ◄─────┘
        │
        ↓
      output
```

Utilizar a variante **Pre-LN**:

```text
LayerNorm
→ Attention
→ Residual

LayerNorm
→ Feed Forward
→ Residual
```

---

# 15. Etapa 13 — Transformer Blocks

Criar uma classe:

```python
TransformerBlock
```

Depois empilhar:

```python
n_layers = 4
```

Estrutura:

```text
Embedding
    ↓
Block 1
    ↓
Block 2
    ↓
Block 3
    ↓
Block 4
    ↓
LayerNorm
    ↓
LM Head
```

---

# 16. Etapa 14 — Language Model Head

Transformar a representação final em logits:

```python
nn.Linear(n_embd, vocab_size)
```

Resultado:

```text
[B, T, vocab_size]
```

Cada posição terá uma distribuição sobre todos os tokens possíveis.

Exemplo:

```text
"gato come"

↓

peixe     42%
carne     21%
ração     10%
...
```

---

# 17. Etapa 15 — Causal Language Modeling

O treinamento continuará sendo:

```text
entrada:

O gato come

alvo:

gato come peixe
```

Para cada posição:

```text
token atual → próximo token
```

Utilizar:

```python
F.cross_entropy()
```

como função de perda.

---

# 18. Etapa 16 — Otimizador

Utilizar inicialmente:

```python
torch.optim.AdamW
```

Configuração inicial:

```python
learning_rate = 3e-4
weight_decay = 0.1
```

O treinamento seguirá:

```text
forward
    ↓
loss
    ↓
zero_grad
    ↓
backward
    ↓
optimizer.step()
```

---

# 19. Etapa 17 — Learning Rate Scheduler

Implementar um scheduler.

Estratégia recomendada:

```text
Warmup
   ↓
Learning Rate máximo
   ↓
Cosine Decay
   ↓
Learning Rate mínimo
```

Exemplo:

```text
LR
│        ╭──────╮
│       ╱        ╲
│      ╱          ╲
│_____/            ╲____
│
└─────────────────────────
        steps
```

---

# 20. Etapa 18 — Gradient Clipping

Para evitar gradientes excessivamente grandes:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

Isso deverá ocorrer antes de:

```python
optimizer.step()
```

---

# 21. Etapa 19 — Métricas

Durante o treinamento registrar:

```text
step
epoch
train_loss
validation_loss
learning_rate
tokens_processed
```

Exemplo:

```text
Step: 10.000

Train Loss: 1.842
Val Loss:   1.921

Learning Rate: 0.00018

Tokens: 12.800.000
```

---

# 22. Etapa 20 — Perplexity

Calcular:

```python
perplexity = exp(loss)
```

A perplexidade fornece outra maneira de acompanhar o desempenho do modelo.

Exemplo:

```text
Loss: 2.30
Perplexity: 9.97
```

Quanto menor, geralmente melhor.

---

# 23. Etapa 21 — Checkpoints

Periodicamente salvar:

```text
checkpoints/
    checkpoint_001.pt
    checkpoint_002.pt
    checkpoint_latest.pt
```

Cada checkpoint deverá conter:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "step": ...,
    "loss": ...,
    "config": ...
}
```

Isso permitirá interromper e continuar o treinamento.

---

# 24. Etapa 22 — Geração de texto

Criar função:

```python
generate(prompt)
```

Fluxo:

```text
prompt
  ↓
tokenizer
  ↓
modelo
  ↓
logits
  ↓
último token
  ↓
sampling
  ↓
novo token
  ↓
adiciona ao contexto
  ↓
repete
```

---

# 25. Etapa 23 — Temperature

Implementar:

```python
logits = logits / temperature
```

Comportamento:

```text
temperature = 0.2
→ mais determinístico

temperature = 0.7
→ equilibrado

temperature = 1.0
→ distribuição original

temperature = 1.5
→ mais aleatório
```

---

# 26. Etapa 24 — Top-K Sampling

Exemplo:

```python
top_k = 40
```

Manter somente os 40 tokens mais prováveis antes do sampling.

Fluxo:

```text
vocab_size = 10.000

        ↓

Top-K = 40

        ↓

40 candidatos

        ↓

sampling
```

---

# 27. Etapa 25 — Top-P Sampling

Implementar nucleus sampling.

Exemplo:

```python
top_p = 0.9
```

Manter o menor conjunto de tokens cuja probabilidade acumulada seja aproximadamente:

```text
90%
```

Permitir:

```text
temperature
top_k
top_p
```

serem configurados independentemente.

---

# 28. Etapa 26 — Weight Tying

Como otimização opcional:

```text
Token Embedding
       ↑
       │
       ↓
   LM Head
```

Compartilhar os pesos do embedding com o output projection.

Isso reduz parâmetros e é utilizado em diversas arquiteturas de linguagem.

---

# 29. Etapa 27 — Mixed Precision

Quando houver GPU compatível, utilizar:

```text
float16
```

ou:

```text
bfloat16
```

através de:

```python
torch.autocast()
```

Objetivos:

- reduzir consumo de VRAM;
- aumentar velocidade;
- permitir modelos maiores.

---

# 30. Etapa 28 — Gradient Accumulation

Caso a GPU não consiga comportar um batch grande:

```text
batch físico = 8
gradient accumulation = 4
```

Simular:

```text
batch efetivo = 32
```

Fluxo:

```text
batch 1 → backward
batch 2 → backward
batch 3 → backward
batch 4 → backward
             ↓
       optimizer.step()
```

---

# 31. Etapa 29 — Contagem de parâmetros

O programa deverá informar:

```text
Trainable parameters:
2,743,296
```

Separando:

```text
Embedding
Attention
Feed Forward
LayerNorm
LM Head
```

Isso permite entender onde estão os parâmetros do modelo.

---

# 32. Etapa 30 — Monitoramento

Criar uma saída semelhante:

```text
╔══════════════════════════════════════╗
║             MINI-GPT                 ║
╠══════════════════════════════════════╣
║ Device: CUDA                         ║
║ Parameters: 2.74M                    ║
║ Context: 128                         ║
║ Layers: 4                            ║
║ Heads: 4                             ║
║ Embedding: 128                       ║
╠══════════════════════════════════════╣
║ Step: 12,500                         ║
║ Train Loss: 1.284                    ║
║ Val Loss: 1.391                      ║
║ Perplexity: 4.02                     ║
║ LR: 0.000184                         ║
║ Tokens: 48.2M                        ║
╚══════════════════════════════════════╝
```

---

# 33. Etapa 31 — Visualização

Salvar gráficos:

```text
training_loss.png
validation_loss.png
learning_rate.png
```

O gráfico principal:

```text
Loss
│\
│ \
│  \
│   \__
│      \____
│           \____
└──────────────────
       steps
```

Também comparar:

```text
Train Loss
Validation Loss
```

para identificar overfitting.

---

# 34. Etapa 32 — Experimentos controlados

Depois que o sistema funcionar, não aumentar tudo ao mesmo tempo.

Testar uma variável por vez.

### Experimento A

```text
layers = 2
embedding = 64
heads = 2
```

### Experimento B

```text
layers = 4
embedding = 128
heads = 4
```

### Experimento C

```text
layers = 8
embedding = 256
heads = 8
```

Comparar:

```text
parâmetros
tempo
VRAM
train loss
validation loss
qualidade da geração
```

---

# 35. Etapa 33 — Testes de overfitting

Treinar primeiro em um corpus extremamente pequeno.

Exemplo:

```text
10.000 tokens
```

Treinar por muitas épocas.

Objetivo:

```text
Train Loss
→ próximo de zero

Validation Loss
→ começa a aumentar
```

Isso demonstra experimentalmente o overfitting.

---

# 36. Etapa 34 — Corpus maior

Depois testar progressivamente:

```text
10 KB
100 KB
1 MB
10 MB
100 MB
```

Medir:

```text
tempo de treinamento
tokens/segundo
memória
loss
qualidade
```

Isso permite observar empiricamente o impacto da quantidade de dados.

---

# 37. Etapa 35 — Experimentos com contexto

Comparar:

```text
block_size = 32
block_size = 64
block_size = 128
block_size = 256
```

Observar:

```text
VRAM
velocidade
loss
capacidade de manter contexto
```

Demonstrar experimentalmente o custo aproximadamente quadrático da attention.

---

# 38. Etapa 36 — Implementar inferência separadamente

O programa de geração não deverá precisar do dataset.

Fluxo:

```text
checkpoint
    +
tokenizer
    ↓
modelo
    ↓
prompt
    ↓
geração
```

Executar:

```bash
python generate.py
```

Exemplo:

```text
Prompt: O gato

Output:
O gato estava sentado perto da janela...
```

---

# 39. Etapa 37 — CLI

Criar uma interface de linha de comando.

Exemplo:

```bash
python train.py \
    --data data/input.txt \
    --batch-size 32 \
    --block-size 128 \
    --layers 4 \
    --heads 4 \
    --embedding 128 \
    --steps 50000
```

E:

```bash
python generate.py \
    --checkpoint checkpoints/latest.pt \
    --prompt "O gato" \
    --temperature 0.8 \
    --top-k 40
```

---

# 40. Configuração inicial recomendada

Para uma máquina comum:

```text
Tokenizer:       caracteres inicialmente
Context:         128
Embedding:       128
Layers:          4
Heads:           4
FFN:             512
Batch:           32
Learning rate:   3e-4
Optimizer:       AdamW
Weight decay:    0.1
Warmup:          5%
Gradient clip:   1.0
```

Depois migrar para BPE.

---

# 41. Ordem de implementação

Não implementar tudo simultaneamente.

Seguir esta ordem:

```text
1. Dataset
       ↓
2. Tokenizer
       ↓
3. Mini-batches
       ↓
4. Embeddings
       ↓
5. Self-Attention
       ↓
6. Multi-Head Attention
       ↓
7. Feed Forward
       ↓
8. Residual Connections
       ↓
9. Transformer Block
       ↓
10. Vários Blocks
       ↓
11. Language Model Head
       ↓
12. Loss
       ↓
13. Backpropagation
       ↓
14. Optimizer
       ↓
15. Validation
       ↓
16. Checkpoints
       ↓
17. Generation
       ↓
18. Temperature
       ↓
19. Top-K
       ↓
20. Top-P
       ↓
21. Scheduler
       ↓
22. Mixed Precision
       ↓
23. Gradient Accumulation
       ↓
24. BPE
       ↓
25. Experimentos
```

---

# 42. Critério de conclusão

O projeto será considerado concluído quando for possível executar:

```bash
python train.py --data data/input.txt
```

e obter:

```text
Corpus: 12.4 MB
Tokens: 3.2M

Model:
Parameters: 2.74M
Layers: 4
Heads: 4
Context: 128

Training...

Step 1000
Train Loss: 2.31
Val Loss: 2.44

Step 2000
Train Loss: 1.92
Val Loss: 2.08

...

Checkpoint saved.
```

E depois:

```bash
python generate.py \
    --checkpoint checkpoints/latest.pt \
    --prompt "O cachorro"
```

produzir texto utilizando o modelo treinado.

---

# 43. Objetivo final do laboratório

Ao terminar, deve ser possível explicar e identificar no código:

```text
Corpus
  ↓
Tokenizer
  ↓
Tokens
  ↓
Embeddings
  ↓
Positional Information
  ↓
Self-Attention
  ↓
Multi-Head Attention
  ↓
Feed Forward
  ↓
Residual Connections
  ↓
LayerNorm
  ↓
Transformer Blocks
  ↓
Logits
  ↓
Softmax
  ↓
Next Token
  ↓
Cross Entropy
  ↓
Backpropagation
  ↓
Gradients
  ↓
AdamW
  ↓
Updated Parameters
```

Esse é o ciclo fundamental:

```text
                ┌──────────────────────┐
                │                      │
                ▼                      │
             Dados                     │
                │                      │
                ▼                      │
             Modelo                    │
                │                      │
                ▼                      │
             Previsão                  │
                │                      │
                ▼                      │
               Loss                    │
                │                      │
                ▼                      │
        Backpropagation                │
                │                      │
                ▼                      │
             Gradients                 │
                │                      │
                ▼                      │
        Atualização dos pesos ─────────┘
```

A partir desse ponto, aumentar o número de parâmetros, dados e GPUs passa a ser principalmente uma questão de **escala e engenharia**, e não de descobrir um mecanismo completamente diferente.