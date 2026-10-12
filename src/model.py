from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import GPTConfig
from src.rms_norm import RMSNorm
from src.rope import precompute_freqs_cis
from src.attention import CausalSelfAttention


class MLP(nn.Module):
    """Feed-Forward Network (MLP) com ativação GELU."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.ffn_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Bloco Transformer com arquitetura Pre-LN usando RMSNorm e conexões residuais."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(
            n_embd=config.n_embd,
            n_heads=config.n_heads,
            block_size=config.block_size,
            dropout=config.dropout,
        )
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN: normalização aplicada antes da atenção e do MLP
        x = x + self.attn(self.ln_1(x), freqs_cis=freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """Modelo Mini-GPT didático com RMSNorm, RoPE e Causal Attention manual."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Embedding de tokens
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Empilhamento de Blocos Transformer
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # RMSNorm final
        self.ln_f = RMSNorm(config.n_embd)

        # Language Model Head (projeção linear final para o vocabulário)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Pré-computação de frequências RoPE para o comprimento máximo de contexto
        head_dim = config.n_embd // config.n_heads
        freqs_cis = precompute_freqs_cis(head_dim, config.block_size)
        self.register_buffer("freqs_cis", freqs_cis)

        # Inicialização cuidadosa de pesos
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(
                f"Sequência com tamanho {t} excede o limite de contexto do modelo ({self.config.block_size})."
            )

        # Token embedding
        x = self.drop(self.wte(idx))

        # Obter frequências RoPE correspondentes ao comprimento atual t
        freqs_cis = self.freqs_cis[:t]

        # Passar pelos blocos Transformer
        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis)

        # Normalização final
        x = self.ln_f(x)

        # Projeção para logits
        logits = self.lm_head(x)

        # Cálculo da Loss com Cross-Entropy causal se houver targets
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    def get_parameter_breakdown(self) -> Dict[str, int]:
        """Calcula o detalhamento de parâmetros por componente do modelo."""
        embedding_params = sum(p.numel() for p in self.wte.parameters())
        attn_params = sum(
            sum(p.numel() for p in b.attn.parameters()) for b in self.blocks
        )
        mlp_params = sum(
            sum(p.numel() for p in b.mlp.parameters()) for b in self.blocks
        )
        norm_params = sum(
            sum(p.numel() for p in b.ln_1.parameters())
            + sum(p.numel() for p in b.ln_2.parameters())
            for b in self.blocks
        ) + sum(p.numel() for p in self.ln_f.parameters())
        head_params = sum(p.numel() for p in self.lm_head.parameters())
        total = sum(p.numel() for p in self.parameters())

        return {
            "Embedding": embedding_params,
            "Attention": attn_params,
            "Feed Forward": mlp_params,
            "RMSNorm": norm_params,
            "LM Head": head_params,
            "Total": total,
        }
