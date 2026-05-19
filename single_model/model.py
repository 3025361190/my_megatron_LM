from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, emd_size, scale, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(emd_size, scale * emd_size)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(scale * emd_size, emd_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, emd_size, num_heads, max_seq_length, dropout=0.1, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emd_size // num_heads
        assert self.head_dim * num_heads == emd_size, "Embedding size must be divisible by number of heads"

        self.qkv_proj = nn.Linear(emd_size, 3 * emd_size, bias=bias)
        self.c_proj = nn.Linear(emd_size, emd_size, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_length, max_seq_length)).view(1, 1, max_seq_length, max_seq_length),
            persistent=False,
        )

    def forward(self, x):
        batch_size, seq_length, emd_size = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self.bias[:, :, :seq_length, :seq_length] == 0
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, emd_size)
        output = self.c_proj(attn_output)
        output = self.out_dropout(output)
        return output


class LayerNorm(nn.Module):
    def __init__(self, emd_size, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(emd_size))
        self.beta = nn.Parameter(torch.zeros(emd_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta


class Block(nn.Module):
    def __init__(self, emd_size, num_heads, max_seq_length, mlp_scale=4, dropout=0.1, bias=True):
        super().__init__()
        self.ln1 = LayerNorm(emd_size)
        self.attn = CausalSelfAttention(emd_size, num_heads, max_seq_length, dropout, bias=bias)
        self.ln2 = LayerNorm(emd_size)
        self.mlp = MLP(emd_size, mlp_scale, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    emd_size: int = 768
    max_seq_length: int = 1024
    num_heads: int = 12
    num_layers: int = 12
    mlp_scale: int = 4
    dropout: float = 0.0
    bias: bool = True


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.emd_size)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.emd_size)
        self.blocks = nn.ModuleList(
            [
                Block(
                    config.emd_size,
                    config.num_heads,
                    config.max_seq_length,
                    config.mlp_scale,
                    config.dropout,
                    bias=config.bias,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.ln_f = LayerNorm(config.emd_size)
        self.head = nn.Linear(config.emd_size, config.vocab_size, bias=config.bias)
        self.head.weight = self.token_embedding.weight

        self._init_weights()
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        batch_size, seq_length = input_ids.size()
        if seq_length > self.config.max_seq_length:
            raise ValueError(f"Sequence length {seq_length} exceeds max_seq_length {self.config.max_seq_length}.")

        token_emb = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(position_ids)
        x = token_emb + position_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        if targets is not None:
            logits = self.head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss

        logits = self.head(x[:, -1, :])
        return logits, None

    def configure_optimizers(self, weight_decay=0.01, learning_rate=1e-4):
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["ln", "embedding", "bias"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        assert len(list(self.parameters())) == len(decay_params) + len(no_decay_params)
        assert len(set(decay_params).intersection(set(no_decay_params))) == 0

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        return optimizer

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        was_training = self.training
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            idx_cond = generated[:, -self.config.max_seq_length :]
            logits, _ = self(idx_cond)
            logits = logits / max(temperature, 1e-5)

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits[logits < values[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

        if was_training:
            self.train()
        return generated
