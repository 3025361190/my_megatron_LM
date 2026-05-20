from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from distributed import (
    VocabUtility,
    copy_to_tensor_model_parallel_region,
    divide,
    gather_from_tensor_model_parallel_region,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
    vocab_parallel_cross_entropy,
)
from random import fork_tensor_model_parallel_rng


def init_method_normal(std):
    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def _initialize_affine_weight_gpu(weight, init_method):
    with fork_tensor_model_parallel_rng():
        init_method(weight)


def _initialize_affine_weight_cpu(
    weight,
    output_size,
    input_size,
    per_partition_size,
    partition_dim,
    init_method,
    stride=1,
):
    master_weight = torch.empty(output_size, input_size, dtype=torch.float32, requires_grad=False)
    init_method(master_weight)
    master_weight = master_weight.to(dtype=weight.dtype)

    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
        gather_output=True,
        init_method=None,
        use_cpu_initialization=False,
        params_dtype=torch.float32,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.output_size_per_partition = divide(output_size, get_tensor_model_parallel_world_size())

        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                input_size,
                dtype=params_dtype,
                device="cpu" if use_cpu_initialization else torch.cuda.current_device(),
            )
        )
        self.bias = (
            nn.Parameter(
                torch.zeros(
                    self.output_size_per_partition,
                    dtype=params_dtype,
                    device="cpu" if use_cpu_initialization else torch.cuda.current_device(),
                )
            )
            if bias
            else None
        )

        if init_method is None:
            init_method = init_method_normal(0.02)
        if use_cpu_initialization:
            _initialize_affine_weight_cpu(
                self.weight,
                output_size,
                input_size,
                self.output_size_per_partition,
                partition_dim=0,
                init_method=init_method,
            )
        else:
            _initialize_affine_weight_gpu(self.weight, init_method)

    def forward(self, input_):
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        output = gather_from_tensor_model_parallel_region(output_parallel) if self.gather_output else output_parallel
        return output, None


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
        input_is_parallel=False,
        init_method=None,
        use_cpu_initialization=False,
        params_dtype=torch.float32,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.input_size_per_partition = divide(input_size, get_tensor_model_parallel_world_size())

        self.weight = nn.Parameter(
            torch.empty(
                output_size,
                self.input_size_per_partition,
                dtype=params_dtype,
                device="cpu" if use_cpu_initialization else torch.cuda.current_device(),
            )
        )
        self.bias = (
            nn.Parameter(
                torch.zeros(
                    output_size,
                    dtype=params_dtype,
                    device="cpu" if use_cpu_initialization else torch.cuda.current_device(),
                )
            )
            if bias
            else None
        )

        if init_method is None:
            init_method = init_method_normal(0.02)
        if use_cpu_initialization:
            _initialize_affine_weight_cpu(
                self.weight,
                output_size,
                input_size,
                self.input_size_per_partition,
                partition_dim=1,
                init_method=init_method,
            )
        else:
            _initialize_affine_weight_gpu(self.weight, init_method)

    def forward(self, input_):
        input_parallel = input_ if self.input_is_parallel else scatter_to_tensor_model_parallel_region(input_)
        output_parallel = F.linear(input_parallel, self.weight)
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output + self.bias
        return output, None


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        init_method=None,
        use_cpu_initialization=False,
        params_dtype=torch.float32,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            num_embeddings,
            get_tensor_model_parallel_rank(),
            self.tensor_model_parallel_size,
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                embedding_dim,
                dtype=params_dtype,
                device="cpu" if use_cpu_initialization else torch.cuda.current_device(),
            )
        )

        if init_method is None:
            init_method = init_method_normal(0.02)
        if use_cpu_initialization:
            _initialize_affine_weight_cpu(
                self.weight,
                num_embeddings,
                embedding_dim,
                self.num_embeddings_per_partition,
                partition_dim=0,
                init_method=init_method,
            )
        else:
            _initialize_affine_weight_gpu(self.weight, init_method)

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            input_mask = None
            masked_input = input_

        output_parallel = F.embedding(masked_input, self.weight)
        if input_mask is not None:
            output_parallel[input_mask, :] = 0.0
        return reduce_from_tensor_model_parallel_region(output_parallel)


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


class ParallelMLP(nn.Module):
    def __init__(self, emd_size, scale=4, dropout=0.1, use_cpu_initialization=False, params_dtype=torch.float32):
        super().__init__()
        init_method = init_method_normal(0.02)
        self.c_fc = ColumnParallelLinear(
            emd_size,
            scale * emd_size,
            gather_output=False,
            init_method=init_method,
            use_cpu_initialization=use_cpu_initialization,
            params_dtype=params_dtype,
        )
        self.gelu = nn.GELU()
        self.c_proj = RowParallelLinear(
            scale * emd_size,
            emd_size,
            input_is_parallel=True,
            init_method=init_method,
            use_cpu_initialization=use_cpu_initialization,
            params_dtype=params_dtype,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.c_fc(x)
        x = self.gelu(x)
        x, _ = self.c_proj(x)
        x = self.dropout(x)
        return x


class ParallelCausalSelfAttention(nn.Module):
    def __init__(
        self,
        emd_size,
        num_heads,
        max_seq_length,
        dropout=0.1,
        bias=True,
        use_cpu_initialization=False,
        params_dtype=torch.float32,
    ):
        super().__init__()
        world_size = get_tensor_model_parallel_world_size()
        assert emd_size % num_heads == 0, "Embedding size must be divisible by number of heads."
        assert num_heads % world_size == 0, "Number of heads must be divisible by tensor parallel size."

        self.head_dim = emd_size // num_heads
        self.num_heads_per_partition = num_heads // world_size
        self.hidden_size_per_partition = emd_size // world_size

        init_method = init_method_normal(0.02)
        self.qkv_proj = ColumnParallelLinear(
            emd_size,
            3 * emd_size,
            gather_output=False,
            init_method=init_method,
            bias=bias,
            use_cpu_initialization=use_cpu_initialization,
            params_dtype=params_dtype,
        )
        self.c_proj = RowParallelLinear(
            emd_size,
            emd_size,
            input_is_parallel=True,
            init_method=init_method,
            bias=bias,
            use_cpu_initialization=use_cpu_initialization,
            params_dtype=params_dtype,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_length, max_seq_length)).view(1, 1, max_seq_length, max_seq_length),
            persistent=False,
        )

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv, _ = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_length, 3, self.num_heads_per_partition, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self.bias[:, :, :seq_length, :seq_length] == 0
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        with fork_tensor_model_parallel_rng():
            attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.hidden_size_per_partition
        )
        output, _ = self.c_proj(attn_output)
        output = self.out_dropout(output)
        return output


class ParallelBlock(nn.Module):
    def __init__(
        self,
        emd_size,
        num_heads,
        max_seq_length,
        mlp_scale=4,
        dropout=0.1,
        bias=True,
        use_cpu_initialization=False,
        params_dtype=torch.float32,
    ):
        super().__init__()
        self.ln1 = LayerNorm(emd_size)
        self.attn = ParallelCausalSelfAttention(
            emd_size,
            num_heads,
            max_seq_length,
            dropout,
            bias=bias,
            use_cpu_initialization=use_cpu_initialization,
            params_dtype=params_dtype,
        )
        self.ln2 = LayerNorm(emd_size)
        self.mlp = ParallelMLP(
            emd_size,
            mlp_scale,
            dropout,
            use_cpu_initialization=use_cpu_initialization,
            params_dtype=params_dtype,
        )

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
    use_cpu_initialization: bool = False
    params_dtype: torch.dtype = torch.float32


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        self.parallel_output = True

        self.token_embedding = VocabParallelEmbedding(
            config.vocab_size,
            config.emd_size,
            use_cpu_initialization=config.use_cpu_initialization,
            params_dtype=config.params_dtype,
        )
        self.position_embedding = nn.Embedding(config.max_seq_length, config.emd_size)
        self.blocks = nn.ModuleList(
            [
                ParallelBlock(
                    config.emd_size,
                    config.num_heads,
                    config.max_seq_length,
                    config.mlp_scale,
                    config.dropout,
                    bias=config.bias,
                    use_cpu_initialization=config.use_cpu_initialization,
                    params_dtype=config.params_dtype,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.ln_f = LayerNorm(config.emd_size)

        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    def _lm_head(self, hidden_states):
        logits_parallel = F.linear(hidden_states, self.token_embedding.weight)
        if self.parallel_output:
            return logits_parallel
        return gather_from_tensor_model_parallel_region(logits_parallel)

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
            logits_parallel = self._lm_head(x)
            loss = vocab_parallel_cross_entropy(logits_parallel, targets).mean()
            return logits_parallel, loss

        logits_parallel = self._lm_head(x[:, -1, :])
        logits = gather_from_tensor_model_parallel_region(logits_parallel)
        return logits, None

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
