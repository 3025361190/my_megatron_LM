import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from distributed import get_data_parallel_group, initialize_model_parallel
from model import GPT, GPTConfig
from random import initialize_random_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train Megatron GPT (TP + DP only).")
    parser.add_argument("--tensor_model_parallel_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--emd_size", type=int, default=768)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--mlp_scale", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--use_cpu_initialization", action="store_true")
    return parser.parse_args()


def initialize_distributed(args):
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    initialize_model_parallel(args.tensor_model_parallel_size)
    return rank, local_rank


def initialize_megatron(args):
    rank, local_rank = initialize_distributed(args)
    initialize_random_seed(args.seed)
    return rank, local_rank


def build_model(args, device):
    config = GPTConfig(
        vocab_size=args.vocab_size,
        emd_size=args.emd_size,
        max_seq_length=args.max_seq_length,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_scale=args.mlp_scale,
        dropout=args.dropout,
        bias=args.bias,
        use_cpu_initialization=args.use_cpu_initialization,
        params_dtype=torch.float32,
    )
    model = GPT(config)
    if device.type == "cuda":
        model = model.to(device)
    return model


def wrap_with_ddp(model, device, use_ddp=True):
    if not use_ddp or not dist.is_initialized() or dist.get_world_size() == 1:
        return model

    if device.type == "cuda":
        return DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            process_group=get_data_parallel_group(),
        )

    return DDP(model, process_group=get_data_parallel_group())


def build_optimizer(model, learning_rate=3e-4, weight_decay=0.1, betas=(0.9, 0.95)):
    raw_model = model.module if isinstance(model, DDP) else model
    param_dict = {name: param for name, param in raw_model.named_parameters() if param.requires_grad}

    decay_params = set()
    no_decay_params = set()

    for module_name, module in raw_model.named_modules():
        for param_name, _ in module.named_parameters(recurse=False):
            full_param_name = f"{module_name}.{param_name}" if module_name else param_name
            if full_param_name not in param_dict:
                continue

            if isinstance(module, nn.Linear) and param_name == "weight":
                decay_params.add(full_param_name)
            else:
                no_decay_params.add(full_param_name)

    inter_params = decay_params & no_decay_params
    union_params = decay_params | no_decay_params
    assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets"
    assert len(param_dict.keys() - union_params) == 0, "some parameters were not separated into either decay/no_decay set"

    optim_groups = [
        {"params": [param_dict[name] for name in sorted(decay_params)], "weight_decay": weight_decay},
        {"params": [param_dict[name] for name in sorted(no_decay_params)], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)


def main():
    args = parse_args()
    rank, local_rank = initialize_megatron(args)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    model = build_model(args, device)
    model = wrap_with_ddp(model, device)
    optimizer = build_optimizer(model)


    if rank == 0:
        print("distributed initialization done")
        print(f"tensor_model_parallel_size = {args.tensor_model_parallel_size}")
        print(f"seed = {args.seed}")
        print("model instantiated and wrapped with DDP")
        print(model)
        print("optimizer constructed")
        print(optimizer)

    if torch.cuda.is_available():
        print(f"rank {rank}: using cuda device {local_rank}")
    else:
        print(f"rank {rank}: using cpu")


if __name__ == "__main__":
    main()
