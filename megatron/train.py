import argparse
import os
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

from distributed import get_data_parallel_group, initialize_model_parallel
from model import GPT, GPTConfig
from random import initialize_random_seed


def log_rank(message, rank=None, only_rank0=False):
    if rank is None and dist.is_initialized():
        rank = dist.get_rank()
    elif rank is None:
        rank = 0

    if only_rank0 and rank != 0:
        return
    print(f"[rank {rank}] {message}", flush=True)


class CharDataset(Dataset):
    def __init__(self, text, block_size, stoi):
        self.block_size = block_size
        self.stoi = stoi
        self.data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def parse_args():
    parser = argparse.ArgumentParser(description="Train Megatron GPT (TP + DP only).")
    parser.add_argument("--tensor_model_parallel_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--emd_size", type=int, default=768)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--mlp_scale", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", action="store_true", default=True)
    parser.add_argument("--use_cpu_initialization", action="store_true")
    return parser.parse_args()


def initialize_distributed(args):
    log_rank("starting torch.distributed initialization", rank=0, only_rank0=True)
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    initialize_model_parallel(args.tensor_model_parallel_size)
    log_rank(
        f"distributed initialized: world_size={dist.get_world_size()}, local_rank={local_rank}, "
        f"tp={args.tensor_model_parallel_size}, dp={dist.get_world_size() // args.tensor_model_parallel_size}",
        rank=rank,
    )
    return rank, local_rank


def initialize_megatron(args):
    log_rank("initializing megatron runtime", rank=0, only_rank0=True)
    rank, local_rank = initialize_distributed(args)
    initialize_random_seed(args.seed)
    log_rank(f"random seed initialized with seed={args.seed}", rank=rank)
    return rank, local_rank


def build_model(args, device):
    log_rank("building GPT model", rank=0, only_rank0=True)
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
    log_rank(f"model moved to device {device}", rank=0, only_rank0=True)
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
    log_rank(
        f"optimizer parameter groups ready: decay={len(decay_params)}, no_decay={len(no_decay_params)}",
        rank=0,
        only_rank0=True,
    )
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)


def build_dataloaders(args):
    log_rank(f"loading dataset from {args.data}", rank=0, only_rank0=True)
    text = Path(args.data).read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    dataset = CharDataset(text, args.max_seq_length, stoi)
    if len(dataset) == 0:
        raise ValueError("Text is too short for the selected max_seq_length.")

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        train_size = len(dataset) - 1
        val_size = 1

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
    )
    log_rank(
        f"dataloaders ready: total_tokens={len(text)}, train_samples={len(train_dataset)}, "
        f"val_samples={len(val_dataset)}, batch_size={args.batch_size}",
        rank=0,
        only_rank0=True,
    )
    return train_loader, val_loader, train_sampler, val_sampler, stoi, itos


def train_one_epoch(
    model,
    optimizer,
    train_loader,
    train_sampler,
    device,
    epoch,
    log_interval,
    rank,
    gradient_accumulation_steps,
):
    model.train()
    train_sampler.set_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    log_rank(f"starting epoch {epoch + 1}", rank=rank, only_rank0=True)

    for step, (x, y) in enumerate(train_loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        micro_step = step % gradient_accumulation_steps
        should_sync = micro_step == gradient_accumulation_steps - 1
        sync_context = (
            nullcontext()
            if should_sync or not isinstance(model, DDP)
            else model.no_sync()
        )

        with sync_context:
            _, loss = model(x, y)
            loss = loss / gradient_accumulation_steps
            loss.backward()

        if should_sync:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % log_interval == 0 and rank == 0:
            print(f"epoch {epoch + 1} step {step}: loss {(loss.item() * gradient_accumulation_steps):.4f}")

    if len(train_loader) % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    log_rank(f"finished epoch {epoch + 1}", rank=rank, only_rank0=True)


def main():
    args = parse_args()
    log_rank(f"launching training from {Path(__file__).resolve()}", rank=0, only_rank0=True)
    rank, local_rank = initialize_megatron(args)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    model = build_model(args, device)
    model = wrap_with_ddp(model, device)
    optimizer = build_optimizer(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    train_loader, val_loader, train_sampler, val_sampler, stoi, itos = build_dataloaders(args)


    if rank == 0:
        print("distributed initialization done", flush=True)
        print(f"tensor_model_parallel_size = {args.tensor_model_parallel_size}", flush=True)
        print(f"seed = {args.seed}", flush=True)
        print("model instantiated and wrapped with DDP", flush=True)
        print(model, flush=True)
        print("optimizer constructed", flush=True)
        print(optimizer, flush=True)
        print(f"train batches = {len(train_loader)}", flush=True)
        print(f"val batches = {len(val_loader)}", flush=True)
        print(f"vocab size = {len(stoi)}", flush=True)

    if torch.cuda.is_available():
        print(f"rank {rank}: using cuda device {local_rank}")
    else:
        print(f"rank {rank}: using cpu")

    for epoch in range(args.epochs):
        train_one_epoch(
            model,
            optimizer,
            train_loader,
            train_sampler,
            device,
            epoch,
            args.log_interval,
            rank,
            args.gradient_accumulation_steps,
        )


if __name__ == "__main__":
    main()
