import argparse
from pathlib import Path
from urllib.request import urlretrieve

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from model import GPT, GPTConfig


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


def build_dataloaders(text, block_size, batch_size, val_ratio):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    dataset = CharDataset(text, block_size, stoi)
    if len(dataset) == 0:
        raise ValueError("Text is too short for the selected block size.")

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        train_size = len(dataset) - 1
        val_size = 1

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader, stoi, itos


def estimate_loss(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)

            total_loss += loss.item()
            total_steps += 1

    model.train()
    return total_loss / max(total_steps, 1)


def decode(indices, itos):
    return "".join(itos[idx] for idx in indices)


def resolve_data_path(data_arg):
    if data_arg is not None:
        data_path = Path(data_arg)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        return data_path

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "tinyshakespeare.txt"

    if not data_path.exists():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"downloading default dataset to {data_path} ...")
        try:
            urlretrieve(url, data_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download the default dataset. "
                "Please rerun with --data pointing to a local text file."
            ) from exc

    return data_path


def main():
    parser = argparse.ArgumentParser(description="Train the single-model GPT on a text file.")
    parser.add_argument("--data", type=str, default=None, help="Path to a UTF-8 text file.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--sample_tokens", type=int, default=100)
    parser.add_argument("--seed_text", type=str, default="The ")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = resolve_data_path(args.data)
    text = data_path.read_text(encoding="utf-8")
    train_loader, val_loader, stoi, itos = build_dataloaders(
        text, args.block_size, args.batch_size, args.val_ratio
    )

    config = GPTConfig(
        vocab_size=len(stoi),
        emd_size=args.n_embd,
        max_seq_length=args.block_size,
        num_heads=args.n_head,
        num_layers=args.n_layer,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
    )

    print(f"device: {device}")
    print(f"data: {data_path}")
    print(f"vocab size: {len(stoi)}")
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            if global_step % args.log_interval == 0:
                print(f"epoch {epoch + 1} step {global_step}: train loss {loss.item():.4f}")

        val_loss = estimate_loss(model, val_loader, device)
        print(f"epoch {epoch + 1}: val loss {val_loss:.4f}")

    seed_ids = [stoi[ch] for ch in args.seed_text if ch in stoi]
    if not seed_ids:
        seed_ids = [0]

    prompt = torch.tensor([seed_ids], dtype=torch.long, device=device)
    generated = model.generate(prompt, max_new_tokens=args.sample_tokens, temperature=0.8, top_k=20)
    print("\nSample:")
    print(decode(generated[0].tolist(), itos))


if __name__ == "__main__":
    main()
