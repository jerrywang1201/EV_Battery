import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .augment import augment_pair
from .model import Encoder1D, ProjectionHead, NTXentLoss
from .utils import set_seed


class SSLDataset(Dataset):
    def __init__(self, X, crop_size):
        self.X = X
        self.crop_size = crop_size

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        a, b = augment_pair(x, self.crop_size)
        return torch.from_numpy(a), torch.from_numpy(b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop", type=int, default=96)
    parser.add_argument("--proj", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    data = np.load(args.data)
    X = data["X"].astype(np.float32)

    ds = SSLDataset(X, args.crop)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = Encoder1D(in_ch=X.shape[2], out_dim=args.proj).to(device)
    proj = ProjectionHead(in_dim=args.proj, proj_dim=args.proj).to(device)
    loss_fn = NTXentLoss()
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        enc.train()
        proj.train()
        total = 0.0
        for a, b in dl:
            a = a.to(device)
            b = b.to(device)
            za = proj(enc(a))
            zb = proj(enc(b))
            loss = loss_fn(za, zb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / max(1, len(dl))
        print(f"Epoch {epoch+1}/{args.epochs} loss={avg:.4f}")

    torch.save(enc.state_dict(), os.path.join(args.out, "encoder.pt"))
    print("Saved encoder.")


if __name__ == "__main__":
    main()
