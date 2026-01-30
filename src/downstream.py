import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .model import Encoder1D


class LabeledDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx]).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--label_frac", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train = np.load(args.data)
    val = np.load(args.val)
    X = train["X"].astype(np.float32)
    y = train["y"].astype(np.float32)
    Xv = val["X"].astype(np.float32)
    yv = val["y"].astype(np.float32)

    n = len(X)
    m = max(1, int(n * args.label_frac))
    idx = np.random.permutation(n)[:m]
    X = X[idx]
    y = y[idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = Encoder1D(in_ch=X.shape[2], out_dim=128).to(device)
    enc.load_state_dict(torch.load(args.ckpt, map_location=device))
    enc.eval()

    reg = torch.nn.Linear(128, 1).to(device)
    opt = torch.optim.Adam(reg.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    dl = DataLoader(LabeledDataset(X, y), batch_size=args.batch, shuffle=True)

    for epoch in range(args.epochs):
        reg.train()
        total = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device).view(-1, 1)
            with torch.no_grad():
                h = enc(xb)
            pred = reg(h)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} loss={total/len(dl):.4f}")

    # Evaluate on val
    enc.eval()
    reg.eval()
    with torch.no_grad():
        hv = enc(torch.from_numpy(Xv).to(device))
        pv = reg(hv).cpu().numpy().squeeze()
    mae = mean_absolute_error(yv, pv)
    rmse = np.sqrt(mean_squared_error(yv, pv))
    print(f"Val MAE: {mae:.4f}  RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
