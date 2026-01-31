import argparse
import csv
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .model import Encoder1D
from .utils import set_seed


class LabeledDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx]).float()


def train_regressor(
    enc,
    X: np.ndarray,
    y: np.ndarray,
    Xv: np.ndarray,
    yv: np.ndarray,
    epochs: int,
    batch: int,
    lr: float,
    label_frac: float,
    seed: int,
    out_dir: str | None,
    save_best: bool,
):
    n = len(X)
    m = max(1, int(n * label_frac))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)[:m]
    X = X[idx]
    y = y[idx]

    device = next(enc.parameters()).device
    reg = torch.nn.Linear(128, 1).to(device)
    opt = torch.optim.Adam(reg.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    dl = DataLoader(LabeledDataset(X, y), batch_size=batch, shuffle=True)

    best = {"rmse": float("inf"), "mae": float("inf"), "epoch": -1}
    best_state = None
    best_pred = None

    for epoch in range(epochs):
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
        print(f"Epoch {epoch+1}/{epochs} loss={total/len(dl):.4f}")

        # Evaluate on val each epoch
        reg.eval()
        with torch.no_grad():
            hv = enc(torch.from_numpy(Xv).to(device))
            pv = reg(hv).cpu().numpy().squeeze()
        mae = mean_absolute_error(yv, pv)
        rmse = np.sqrt(mean_squared_error(yv, pv))
        if rmse < best["rmse"]:
            best = {"rmse": float(rmse), "mae": float(mae), "epoch": epoch + 1}
            best_state = reg.state_dict()
            best_pred = pv

    if out_dir and save_best and best_state is not None:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(best_state, os.path.join(out_dir, "best_reg.pt"))
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(best, f, indent=2)

    return best, best_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--label_frac", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runs/downstream")
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save_pred", action="store_true")
    parser.add_argument("--curve", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    train = np.load(args.data)
    val = np.load(args.val)
    X = train["X"].astype(np.float32)
    y = train["y"].astype(np.float32)
    Xv = val["X"].astype(np.float32)
    yv = val["y"].astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = Encoder1D(in_ch=X.shape[2], out_dim=128).to(device)
    enc.load_state_dict(torch.load(args.ckpt, map_location=device))
    enc.eval()

    best, best_pred = train_regressor(
        enc=enc,
        X=X,
        y=y,
        Xv=Xv,
        yv=yv,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        label_frac=args.label_frac,
        seed=args.seed,
        out_dir=args.out,
        save_best=args.save_best,
    )
    print(f"Best Val MAE: {best['mae']:.4f}  RMSE: {best['rmse']:.4f} @ epoch {best['epoch']}")

    if best_pred is not None and (args.plot or args.save_pred or args.curve):
        os.makedirs(args.out, exist_ok=True)

    if args.save_pred and best_pred is not None:
        csv_path = os.path.join(args.out, "val_predictions.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "true", "pred"])
            for i, (t, p) in enumerate(zip(yv, best_pred)):
                writer.writerow([i, float(t), float(p)])
        print(f"Saved predictions: {csv_path}")

    if (args.plot or args.curve) and best_pred is not None:
        import matplotlib.pyplot as plt

    if args.plot and best_pred is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(yv, best_pred, s=8, alpha=0.6)
        lo = float(min(yv.min(), best_pred.min()))
        hi = float(max(yv.max(), best_pred.max()))
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)
        ax.set_xlabel("True SoH")
        ax.set_ylabel("Predicted SoH")
        ax.set_title("Downstream: Predicted vs True")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "pred_vs_true.png"), dpi=150)
        plt.close(fig)

    if args.curve and best_pred is not None:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(yv, label="True", linewidth=1.2)
        ax.plot(best_pred, label="Pred", linewidth=1.2)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("SoH")
        ax.set_title("Downstream: True vs Pred (val)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "pred_vs_true_curve.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
