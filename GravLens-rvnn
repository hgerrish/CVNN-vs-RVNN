import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def log_history(path, row):
    """Append one epoch's metrics to a CSV loss-history file (writes header on first call)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def to_centered_fft_channels(x):
    """Convert (N, 1, H, W) real images into (N, 2, H, W) centered 2D FFT [real, imag]."""
    # Compute 2D FFT over spatial dims and shift zero-frequency component to center
    fft_c = torch.fft.fftshift(torch.fft.fft2(x.squeeze(1)), dim=(-2, -1))
    return torch.stack([fft_c.real, fft_c.imag], dim=1).to(torch.float32)


NUM_PARAMS = 4  # theta_E, f, eps_x, eps_y


class GravLensRealFFTCNN(nn.Module):
    """Real-valued CNN with BatchNorm operating on 2-channel (real, imag) centered 2D FFT."""

    def __init__(self, in_channels=2, num_params=NUM_PARAMS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(2),  # 64 -> 32

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2),  # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2),  # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_params),
            nn.Sigmoid(),  # targets are min-max normalized to [0,1]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.regressor(x)


def evaluate(model, X, y, batch_size=128):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].to(device)
            yb = y[i:i + batch_size].to(device)
            pred = model(xb)
            total_loss += criterion(pred, yb).item()
    return total_loss / (len(X) * y.shape[1])


def train_model(model, train_X, train_y, val_X, val_y, epochs, lr,
                batch_size=64, save_dir="saved_models/lens_rv_fft"):
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_vloss = float("inf")
    n = len(train_X)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        t0 = time.time()
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = train_X[idx].to(device)
            yb = train_y[idx].to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)

        train_loss = total_loss / n
        val_loss = evaluate(model, val_X, val_y)

        print(f"Epoch {epoch+1}/{epochs} ({time.time()-t0:.1f}s) "
              f"train_loss={train_loss:.4e} val_loss={val_loss:.4e}")

        log_history("results/lens_rv_fft_loss_history.csv", {
            "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
        })

        if val_loss < best_vloss:
            best_vloss = val_loss
            fname = f"{save_dir}/lens_rv_fft_model-{epoch+1:04d}-vloss-{val_loss:.4e}.pth"
            torch.save(model, fname)
            print(f"  Saved new best model: {fname}")


if __name__ == "__main__":
    DATA_DIR = "prepped_data/lens_rv"
    train_X_raw = torch.from_numpy(np.load(os.path.join(DATA_DIR, "train_X.npy"))).to(torch.float32)
    train_y = torch.from_numpy(np.load(os.path.join(DATA_DIR, "train_y.npy"))).to(torch.float32)
    val_X_raw = torch.from_numpy(np.load(os.path.join(DATA_DIR, "val_X.npy"))).to(torch.float32)
    val_y = torch.from_numpy(np.load(os.path.join(DATA_DIR, "val_y.npy"))).to(torch.float32)

    # Convert pixel images to centered 2D FFT (Real, Imag) channels
    train_X = to_centered_fft_channels(train_X_raw)
    val_X = to_centered_fft_channels(val_X_raw)

    print("train_X FFT shape:", train_X.shape, "val_X FFT shape:", val_X.shape)

    model = GravLensRealFFTCNN(in_channels=2)
    print("Model has", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")

    train_model(model, train_X, train_y, val_X, val_y, epochs=500, lr=0.001)
