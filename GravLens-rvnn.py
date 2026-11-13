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


def get_next_run_dir(base_dir="results", prefix="rv_fft_run"):
    """Find the next available numbered run folder (e.g., results/rv_fft_run_1)."""
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(f"{prefix}_")
    ]
    run_nums = []
    for d in existing:
        try:
            run_nums.append(int(d.split("_")[-1]))
        except ValueError:
            pass
    next_num = max(run_nums, default=0) + 1
    run_dir = os.path.join(base_dir, f"{prefix}_{next_num}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, next_num


def save_run_summary(run_dir, run_num, epoch_times, extra_metrics=None):
    """Save average epoch time and run metadata to summary.csv."""
    avg_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    summary_path = os.path.join(run_dir, "summary.csv")
    row = {
        "run": run_num,
        "num_epochs": len(epoch_times),
        "avg_epoch_time_sec": round(avg_time, 4),
        "total_time_sec": round(sum(epoch_times), 4),
    }
    if extra_metrics:
        row.update(extra_metrics)
    log_history(summary_path, row)
    print(f"Run {run_num} complete | Avg epoch time: {avg_time:.4f}s | Saved to {run_dir}")


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

    # Create a new numbered folder for this run (e.g., results/rv_fft_run_1)
    run_dir, run_num = get_next_run_dir(base_dir="results", prefix="rv_fft_run")
    history_path = os.path.join(run_dir, "loss_history.csv")
    epoch_times = []
    print(f"Starting Run {run_num} -> saving outputs to {run_dir}")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

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

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_duration = time.perf_counter() - t0
        epoch_times.append(epoch_duration)

        print(f"Epoch {epoch+1}/{epochs} ({epoch_duration:.1f}s) "
              f"train_loss={train_loss:.4e} val_loss={val_loss:.4e}")

        log_history(history_path, {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_sec": round(epoch_duration, 4),
        })

        if val_loss < best_vloss:
            best_vloss = val_loss
            fname = f"{save_dir}/lens_rv_fft_model-{epoch+1:04d}-vloss-{val_loss:.4e}.pth"
            torch.save(model, fname)
            print(f"  Saved new best model: {fname}")

    # Save average epoch runtime after all epochs finish
    save_run_summary(run_dir, run_num, epoch_times, extra_metrics={"best_val_loss": best_vloss})


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

