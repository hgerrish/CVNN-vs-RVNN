import csv
import json
import math
import os
import platform
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NUM_PARAMS = 4  # theta_E, f, eps_x, eps_y


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_hash():
    """Retrieve the current Git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "not-a-git-repo"


def save_metadata(run_dir, run_num, seed):
    """Save exact seed, hardware config, and commit hash to metadata.json."""
    metadata = {
        "run": run_num,
        "model_type": "GravLensComplexCNN_FFT",
        "seed": seed,
        "git_commit_hash": get_git_hash(),
        "hardware": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "cpu": platform.processor() or platform.machine(),
            "os": platform.platform(),
            "pytorch_version": torch.__version__,
        }
    }
    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)


def log_history(path, row):
    """Append one epoch's metrics to a CSV loss-history file (writes header on first call)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def get_next_run_dir(base_dir="results", prefix="cv_fft_run"):
    """Find the next available numbered run folder (e.g., results/cv_fft_run_1)."""
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


def save_run_summary(run_dir, run_num, seed, epoch_times, extra_metrics=None):
    """Save average epoch time and run metadata to summary.csv."""
    avg_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    summary_path = os.path.join(run_dir, "summary.csv")
    row = {
        "run": run_num,
        "seed": seed,
        "num_epochs": len(epoch_times),
        "avg_epoch_time_sec": round(avg_time, 4),
        "total_time_sec": round(sum(epoch_times), 4),
    }
    if extra_metrics:
        row.update(extra_metrics)
    log_history(summary_path, row)
    print(f"Run {run_num} (Seed {seed}) complete | Avg epoch time: {avg_time:.4f}s | Saved to {run_dir}")


def to_centered_complex_fft(x):
    """Convert (N, 1, H, W) real images into (N, 1, H, W) complex64 centered 2D FFT."""
    fft_c = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    return fft_c.to(torch.complex64)


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.padding = padding
        self.weight_real = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_imag = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias_real = nn.Parameter(torch.zeros(out_channels))
        self.bias_imag = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))

    def forward(self, x):
        real, imag = x.real, x.imag
        out_real = (F.conv2d(real, self.weight_real, self.bias_real, padding=self.padding)
                    - F.conv2d(imag, self.weight_imag, None, padding=self.padding))
        out_imag = (F.conv2d(real, self.weight_imag, self.bias_imag, padding=self.padding)
                    + F.conv2d(imag, self.weight_real, None, padding=self.padding))
        return torch.complex(out_real, out_imag)


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_real = nn.Parameter(torch.zeros(out_features))
        self.bias_imag = nn.Parameter(torch.zeros(out_features))
        std = math.sqrt(1.0 / in_features)
        nn.init.uniform_(self.weight_real, -std * math.sqrt(3), std * math.sqrt(3))
        nn.init.uniform_(self.weight_imag, -std * math.sqrt(3), std * math.sqrt(3))

    def forward(self, x):
        real, imag = x.real, x.imag
        out_real = torch.matmul(real, self.weight_real.t()) - torch.matmul(imag, self.weight_imag.t()) + self.bias_real
        out_imag = torch.matmul(real, self.weight_imag.t()) + torch.matmul(imag, self.weight_real.t()) + self.bias_imag
        return torch.complex(out_real, out_imag)


class CRELU(nn.Module):
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))


class ComplexAvgPool2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return torch.complex(F.avg_pool2d(x.real, self.kernel_size), F.avg_pool2d(x.imag, self.kernel_size))


class ComplexAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        real = F.adaptive_avg_pool2d(x.real, self.output_size)
        imag = F.adaptive_avg_pool2d(x.imag, self.output_size)
        return torch.complex(real, imag)


class ComplexBatchNorm(nn.Module):
    """Complex Batch Normalization (Trabelsi et al. 2018, "Deep Complex Networks")."""

    def __init__(self, num_features, spatial=True, eps=1e-5, momentum=0.1):
        super().__init__()
        self.spatial = spatial
        self.eps = eps
        self.momentum = momentum

        self.gamma_rr = nn.Parameter(torch.full((num_features,), 1.0 / math.sqrt(2)))
        self.gamma_ii = nn.Parameter(torch.full((num_features,), 1.0 / math.sqrt(2)))
        self.gamma_ri = nn.Parameter(torch.zeros(num_features))
        self.beta_real = nn.Parameter(torch.zeros(num_features))
        self.beta_imag = nn.Parameter(torch.zeros(num_features))

        self.register_buffer("running_mean_real", torch.zeros(num_features))
        self.register_buffer("running_mean_imag", torch.zeros(num_features))
        self.register_buffer("running_Vrr", torch.full((num_features,), 0.5))
        self.register_buffer("running_Vii", torch.full((num_features,), 0.5))
        self.register_buffer("running_Vri", torch.zeros(num_features))

    def _dims_and_shape(self):
        if self.spatial:
            return (0, 2, 3), (1, -1, 1, 1)
        return (0,), (1, -1)

    def forward(self, x):
        real, imag = x.real, x.imag
        dims, shape = self._dims_and_shape()

        if self.training:
            mean_r = real.mean(dim=dims)
            mean_i = imag.mean(dim=dims)
            xc = real - mean_r.view(*shape)
            yc = imag - mean_i.view(*shape)
            Vrr = (xc ** 2).mean(dim=dims) + self.eps
            Vii = (yc ** 2).mean(dim=dims) + self.eps
            Vri = (xc * yc).mean(dim=dims)

            with torch.no_grad():
                self.running_mean_real.mul_(1 - self.momentum).add_(self.momentum * mean_r)
                self.running_mean_imag.mul_(1 - self.momentum).add_(self.momentum * mean_i)
                self.running_Vrr.mul_(1 - self.momentum).add_(self.momentum * Vrr)
                self.running_Vii.mul_(1 - self.momentum).add_(self.momentum * Vii)
                self.running_Vri.mul_(1 - self.momentum).add_(self.momentum * Vri)
        else:
            mean_r, mean_i = self.running_mean_real, self.running_mean_imag
            Vrr, Vii, Vri = self.running_Vrr, self.running_Vii, self.running_Vri
            xc = real - mean_r.view(*shape)
            yc = imag - mean_i.view(*shape)

        tau = Vrr + Vii
        delta = Vrr * Vii - Vri ** 2
        s = torch.sqrt(torch.clamp(delta, min=self.eps))
        t = torch.sqrt(torch.clamp(tau + 2 * s, min=self.eps))
        rst = 1.0 / (s * t)
        Wrr = ((Vii + s) * rst).view(*shape)
        Wii = ((Vrr + s) * rst).view(*shape)
        Wri = (-Vri * rst).view(*shape)

        x_hat = Wrr * xc + Wri * yc
        y_hat = Wri * xc + Wii * yc

        gamma_rr = self.gamma_rr.view(*shape)
        gamma_ii = self.gamma_ii.view(*shape)
        gamma_ri = self.gamma_ri.view(*shape)
        beta_r = self.beta_real.view(*shape)
        beta_i = self.beta_imag.view(*shape)

        out_real = gamma_rr * x_hat + gamma_ri * y_hat + beta_r
        out_imag = gamma_ri * x_hat + gamma_ii * y_hat + beta_i
        return torch.complex(out_real, out_imag)


class GravLensComplexCNN(nn.Module):
    def __init__(self, num_params=NUM_PARAMS):
        super().__init__()
        self.features = nn.Sequential(
            ComplexConv2d(1, 16, kernel_size=3, padding=1), ComplexBatchNorm(16, spatial=True), CRELU(), ComplexAvgPool2d(2),
            ComplexConv2d(16, 32, kernel_size=3, padding=1), ComplexBatchNorm(32, spatial=True), CRELU(), ComplexAvgPool2d(2),
            ComplexConv2d(32, 64, kernel_size=3, padding=1), ComplexBatchNorm(64, spatial=True), CRELU(), ComplexAvgPool2d(2),
        )
        self.pool = ComplexAdaptiveAvgPool2d(1)
        self.fc1 = ComplexLinear(64, 64)
        self.bn_fc1 = ComplexBatchNorm(64, spatial=False)
        self.act = CRELU()
        self.fc2 = ComplexLinear(64, num_params)
        self.readout = nn.Linear(num_params * 2, num_params)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.act(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)
        logits = self.readout(torch.cat([x.real, x.imag], dim=-1))
        return torch.sigmoid(logits)


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
    return total_loss / (len(X) * y.shape)


def train_model(model, train_X, train_y, val_X, val_y, epochs, lr, seed,
                batch_size=64, base_save_dir="saved_models/lens_cv",
                start_epoch=0, best_vloss=float("inf"),
                ckpt_prefix="lens_cv_model"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n = len(train_X)
    total_epochs = start_epoch + epochs

    # Create a new numbered folder for this run (e.g., results/cv_fft_run_1)
    run_dir, run_num = get_next_run_dir(base_dir="results", prefix="cv_fft_run")

    # Save metadata.json immediately to verify seed, hardware, and git commit
    save_metadata(run_dir, run_num, seed)

    # Isolate model checkpoints per run/seed so trials don't overwrite each other
    save_dir = os.path.join(base_save_dir, f"run_{run_num}_seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    history_path = os.path.join(run_dir, "loss_history.csv")
    epoch_times = []
    print(f"\nStarting Run {run_num} (Seed {seed}) -> saving outputs to {run_dir}")

    for epoch in range(start_epoch, total_epochs):
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

        print(f"[Seed {seed}] Epoch {epoch+1}/{total_epochs} ({epoch_duration:.1f}s) "
              f"train_loss={train_loss:.4e} val_loss={val_loss:.4e}")

        log_history(history_path, {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_sec": round(epoch_duration, 4),
        })

        if val_loss < best_vloss:
            best_vloss = val_loss
            fname = f"{save_dir}/{ckpt_prefix}-{epoch+1:04d}-vloss-{val_loss:.4e}.pth"
            torch.save(model, fname)
            print(f"  Saved new best model: {fname}")

    save_run_summary(run_dir, run_num, seed, epoch_times, extra_metrics={"best_val_loss": best_vloss})


if __name__ == "__main__":
    DATA_DIR = "prepped_data/lens_rv"
    train_X_raw = torch.from_numpy(np.load(os.path.join(DATA_DIR, "train_X.npy"))).to(torch.float32)
    train_y = torch.from_numpy(np.load(os.path.join(DATA_DIR, "train_y.npy"))).to(torch.float32)
    val_X_raw = torch.from_numpy(np.load(os.path.join(DATA_DIR, "val_X.npy"))).to(torch.float32)
    val_y = torch.from_numpy(np.load(os.path.join(DATA_DIR, "val_y.npy"))).to(torch.float32)

    train_X = to_centered_complex_fft(train_X_raw)
    val_X = to_centered_complex_fft(val_X_raw)
    print("train_X complex FFT shape:", train_X.shape, "val_X complex FFT shape:", val_X.shape)

    SEEDS = [42, 123, 456, 789, 1024]

    for seed in SEEDS:
        set_seed(seed)
        model = GravLensComplexCNN()
        if seed == SEEDS:
            print("Model has", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")

        train_model(model, train_X, train_y, val_X, val_y, epochs=500, lr=0.001, seed=seed)
