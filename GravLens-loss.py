import csv
import os
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_loss_csv(path):
    """Read epoch, train_loss, and val_loss columns from a log_history CSV file."""
    if not os.path.exists(path):
        print(f"Skipping (not found): {path}")
        return None
    epochs, train_loss, val_loss = [], [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
    return {"epochs": epochs, "train_loss": train_loss, "val_loss": val_loss}


# Map display labels to the CSV paths written by your training scripts
runs = {
    "RVNN (FFT + BatchNorm)": "results/lens_rv_fft_loss_history.csv",
    "CVNN (FFT + ComplexBatchNorm)": "results/lens_cv_loss_history.csv",
    "RVNN (Pixel Baseline)": "results/lens_rv_loss_history.csv",
}

loaded = {label: data for label, path in runs.items() if (data := load_loss_csv(path)) is not None}

if not loaded:
    raise FileNotFoundError("No loss history CSVs found in 'results/'. Check your CSV filenames.")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

for label, data in loaded.items():
    axs[0].plot(data["epochs"], data["train_loss"], label=label, linewidth=1.5)
    axs[1].plot(data["epochs"], data["val_loss"], label=label, linewidth=1.5)

    best_idx = min(range(len(data["val_loss"])), key=lambda i: data["val_loss"][i])
    print(f"{label:32s} | Best Val Loss: {data['val_loss'][best_idx]:.4e} (Epoch {data['epochs'][best_idx]})")

for ax, title in zip(axs, ["Training Loss (MSE)", "Validation Loss (MSE)"]):
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()

axs[0].set_ylabel("MSE Loss (log scale)")

plt.tight_layout()
out_path = "results/loss_curves_comparison.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved loss curve comparison plot to {out_path}")
