import glob
import importlib.util
import json
import os
import re
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load GravLens-rvnn.py cleanly despite the hyphen in the filename
script_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(script_dir, "GravLens-rvnn.py")
spec = importlib.util.spec_from_file_location("GravLens_rvnn", module_path)
rvnn_mod = importlib.util.module_from_spec(spec)
sys.modules["GravLens_rvnn"] = rvnn_mod
spec.loader.exec_module(rvnn_mod)

GravLensRealFFTCNN = rvnn_mod.GravLensRealFFTCNN
to_centered_fft_channels = rvnn_mod.to_centered_fft_channels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def find_best_model(save_dir="saved_models/lens_rv_fft"):
    files = glob.glob(os.path.join(save_dir, "lens_rv_fft_model-*.pth"))
    if not files:
        raise FileNotFoundError(
            f"No trained models found in '{save_dir}/'. Run GravLens-rvnn.py first to train the model."
        )
    def loss_from_name(p):
        m = re.search(r"vloss-([0-9eE+\-.]+)\.pth$", p)
        return float(m.group(1)) if m else float("inf")
    return min(files, key=loss_from_name)


MODEL_PATH = os.environ.get("MODEL_PATH") or find_best_model()
print(f"Using model : {MODEL_PATH}")

model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model = model.to(device)
model.eval()

with open("prepped_data/lens_scaler.json") as f:
    scaler = json.load(f)
param_names = scaler["param_names"]
y_min = np.array(scaler["y_min"])
y_max = np.array(scaler["y_max"])

test_X_raw = torch.from_numpy(np.load("prepped_data/lens_rv/test_X.npy")).to(torch.float32)
test_X = to_centered_fft_channels(test_X_raw)
test_y_norm = np.load("prepped_data/lens_rv/test_y.npy")
print("Test set FFT shape:", test_X.shape)

preds_norm = []
with torch.no_grad():
    for i in range(0, len(test_X), 128):
        preds_norm.append(model(test_X[i:i + 128].to(device)).cpu().numpy())
preds_norm = np.concatenate(preds_norm, axis=0)

# Un-normalize back to physical units
def denorm(y_norm):
    return y_norm * (y_max - y_min) + y_min

y_true = denorm(test_y_norm)
y_pred = denorm(preds_norm)

# R^2 and median relative error per parameter
print("\nPer-parameter performance:")
r2_scores, rel_errors = [], []
for j, name in enumerate(param_names):
    yt, yp = y_true[:, j], y_pred[:, j]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rel_err = np.median(np.abs((yp - yt) / yp)) * 100
    r2_scores.append(r2)
    rel_errors.append(rel_err)
    print(f"  {name:10s}: R2={r2:.4f}  median relative error={rel_err:.2f}%")

# Scatter plots: true vs predicted
os.makedirs("results", exist_ok=True)
fig, axs = plt.subplots(1, len(param_names), figsize=(4.5 * len(param_names), 4.5))
for j, name in enumerate(param_names):
    ax = axs[j]
    yt, yp = y_true[:, j], y_pred[:, j]
    ax.scatter(yt, yp, s=4, alpha=0.4, color="tab:green")
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, "r-", linewidth=1)
    ax.set_xlabel("True value")
    ax.set_ylabel("Prediction")
    ax.set_title(f"{name}\nR2={r2_scores[j]:.3f}, err={rel_errors[j]:.1f}%")
plt.tight_layout()
out_path = "results/gravlens_rv_fft_scatter.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved scatter plot to {out_path}")
