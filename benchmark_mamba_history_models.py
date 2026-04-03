"""
Advanced benchmark with two new improvements while keeping three baselines:
1. UNet
2. SharedWeightUNet
3. ChannelWeightUNet
4. MambaLiteSharedWeightUNet
5. HistoryWeightedSharedWeightUNet
6. MambaLiteHistoryWeightedSharedWeightUNet

Notes:
- "MambaLite" here is a lightweight Mamba-inspired temporal mixer.
- It does not rely on external mamba_ssm dependency.
- History weighting is learned by a small network from normalized sample age.
"""

import copy
import csv
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from config import DEVICE, BATCH_SIZE, PROCESSED_DATA_PATH,INPUT_FRAMES
from models import UNet
from temporal_weight_unet_1 import MinimalTimeAwareUNet, ChannelTimeAwareUNet,LightweightUNet,LightweightSharedWeightUNet, LightweightIndependentWeightUNet
from temporal_weight_unet_1 import LightweightSharedWeightUNet_History, LightweightIndependentWeightUNet_History
from scientific_evaluator import ScientificEvaluator

import torch.onnx
from prettytable import PrettyTable
import onnxruntime as ort
import time
import seaborn as sns
import matplotlib.pyplot as plt


RESULTS_DIR = "./benchmark_results_advanced"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class IndexedTensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        n = len(X)
        self.index_ratio = torch.linspace(0.0, 1.0, steps=n).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.index_ratio[idx]


class TemporalMixerBlock(nn.Module):
    #  A lightweight temporal mixer inspired by Mamba-style mixing, but simplified for our use case.
    def __init__(self, in_channels, hidden_dim=32):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, hidden_dim)
        self.dw_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, C)
        h = self.in_proj(x)
        h = F.gelu(h)
        h = h.transpose(1, 2)  # (B, hidden, T)
        h = self.dw_conv(h)
        h = h.transpose(1, 2)  # (B, T, hidden)
        g = torch.sigmoid(self.gate(h))
        h = h * g
        logits = self.out_proj(h).squeeze(-1)  # (B, T)
        return logits
def load_processed_data():
    print("\n" + "=" * 100)
    print("Loading processed data")
    print("=" * 100)

    arrays = {}
    for name in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
        path = os.path.join(PROCESSED_DATA_PATH, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        arrays[name] = np.load(path)
        print(f"{name}: {arrays[name].shape}")

    arrays["input_channels"] = arrays["X_train"].shape[-1]
    arrays["target_channels"] = arrays["y_train"].shape[-1] if arrays["y_train"].ndim == 5 else 1
    print(f"Input channels: {arrays['input_channels']}")
    print(f"Target channels: {arrays['target_channels']}")
    return arrays


def build_loader(X, y, batch_size=BATCH_SIZE, shuffle=False,num_workers=0):
    dataset = IndexedTensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,drop_last=True)


def squeeze_time_dim(tensor, name):
    """
    Expect prediction/target shapes:
    - single target: (B, 1, H, W) -> (B, H, W)
    - multi target:  (B, 1, H, W, C) -> (B, H, W, C)
    """
    if tensor.ndim not in (4, 5):
        raise ValueError(f"{name} has unexpected shape: {tuple(tensor.shape)}")
    if tensor.shape[1] != 1:
        raise ValueError(f"{name} expects time dimension=1, got shape: {tuple(tensor.shape)}")
    return tensor.squeeze(1)


def evaluate_rmse(model, loader):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for X, y, _ in loader:
            X = X.to(DEVICE)
            y = squeeze_time_dim(y.to(DEVICE), "target")
            pred = squeeze_time_dim(model(X), "prediction")
            loss = F.mse_loss(pred, y)
            total_loss += loss.item() * X.shape[0]
            total_samples += X.shape[0]
    return np.sqrt(total_loss / total_samples)


def collect_predictions(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y, _ in loader:
            X = X.to(DEVICE)
            y = squeeze_time_dim(y.to(DEVICE), "target")
            pred = squeeze_time_dim(model(X), "prediction")
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    return np.concatenate(preds).flatten(), np.concatenate(targets).flatten()


def build_model_family(input_channels, target_channels):
    return {
        # "UNet": UNet(input_channels=input_channels, output_channels=target_channels).to(DEVICE),
        # "SharedWeightUNet": MinimalTimeAwareUNet(
        #     input_channels=input_channels, output_channels=target_channels
        # ).to(DEVICE),
        # "ChannelWeightUNet": ChannelTimeAwareUNet(
        #     input_channels=input_channels, output_channels=target_channels
        # ).to(DEVICE),
        "1_LightweightUNet": LightweightUNet(input_channels, output_channels=target_channels).to(DEVICE),
        "2_LightSharedWeight": LightweightSharedWeightUNet(input_channels, output_channels=target_channels, input_frames=INPUT_FRAMES).to(DEVICE),
        "3_LightIndepWeight": LightweightIndependentWeightUNet(input_channels, output_channels=target_channels, input_frames=INPUT_FRAMES).to(DEVICE),
        "4_LightSharedWeight_History": LightweightSharedWeightUNet_History(input_channels, output_channels=target_channels, input_frames=INPUT_FRAMES).to(DEVICE),
        "5_LightIndepWeight_History": LightweightIndependentWeightUNet_History(input_channels, output_channels=target_channels, input_frames=INPUT_FRAMES).to(DEVICE),
    }


def train_one_model(model, model_name, train_loader, val_loader, num_epochs=40, lr=1e-4, patience=6):
    print("\n" + "=" * 100)
    print(f"Training {model_name}")
    print("=" * 100)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_state = None
    best_val_rmse = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for X, y, index_ratio in train_loader:
            X = X.to(DEVICE)
            y = squeeze_time_dim(y.to(DEVICE), "target")
            index_ratio = index_ratio.to(DEVICE)

            optimizer.zero_grad()
            pred = squeeze_time_dim(model(X), "prediction")

            if hasattr(model, "compute_history_weights"):
                per_sample_loss = F.mse_loss(pred, y, reduction="none")
                reduce_dims = tuple(range(1, per_sample_loss.ndim))
                per_sample_loss = per_sample_loss.mean(dim=reduce_dims)
                sample_weights = model.compute_history_weights(index_ratio)
                loss = (per_sample_loss * sample_weights).mean()
            else:
                loss = F.mse_loss(pred, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.shape[0]
            total_samples += X.shape[0]

        train_rmse = np.sqrt(total_loss / total_samples)
        val_rmse = evaluate_rmse(model, val_loader)
        scheduler.step(val_rmse)

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"Train RMSE {train_rmse:.5f} | Val RMSE {val_rmse:.5f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch + 1}, best epoch = {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_rmse, best_epoch, params


def summarize_metrics(title, metrics_list):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    print(f"{'Model':<36} {'RMSE':<10} {'MAE':<10} {'Bias':<10} {'R2':<10} {'ExtremeRMSE':<12}")
    print("-" * 120)
    for metrics in metrics_list:
        print(
            f"{metrics['model_name']:<36} "
            f"{metrics['rmse']:<10.5f} "
            f"{metrics['mae']:<10.5f} "
            f"{metrics['bias']:+<10.5f} "
            f"{metrics['r_squared']:<10.5f} "
            f"{metrics['rmse_extreme']:<12.5f}"
        )


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {path}")


def save_metric_plots(results, path_prefix):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skip plots.")
        return

    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)

    model_names = [item["model_name"] for item in results]
    test_rmse = [item["test_metrics"]["rmse"] for item in results]
    extreme_rmse = [item["test_metrics"]["rmse_extreme"] for item in results]
    params_m = [item["params"] / 1e6 for item in results]

    plt.figure(figsize=(10, 5))
    plt.bar(model_names, test_rmse)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Test RMSE")
    plt.title("Model Comparison: Test RMSE")
    plt.tight_layout()
    plt.savefig(f"{path_prefix}_test_rmse.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(model_names, extreme_rmse)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Extreme RMSE")
    plt.title("Model Comparison: Extreme RMSE")
    plt.tight_layout()
    plt.savefig(f"{path_prefix}_extreme_rmse.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for item, x, y in zip(results, params_m, test_rmse):
        plt.scatter(x, y, s=80)
        plt.annotate(item["model_name"], (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel("Parameters (Millions)")
    plt.ylabel("Test RMSE")
    plt.title("Params vs Test RMSE")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{path_prefix}_params_vs_rmse.png", dpi=180)
    plt.close()
    print(f"Saved figures with prefix: {path_prefix}")

def plot_optimization_results(results, save_dir):
    """生成优化前后的对比图"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. 文件大小对比
    sizes = [r['size_mb'] for r in results]
    names = [r['name'] for r in results]
    axes[0].bar(names, sizes, color=['#4c72b0', '#55a868', '#c44e52'])
    axes[0].set_title('模型文件大小对比 (MB)', fontsize=14)
    axes[0].set_ylabel('大小 (MB)')
    for i, v in enumerate(sizes):
        axes[0].text(i, v + 0.1, f'{v:.2f}', ha='center')

    # 2. 推理延迟对比
    latencies = [r['latency_ms'] for r in results]
    axes[1].bar(names, latencies, color=['#4c72b0', '#55a868', '#c44e52'])
    axes[1].set_title('推理延迟对比 (ms/batch)', fontsize=14)
    axes[1].set_ylabel('时间 (ms)')
    for i, v in enumerate(latencies):
        axes[1].text(i, v + 0.1, f'{v:.2f}', ha='center')

    # 3. 精度损失 (RSE/RMSE 变化)
    # 假设原始模型指标为 1.0，看量化后偏离多少
    accs = [r['acc_ratio'] for r in results] 
    axes[2].bar(names, accs, color=['#4c72b0', '#55a868', '#c44e52'])
    axes[2].axhline(1.0, color='red', linestyle='--', label='原始精度基线')
    axes[2].set_title('相对精度保持率 (1.0 = 无损)', fontsize=14)
    axes[2].set_ylabel('精度比率')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "optimization_comparison.png"))
    plt.close()
    print(f"📊 对比图已保存至: {os.path.join(save_dir, 'optimization_comparison.png')}")

def run_benchmark(num_epochs=120, batch_size=BATCH_SIZE, lr=1e-4, seed=42):
    set_seed(seed)
    arrays = load_processed_data()
    from config import NUM_WORKERS

    # train_loader = build_loader(arrays["X_train"], arrays["y_train"], batch_size=batch_size, shuffle=True)
    # val_loader = build_loader(arrays["X_val"], arrays["y_val"], batch_size=batch_size, shuffle=False)
    # test_loader = build_loader(arrays["X_test"], arrays["y_test"], batch_size=batch_size, shuffle=False)
    train_loader = build_loader(arrays["X_train"], arrays["y_train"], batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = build_loader(arrays["X_val"], arrays["y_val"], batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = build_loader(arrays["X_test"], arrays["y_test"], batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    evaluator = ScientificEvaluator("targets")
    models = build_model_family(arrays["input_channels"], arrays["target_channels"])

    results = []
    for model_name, model in models.items():
        trained_model, best_val_rmse, best_epoch, params = train_one_model(
            model, model_name, train_loader, val_loader, num_epochs=num_epochs, lr=lr
        )
        os.makedirs("./saved_models", exist_ok=True)
        # 建议加上时间戳或者直接用 model_name，避免覆盖
        save_path = f"./saved_models/{model_name}_final.pth"
        torch.save(trained_model.state_dict(), save_path)
        print(f"✅ 模型权重已保存: {save_path}")
        val_preds, val_targets = collect_predictions(trained_model, val_loader)
        test_preds, test_targets = collect_predictions(trained_model, test_loader)

        val_metrics = evaluator.evaluate_comprehensive(val_preds, val_targets, f"{model_name} [Val]")
        test_metrics = evaluator.evaluate_comprehensive(test_preds, test_targets, f"{model_name} [Test]")

        results.append({
            "model_name": model_name,
            "params": params,
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        })

    summarize_metrics("Validation Summary", [item["val_metrics"] for item in results])
    summarize_metrics("Test Summary", [item["test_metrics"] for item in results])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows = []
    detailed_rows = []
    for item in results:
        summary_rows.append({
            "model_name": item["model_name"],
            "params": item["params"],
            "best_epoch": item["best_epoch"],
            "best_val_rmse": item["best_val_rmse"],
            "test_rmse": item["test_metrics"]["rmse"],
            "test_mae": item["test_metrics"]["mae"],
            "test_bias": item["test_metrics"]["bias"],
            "test_r2": item["test_metrics"]["r_squared"],
            "test_extreme_rmse": item["test_metrics"]["rmse_extreme"],
        })
        for split_name in ["val_metrics", "test_metrics"]:
            metrics = item[split_name]
            detailed_rows.append({
                "model_name": metrics["model_name"],
                "split": "val" if split_name == "val_metrics" else "test",
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "bias": metrics["bias"],
                "r_squared": metrics["r_squared"],
                "rmse_extreme": metrics["rmse_extreme"],
                "rmse_extreme_p99": metrics["rmse_extreme_p99"],
                "mae_extreme": metrics["mae_extreme"],
                "samples_extreme_ratio": metrics["samples_extreme_ratio"],
            })

    save_csv(summary_rows, os.path.join(RESULTS_DIR, f"advanced_summary_{timestamp}.csv"))
    save_csv(detailed_rows, os.path.join(RESULTS_DIR, f"advanced_detailed_{timestamp}.csv"))
    save_metric_plots(results, os.path.join(RESULTS_DIR, f"advanced_plots_{timestamp}"))

    best = min(results, key=lambda x: x["test_metrics"]["rmse"])
    # =======================================================
    print("\n" + "=" * 120)
    print("Conclusion")
    print("=" * 120)
    print(f"Best test RMSE: {best['model_name']} ({best['test_metrics']['rmse']:.5f})")
    print(f"Current target shape is compatible with y=(N, 1, H, W, C) and multi-target C={arrays['target_channels']}.")
    print("New improvements included:")
    print("- LightweightIndependentWeightUNet: learned history sample weighting for shared weights")
    print("- LightweightIndependentWeightUNet_History: learned history sample weighting for independent weights")


if __name__ == "__main__":
    print("\n" + "=" * 120)
    print("Advanced Benchmark: Baselines + MambaLite + History Weighting")
    print("=" * 120)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    run_benchmark(num_epochs=120, batch_size=BATCH_SIZE, lr=1e-4, seed=42)
