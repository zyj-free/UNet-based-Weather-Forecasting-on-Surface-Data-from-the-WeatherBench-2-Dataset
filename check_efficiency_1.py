import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time
import os

from config import DEVICE, INPUT_FRAMES
from temporal_weight_unet_1 import (
    LightweightSharedWeightUNet_History,
    LightweightIndependentWeightUNet_History
)

# ===================== 1. 轻量化函数（绝对安全版） =====================
def apply_hardcore_lightweight(model):
    model.eval()
    # 最安全的 FP16 转换方式
    for param in model.parameters():
        param.data = param.data.half()
    return model

# ===================== 2. 评测函数（零报错） =====================
def evaluate_hardware(model, data_path, device, model_name=""):
    torch.cuda.empty_cache()

    # 加载数据
    x_np = np.load(os.path.join(data_path, "X_test.npy"))
    y_np = np.load(os.path.join(data_path, "y_test.npy"))
    x = torch.tensor(x_np[:4], dtype=torch.float32)
    y = torch.tensor(y_np[:4], dtype=torch.float32)

    model.eval()
    model.to(device)

    # ===================== 【终极修复】强制统一精度 =====================
    is_fp16 = next(model.parameters()).dtype == torch.float16
    if is_fp16:
        x = x.half().to(device)
        y = y.half().to(device)
    else:
        x = x.to(device)
        y = y.to(device)

    # 预热
    with torch.no_grad():
        _ = model(x[:1])

    # 测速
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        pred = model(x)
    torch.cuda.synchronize()
    infer_ms = (time.time() - t0) * 1000

    mem = torch.cuda.memory_allocated(device) / 1024**2

    # 计算指标时转回 float32，保证稳定
    pred = pred.float().squeeze(1)
    y = y.float().squeeze(1)

    rmse = torch.sqrt(torch.mean((pred - y) ** 2)).item()
    mae = torch.abs(pred - y).mean().item()
    r2 = r2_score(y.cpu().numpy().flatten(), pred.cpu().numpy().flatten())
    params = sum(p.numel() for p in model.parameters())

    return {
        "Model": model_name,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "Time(ms)": round(infer_ms, 2),
        "Params(M)": round(params / 1e6, 3),
        "GPU_Memory(MB)": round(mem, 2),
        "Throughput": round(x.shape[0] / (infer_ms / 1000), 2)
    }

# ===================== 3. 主程序 =====================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    PROCESSED_DATA_PATH = './processed_data/v_wind_2022_COV/'

    print(f"🚀 硬件轻量化评测（Windows 100% 可运行）")

    input_channels = 11
    target_channels = 2
    results = []

    models_to_test = [
        ("Model4 (Shared)", LightweightSharedWeightUNet_History),
        ("Model5 (Independent)", LightweightIndependentWeightUNet_History)
    ]

    for model_name, model_class in models_to_test:
        print(f"\n==================================================")
        print(f"评测: {model_name}")
        print(f"==================================================")

        # ========== 1. 原版 ==========
        model_orig = model_class(
            input_channels=input_channels,
            output_channels=target_channels,
            input_frames=INPUT_FRAMES
        ).to(device)

        res_orig = evaluate_hardware(model_orig, PROCESSED_DATA_PATH, device, f"{model_name}")
        results.append(res_orig)
        print(f"✅ 原版   | RMSE: {res_orig['RMSE']} | Time: {res_orig['Time(ms)']}ms")

        # ========== 2. FP16 轻量化 ==========
        model_fp16 = model_class(
            input_channels=input_channels,
            output_channels=target_channels,
            input_frames=INPUT_FRAMES
        )
        model_fp16 = apply_hardcore_lightweight(model_fp16).to(device)

        res_fp16 = evaluate_hardware(model_fp16, PROCESSED_DATA_PATH, device, f"{model_name} (FP16)")
        results.append(res_fp16)
        print(f"✅ FP16   | RMSE: {res_fp16['RMSE']} | Time: {res_fp16['Time(ms)']}ms")

        del model_orig, model_fp16
        torch.cuda.empty_cache()

    # ===================== 结果输出 =====================
    df = pd.DataFrame(results)

    def calc_metrics(row):
        base = row['Model'].replace(' (FP16)', '')
        if 'FP16' in row['Model']:
            base_row = df[df['Model'] == base].iloc[0]
            speedup = base_row['Time(ms)'] / row['Time(ms)']
            mem_saved = (1 - row['GPU_Memory(MB)'] / base_row['GPU_Memory(MB)']) * 100
            return pd.Series([round(speedup, 2), round(mem_saved, 1)])
        return pd.Series([1.0, 0.0])

    df[['SpeedUp', 'Mem_Saved(%)']] = df.apply(calc_metrics, axis=1)

    print("\n" + "=" * 115)
    print(f"{'Model':<27} {'Time(ms)':<10} {'GPU(MB)':<10} {'SpeedUp':<10} {'MemSaved':<10} {'RMSE':<10}")
    print("-" * 115)
    for _, row in df.iterrows():
        print(f"{row['Model']:<27} {row['Time(ms)']:<10} {row['GPU_Memory(MB)']:<10} {row['SpeedUp']:<10} {row['Mem_Saved(%)']:<10} {row['RMSE']:<10}")
    print("=" * 115)

    # 画图
    plt.figure(figsize=(14, 5))
    names = df['Model']
    times = df['Time(ms)']
    mems = df['GPU_Memory(MB)']

    plt.subplot(121)
    plt.bar(names, times, color=['#4285F4', '#EA4335', '#4285F4', '#EA4335'])
    plt.title('Inference Time (ms)')
    plt.xticks(rotation=30, ha='right')

    plt.subplot(122)
    plt.bar(names, mems, color=['#4285F4', '#EA4335'])
    plt.title('GPU Memory (MB)')
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig("hardware_benchmark.png", dpi=300)
    plt.show()

    df.to_csv("hardware_benchmark.csv", index=False, encoding="utf-8-sig")
    print("\n🎉 评测完成！")