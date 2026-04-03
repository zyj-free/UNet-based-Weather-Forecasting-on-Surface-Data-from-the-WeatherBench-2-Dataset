import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time
import os
import pickle

from config import DEVICE, INPUT_FRAMES
from temporal_weight_unet_1 import (
    LightweightSharedWeightUNet_History,
    LightweightIndependentWeightUNet_History
)

# ===================== 1. 硬核轻量化函数 =====================
def apply_hardcore_lightweight(model, mode="fp16"):
    model.eval()
    if mode == "fp16":
        print("   ⚡ 正在应用 FP16 半精度...")
        # 再次强调使用 .to 方法
        model = model.to(torch.float16)
    return model

# ===================== 2. 硬件级评测函数 =====================
def evaluate_hardware(model, data_path, device, model_name=""):
    torch.cuda.empty_cache()
    
    # 1. 加载原始数据 (永远是 float32)
    x_np = np.load(os.path.join(data_path, "X_test.npy"))
    y_np = np.load(os.path.join(data_path, "y_test.npy"))
    x = torch.tensor(x_np[:4], dtype=torch.float32) 
    y = torch.tensor(y_np[:4], dtype=torch.float32)
    
    model.eval()
    model.to(device)
    
    # 🔥🔥🔥 核心修复：无条件对齐模型精度 🔥🔥🔥
    # 获取模型权重的数据类型 (如果是 FP16 模型，这里就是 torch.float16)
    model_dtype = next(model.parameters()).dtype
    
    # 将输入数据强制转换为模型的数据类型
    x_in = x.to(device).to(model_dtype)
    y_in = y.to(device).to(model_dtype)
    
    print(f"      [调试] 模型精度: {model_dtype}, 输入数据精度: {x_in.dtype}") # 调试打印

    # 2. 预热
    with torch.no_grad():
        _ = model(x_in[:1])
    
    # 3. 记录显存
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(device) / 1024**2

    # 4. 计时
    t0 = time.time()
    with torch.no_grad():
        pred = model(x_in)
    torch.cuda.synchronize()
    infer_ms = (time.time() - t0) * 1000
    
    # 5. 记录显存
    mem_after = torch.cuda.memory_allocated(device) / 1024**2
    peak_memory = max(mem_before, mem_after)

    # --- 精度计算 (转回 float32) ---
    pred = pred.float()
    y_gt = y_in.float()
    pred = pred.squeeze(1)
    y_gt = y_gt.squeeze(1)

    rmse = torch.sqrt(torch.mean((pred-y_gt)**2)).item()
    mae = torch.abs(pred-y_gt).mean().item()
    r2 = r2_score(y_gt.cpu().numpy().flatten(), pred.cpu().numpy().flatten())

    params = sum(p.numel() for p in model.parameters())
    
    return {
        "Model": model_name,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "Time(ms)": round(infer_ms, 2),
        "Params(M)": round(params/1e6, 3),
        "GPU_Memory(MB)": round(peak_memory, 2),
        "Throughput": round(x.shape[0] / (infer_ms/1000), 2)
    }

# ===================== 3. 主程序 =====================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    PROCESSED_DATA_PATH = './processed_data/v_wind_2022_COV/'
    
    print(f"🚀 正在运行硬件级轻量化评测 (设备: {device})")
    
    input_channels = 11 
    target_channels = 2 
    
    results = []

    models_to_test = [
        ("Model4 (Shared)", LightweightSharedWeightUNet_History),
        ("Model5 (Independent)", LightweightIndependentWeightUNet_History)
    ]

    for model_name, model_class in models_to_test:
        print(f"\n{'='*50}")
        print(f"正在评测组: {model_name}")
        print(f"{'='*50}")

        # 1. 原版
        print(f"-> 初始化 {model_name} (Original)...")
        model_orig = model_class(input_channels, output_channels=target_channels, input_frames=INPUT_FRAMES)
        model_orig = model_orig.to(device)
        
        res_orig = evaluate_hardware(model_orig, PROCESSED_DATA_PATH, device, model_name=f"{model_name}")
        results.append(res_orig)
        print(f"   ✅ 原版完成. RMSE: {res_orig['RMSE']}, Time: {res_orig['Time(ms)']}ms")

        # 2. FP16 版
        print(f"-> 初始化 {model_name} (FP16)...")
        model_fp16 = model_class(input_channels, output_channels=target_channels, input_frames=INPUT_FRAMES)
        
        # 先转到 CPU 进行精度转换（更稳妥）
        model_fp16 = apply_hardcore_lightweight(model_fp16, mode="fp16")
        # 再转到 GPU
        model_fp16 = model_fp16.to(device) 
        
        res_fp16 = evaluate_hardware(model_fp16, PROCESSED_DATA_PATH, device, model_name=f"{model_name} (FP16)")
        results.append(res_fp16)
        print(f"   ✅ FP16版完成. RMSE: {res_fp16['RMSE']}, Time: {res_fp16['Time(ms)']}ms")
        
        # 清理
        del model_orig, model_fp16
        torch.cuda.empty_cache()

    # ===================== 4. 结果展示 =====================
    df = pd.DataFrame(results)
    
    def calc_metrics(row):
        base_name = row['Model'].replace(' (FP16)', '')
        if '(FP16)' in row['Model']:
            base_row = df.loc[df['Model'] == base_name].iloc[0]
            speed_up = row['Time(ms)'] / base_row['Time(ms)']
            mem_saved = (1 - row['GPU_Memory(MB)'] / base_row['GPU_Memory(MB)']) * 100
            return pd.Series({'SpeedUp': round(speed_up, 2), 'Mem_Saved(%)': round(mem_saved, 1)})
        else:
            return pd.Series({'SpeedUp': 1.0, 'Mem_Saved(%)': 0.0})

    df[['SpeedUp', 'Mem_Saved(%)']] = df.apply(calc_metrics, axis=1)

    print("\n" + "="*110)
    print(f"{'Model':<25} {'Time(ms)':<10} {'GPU_Mem(MB)':<12} {'SpeedUp':<10} {'Mem_Saved(%)':<12} {'RMSE':<10}")
    print("-"*110)
    for _, row in df.iterrows():
        print(f"{row['Model']:<25} {row['Time(ms)']:<10} {row['GPU_Memory(MB)']:<12} {row['SpeedUp']:<10} {row['Mem_Saved(%)']:<12} {row['RMSE']:<10}")
    print("="*110)

    # 画图
    plt.figure(figsize=(15, 5))
    names = df['Model'].tolist()
    times = df['Time(ms)'].tolist()
    mems = df['GPU_Memory(MB)'].tolist()
    
    plt.subplot(121)
    bars = plt.bar(names, times, color=["#447adb", "#447adb", "#e45649", "#e45649"])
    plt.title("Inference Latency (ms)")
    plt.ylabel("Time (ms)")
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(122)
    bars = plt.bar(names, mems, color=["#447adb", "#447adb", "#e45649", "#e45649"])
    plt.title("GPU Memory Footprint (MB)")
    plt.ylabel("Memory (MB)")
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("hardware_benchmark.png", dpi=300)
    plt.show()

    df.to_csv("hardware_benchmark.csv", index=False)
    print("✅ 评测完成！")