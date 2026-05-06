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
    x = torch.tensor(x_np[:100], dtype=torch.float32) 
    y = torch.tensor(y_np[:100], dtype=torch.float32)
    
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

    ## ===================== 4. 结果展示 =====================
df = pd.DataFrame(results)

def calc_metrics(row):
    base_name = row['Model'].replace(' (FP16)', '')
    if '(FP16)' in row['Model']:
        base_row = df.loc[df['Model'] == base_name].iloc[0]
        # 修正：加速比 = 原版时间 / 新版时间
        speed_up = base_row['Time(ms)'] / row['Time(ms)'] 
        # 显存节省百分比
        mem_saved = (base_row['GPU_Memory(MB)'] - row['GPU_Memory(MB)']) / base_row['GPU_Memory(MB)'] * 100
        return pd.Series({'SpeedUp': round(speed_up, 2), 'Mem_Saved(%)': round(mem_saved, 1)})
    else:
        return pd.Series({'SpeedUp': 1.0, 'Mem_Saved(%)': 0.0})

df[['SpeedUp', 'Mem_Saved(%)']] = df.apply(calc_metrics, axis=1)

print("\n" + "="*120)
print(f"{'Model':<25} {'Time(ms)':<10} {'GPU_Mem(MB)':<14} {'SpeedUp':<10} {'Mem_Saved(%)':<12} {'RMSE':<10}")
print("-"*120)
for _, row in df.iterrows():
    print(f"{row['Model']:<25} {row['Time(ms)']:<10} {row['GPU_Memory(MB)']:<14} {row['SpeedUp']:<10} {row['Mem_Saved(%)']:<12} {row['RMSE']:<10}")
print("="*120)

# ===================== 5. 专业级绘图 (包含显存 Memory) =====================
plt.style.use('seaborn-v0_8-whitegrid') # 专业风格

# 准备数据
names = df['Model'].tolist()
rmses = df['RMSE'].tolist()
speeds = df['Time(ms)'].tolist()
mems = df['GPU_Memory(MB)'].tolist() # <-- 新增：显存数据

plt.figure(figsize=(18, 5))

# 图1: RMSE (精度)
plt.subplot(131)
bars = plt.bar(names, rmses, color='#4c72b0', alpha=0.8)
plt.title("RMSE (Accuracy)", fontsize=14, pad=20)
plt.ylabel("RMSE Value")
plt.xticks(rotation=45, ha='right')
# 柱顶标注数值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.0005, 
             f'{yval:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 图2: 推理速度 (Latency)
plt.subplot(132)
bars = plt.bar(names, speeds, color='#55a868', alpha=0.8)
plt.title("Inference Latency (ms)", fontsize=14, pad=20)
plt.ylabel("Time (ms)")
plt.xticks(rotation=45, ha='right')
# 柱顶标注数值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, 
             f'{yval:.2f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 图3: 显存占用 (Memory) - 重点补充
plt.subplot(133)
bars = plt.bar(names, mems, color='#c44e52', alpha=0.8) 
plt.title("GPU Memory Footprint (MB)", fontsize=14, pad=20)
plt.ylabel("Memory (MB)")
plt.xticks(rotation=45, ha='right')
# 柱顶标注数值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, 
             f'{int(yval)}MB', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout(pad=3.0) # 增加子图间距，防止文字重叠
save_path_img = "hardware_benchmark_detailed.png"
plt.savefig(save_path_img, dpi=300, bbox_inches='tight') # 高清保存
print(f"📊 对比图已保存: {save_path_img}")
plt.show()

# 保存 CSV
df.to_csv("hardware_benchmark.csv", index=False)
print("✅ 评测完成！")