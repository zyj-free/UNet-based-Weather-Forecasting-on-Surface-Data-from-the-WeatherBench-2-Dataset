import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
from thop import profile
from sklearn.metrics import r2_score
import torch.nn.functional as F

# from config import INPUT_CHANNELS, OUTPUT_CHANNELS
INPUT_CHANNELS = 11  
OUTPUT_CHANNELS = 2  
# 导入你现有的所有模型
from temporal_weight_unet_1 import (
    LightweightUNet,
    LightweightSharedWeightUNet,
    LightweightIndependentWeightUNet,
    LightweightSharedWeightUNet_History,
    LightweightIndependentWeightUNet_History
)
PROCESSED_DATA_PATH = './processed_data/v_wind_2022_COV/'
base_dir = "./benchmark_results_advanced/efficiency"
os.makedirs(base_dir, exist_ok=True)
def get_next_filename(directory, prefix="run"):
    """自动获取下一个可用的文件名编号"""
    existing_files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".xlsx")]
    if not existing_files:
        return f"{prefix}_0.xlsx"
    
    # 提取现有的编号
    numbers = []
    for f in existing_files:
        try:
            # 提取 run_X.xlsx 中的 X
            num = int(f.replace(prefix, "").replace(".xlsx", "").replace("_", ""))
            numbers.append(num)
        except ValueError:
            continue
            
    next_num = max(numbers) + 1 if numbers else 0
    return f"{prefix}_{next_num}.xlsx"

next_file = get_next_filename(base_dir)
save_path_xlsx = os.path.join(base_dir, next_file)
save_path_img = os.path.join(base_dir, f"plot_{next_file.replace('.xlsx', '.png')}")

print(f"📂 结果将保存到: {base_dir}")
print(f"📄 本次文件名: {next_file}")
# ===================== 1. 统一评测函数 =====================
def evaluate_model(model, x, y, device="cuda"):
    # 1. 强制清理显存
    torch.cuda.empty_cache()
    
    model.eval()
    model.to(device)
    
    # 2. 数据维度安全检查
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
    
    # 限制 Batch Size 防止 OOM
    batch_size = 1 if x.shape[0] > 4 else x.shape[0]
    x_in = x[:batch_size].to(device)
    y_in = y[:batch_size].to(device)

    # 推理速度
    start = time.time()
    with torch.no_grad():
        pred = model(x_in)
    infer_time = (time.time() - start) * 1000

    # 计算指标
    pred = pred.squeeze(1)
    y_gt = y_in.squeeze(1)

    rmse = torch.sqrt(torch.mean((pred - y_gt) ** 2)).item()
    mae = torch.mean(torch.abs(pred - y_gt)).item()
    r2 = r2_score(y_gt.cpu().numpy().flatten(), pred.cpu().numpy().flatten())

    # --- 修改点：使用原生方法统计参数量 ---
    # thop 不支持 Mamba，所以用 PyTorch 原生方法统计 Params
    # 注意：这里不计算 FLOPs，因为 Mamba 的 FLOPs 计算需要专门的库，thop 算出来是错的
    params = sum(p.numel() for p in model.parameters())
    
    # 为了兼容你的 Excel 表格，这里给 FLOPs 一个占位符，或者你可以尝试估算
    # 如果你想尝试计算 FLOPs，需要安装 'fvcore' 库，但这比较复杂
    flops = 0 

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "time_ms": infer_time,
        "params(M)": params / 1e6, # 转换为百万
        "flops(G)": flops / 1e9,   # 暂时为 0
    }

# ===================== 2. 实例化所有你现有的模型 =====================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 你现有的4个模型，直接对比！
models = {
    "1": LightweightUNet(input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS),
    "2": LightweightSharedWeightUNet(input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS),
    "3": LightweightIndependentWeightUNet(input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS),
    "4": LightweightSharedWeightUNet_History(input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS),
    "5": LightweightIndependentWeightUNet_History(input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS),
}


# ==================================================
print("📂 正在加载真实测试数据...")
# 构造测试数据
x_np = np.load(os.path.join(PROCESSED_DATA_PATH, "X_test.npy"))
y_np = np.load(os.path.join(PROCESSED_DATA_PATH, "y_test.npy"))
print(f"原始 X 形状: {x_np.shape}")
print(f"原始 y 形状: {y_np.shape}")

with open(os.path.join(PROCESSED_DATA_PATH, "scalers.pkl"), "rb") as f:
    scalers = pickle.load(f)
    print("🔍 Scaler 文件内的所有键名是：", scalers.keys()) # 这行最关键
    print("📄 Scaler 文件内容预览：", scalers)
scaler_x = scalers['input'] 
scaler_y = scalers['target']

# 转换为 Tensor (假设数据已经是 B, T, H, W, C 格式)
x = torch.tensor(x_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)


# ===================== 3. 统一评测 =====================
results = {}
for name, model in models.items():
    print(f"Evaluating {name}...")
    
    # 每次评测前清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    try:
        res = evaluate_model(model, x, y, device)
        results[name] = res
    except RuntimeError as e:
        print(f"❌ 模型 {name} 评测失败: {e}")
        # 失败时记录空数据或跳过
        results[name] = {"rmse": 0, "mae": 0, "r2": 0, "time_ms": 0, "params(M)": 0, "flops(G)": 0}
if results:
    df = pd.DataFrame.from_dict(results, orient='index')
    cols = ['params(M)', 'flops(G)', 'time_ms', 'rmse', 'mae', 'r2']
    df = df[cols]
    df.to_excel(save_path_xlsx)
    print(f"\n✅ Excel 表格已保存: {save_path_xlsx}")
# ===================== 4. 输出表格 =====================
print("\n===== 模型统一评测结果 =====")
for name, res in results.items():
    print(f"[{name}]")
    print(f" RMSE: {res['rmse']:.4f} | MAE: {res['mae']:.4f} | R2: {res['r2']:.4f}")
    print(f" Params: {res['params(M)']:.2f}M | FLOPs: {res['flops(G)']:.2f}G")
    print(f" Speed: {res['time_ms']:.2f}ms\n")

# ===================== 5. 画对比图 =====================
def plot_comparison(results):
    plt.style.use('seaborn-v0_8-whitegrid')
    names = list(results.keys())
    rmses = [results[n]["rmse"] for n in names]
    speeds = [results[n]["time_ms"] for n in names]
    params = [results[n]["params(M)"] for n in names]

    plt.figure(figsize=(12, 4))
    
    # 图1: RMSE
    plt.subplot(131)
    bars = plt.bar(names, rmses, color='#4c72b0')
    plt.title("RMSE (Lower is Better)", fontsize=12)
    plt.ylabel("Value")
    plt.xticks(rotation=45, ha='right')
    # 在柱子上标数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, round(yval, 3), ha='center', va='bottom', fontsize=8)

    # 图2: 推理速度
    plt.subplot(132)
    bars = plt.bar(names, speeds, color='#55a868')
    plt.title("Inference Speed (ms) (Lower is Better)", fontsize=12)
    plt.ylabel("Time (ms)")
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, round(yval, 2), ha='center', va='bottom', fontsize=8)

    # 图3: 参数量
    plt.subplot(133)
    bars = plt.bar(names, params, color='#c44e52')
    plt.title("Parameters (M) (Lower is Better)", fontsize=12)
    plt.ylabel("Params (Millions)")
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, round(yval, 2), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path_img, dpi=300) # 高分辨率保存
    print(f"📊 对比图已保存: {save_path_img}")
    plt.show()

plot_comparison(results)