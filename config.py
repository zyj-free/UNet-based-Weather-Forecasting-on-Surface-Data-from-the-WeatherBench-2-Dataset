
import torch
import torch.cuda as cuda
import os

# ==================== 设备设置 ====================
def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"✅ 使用 GPU: {gpu_name}")
        print(f"   GPU 显存: {gpu_memory:.1f} GB")
        print(f"   CUDA 版本: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("⚠️ 使用 CPU (未检测到 GPU)")

    return device



DEVICE = get_device()
# ==================== 数据参数 ====================
# Google Cloud Storage路径
DATA_PATH = "gs://weatherbench2/datasets/era5_daily/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"
#50
# LON_MIN, LON_MAX = 81.0, 127.5
# LAT_MIN, LAT_MAX = 10.5, 58.5

# LON_MIN, LON_MAX = 70.0, 141.5
# LAT_MIN, LAT_MAX = 3, 52

LON_MIN, LON_MAX = 70.0, 150
LAT_MIN, LAT_MAX = 0.0, 61.5

TIME_START = '2014-09-01'
TIME_END = '2020-12-31'

VARIABLES = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_temperature'
]

TARGET_VARIABLE = '2m_temperature'


#位势高度、温度、为相逢（U风），经向风（V风）

# ==================== 时间序列参数 ====================
INPUT_FRAMES = 7      # 用过去7个时间步
PRED_FRAMES = 1       # 预测未来1个时间步
TIME_STEP_HOURS = 24   # 数据时间间隔

# ==================== 训练参数 ====================
TRAIN_SPLIT = 0.8     # 训练集比例
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 模型参数 ====================
INPUT_CHANNELS = len(VARIABLES)  # 输入变量数

# ==================== 路径设置 ====================
PROCESSED_DATA_PATH = './processed_data/'
MODEL_SAVE_PATH = './saved_models_newyear/'
FIGURE_SAVE_PATH = './figures_early_stop_newyear/'

import os
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(FIGURE_SAVE_PATH, exist_ok=True)

# 打印配置信息
print("\n" + "="*50)
print("配置信息")
print("="*50)
print(f"设备: {DEVICE}")
print(f"数据路径: {DATA_PATH}")
print(f"区域: lon[{LON_MIN}, {LON_MAX}], lat[{LAT_MIN}, {LAT_MAX}]")
print(f"时间范围: {TIME_START} 到 {TIME_END}")
print(f"变量: {VARIABLES}")
print(f"Batch size: {BATCH_SIZE}")
print("="*50 + "\n")