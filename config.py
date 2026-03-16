
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
# DATA_PATH = "gs://weatherbench2/datasets/era5_daily/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"

# DATA_PATH = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr"
# weatherbench2/datasets/era5_daily/1959-2023_01_10-full_37-1h-512x256_equiangular_conservative.zarr
DATA_PATH = "gs://weatherbench2/datasets/era5_daily/1959-2023_01_10-full_37-1h-512x256_equiangular_conservative.zarr"
#50


LOCAL_CACHE_PATH = './data_cache'
# LON_MIN, LON_MAX = 81.0, 127.5
# LAT_MIN, LAT_MAX = 10.5, 58.5

# LON_MIN, LON_MAX = 70.0, 141.5
# LAT_MIN, LAT_MAX = 3, 52

# LON_MIN, LON_MAX = 70.0, 150
# LAT_MIN, LAT_MAX = 0.0, 61.5

LON_MIN, LON_MAX = 70.0, 141.75
LAT_MIN, LAT_MAX = 0.0, 56
# TIME_START = '2017-01-01'
# TIME_END = '2020-06-30'      # ✅ 改为 2020 上半年
#
# # ==================== 数据集划分 ====================
# TIME_START = '2016-01-01'
# TIME_END = '2020-12-31'
# # 训练集：4年 (2016-2019)
# #
# TIME_START_TRAIN = '2016-01-01'
# TIME_END_TRAIN = '2019-06-30'
#
# # 验证集：2020 上半年
# TIME_START_VAL = '2019-07-01'
# TIME_END_VAL = '2019-12-31'
#
# # 测试集：2020
# TIME_START_TEST = '2020-01-01'
# TIME_END_TEST = '2020-12-31'

TIME_START = '2017-01-01'
TIME_END = '2022-12-31'
# 训练集
#
# ==============滚动交叉验证（Rolling Window CV）================
USE_ROLLING_CV = True
# Fold 1: Train(2017), Val(2018)
# Fold 2: Train(2017-2018), Val(2019)
# Fold 3: Train(2017-2019), Val(2020)
# Fold 4: Train(2017-2020), Val(2021)
CV_VAL_YEARS = [2018, 2019, 2020, 2021]
TEST_YEAR = 2022

TIME_START_TRAIN = '2017-01-01'
TIME_END_TRAIN = '2020-12-31'

# 验证集
TIME_START_VAL = '2021-01-01'
TIME_END_VAL = '2021-12-31'

# 测试集
TIME_START_TEST = '2022-01-01'
TIME_END_TEST = '2022-12-31'
# --- 调试/快速实验配置 (推荐先用这个) ---
# 训练集：只用 1 年 (2019) -> 数据量减少 75%，内存压力骤减
# TIME_START_TRAIN = '2019-01-01'
# TIME_END_TRAIN = '2019-12-31'
# #
# # 验证集：2020 上半年 (保持不变，符合论文)
# TIME_START_VAL = '2020-01-01'
# TIME_END_VAL = '2020-05-31'
#
# # 测试集：2020 下半年 (保持不变，严格符合论文 WB2 评估标准)
# TIME_START_TEST = '2020-06-01'
# TIME_END_TEST = '2020-12-31'

VARIABLES = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_temperature'
]

TARGET_VARIABLE = '2m_temperature'
# TARGET_VARIABLE = '10m_u_component_of_wind'
# TARGET_VARIABLE = '10m_v_component_of_wind'

#位势高度、温度、为相逢（U风），经向风（V风）


# ==================== 时间序列参数 ====================
INPUT_FRAMES = 7                    # 用过去24小时 (4×6h)
PRED_FRAMES = 1                    # 预测未来6小时 (6h)
TIME_STEP_HOURS = 24                # 数据时间间隔
TIME_SAMPLING_STRIDE = 1

H = 80  # 根据您的数据设置 lat
W = 102  # 根据您的数据设置 lon
# ==================== 训练参数 ====================
TRAIN_SPLIT = 0.8     # 训练集比例
# BATCH_SIZE = 8
BATCH_SIZE = 32#unet fnn mlp
EPOCHS = 150
LEARNING_RATE = 5e-4
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_WORKERS = 8
# ==================== 模型参数 ====================
INPUT_CHANNELS = len(VARIABLES)  # 输入变量数

# ==================== 路径设置 ====================
PROCESSED_DATA_PATH = './processed_data/temperature_2022/'
# MODEL_SAVE_PATH = './saved_models/v_wind/unet/'
# FIGURE_SAVE_PATH = './figures/v_wind/unet/'
MODEL_SAVE_PATH = './saved_models/temperature_2022/unet/'
FIGURE_SAVE_PATH = './figures/temperature_2022/unet/'

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