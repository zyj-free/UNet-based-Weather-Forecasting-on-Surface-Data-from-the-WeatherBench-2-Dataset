
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

TIME_START = '2017-01-01'
TIME_END = '2022-12-31'

# ==================== 改进方案：加权验证集（推荐） ====================
# 问题：原来验证集只有1年，且2021和2022分布差异大（2022是超级EL Niño年）
# 解决：扩大验证集到1.5年，包含正常年份(2020下半年+2021)和测试年份相邻
# 
# 训练集：2017-01-01 到 2020-06-30 (3.5年稳定年份)
# 验证集：2020-07-01 到 2021-12-31 (1.5年，包含正常年份和异常年份)
# 测试集：2022-01-01 到 2022-12-31 (1年超级EL Niño年)
# ====================================================================

TIME_START_TRAIN = '2017-01-01'
TIME_END_TRAIN = '2020-06-30'      # 3.5年

TIME_START_VAL = '2020-07-01'
TIME_END_VAL = '2021-12-31'        # 1.5年（包含正常年份和异常年份）

TIME_START_TEST = '2022-01-01'
TIME_END_TEST = '2022-12-31'       # 1年超级EL Niño年

VARIABLES = [
    '10m_u_component_of_wind',        # [目标] 10米U向风速
    '10m_v_component_of_wind',        # [目标] 10米V向风速
    '2m_temperature',                # [状态] 2米气温 (影响大气稳定度)
    'mean_surface_sensible_heat_flux',# [驱动] 地表感热通量 (核心：热力湍流混合，直接影响风速垂直交换)
    'leaf_area_index_high_vegetation' # [地表] 高植被叶面积指数 (核心：地表粗糙度，决定摩擦力大小)
]
COVARIATE_RAW_VARS = [
    'mean_sea_level_pressure',       # 用于计算 MSLP 异常
    '2m_dewpoint_temperature',        # 用于计算湿度异常 (配合 2m_temperature)
    'boundary_layer_height',        # 举例：边界层高度
    'mean_surface_latent_heat_flux',            # [辅助] 潜热通量 (能量平衡)
    'mean_top_downward_short_wave_radiation_flux', # [辅助] 太阳短波辐射 (昼夜/季节信号)
    'mean_surface_net_long_wave_radiation_flux'     # [辅助] 净长波辐射 (夜间稳定度信号)
]

# ✅ 10m_u_component_of_wind
# ✅ 10m_v_component_of_wind
# ✅ 2m_temperature
# ✅ mean_sea_level_pressure       (新)
# ✅ 2m_dewpoint_temperature       (新)
# TARGET_VARIABLE = '2m_temperature'
# TARGET_VARIABLE = '10m_u_component_of_wind'
# TARGET_VARIABLE = '10m_v_component_of_wind'

TARGET_VARIABLES = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind'
]
# ==================== 协变量配置 ====================
# 是否添加协变量（MSLP异常 + 相对湿度指标）
ADD_COVARIATES = True   # 改为 False 可以快速测试

# 选择哪些协变量
# 默认选择：MSLP异常 和 相对湿度 (不加BLH保持速度)
COVARIATE_LIST = ['mslp_anom', 'humid_anom', 'blh', 'lhf_norm', 'sw_rad_norm', 'lw_rad_norm']
N_COVARIATES = len(COVARIATE_LIST) if ADD_COVARIATES else 0

# ==================== 时间序列参数 ====================
INPUT_FRAMES = 7                    # 用过去24小时 (4×6h)
PRED_FRAMES = 1                    # 预测未来6小时 (6h)
TIME_STEP_HOURS = 24   # 数据时间间隔
TIME_SAMPLING_STRIDE = 1

H = 80  # 根据您的数据设置 lat
W = 102  # 根据您的数据设置 lon
# ==================== 训练参数 ====================
TRAIN_SPLIT = 0.8     # 训练集比例
BATCH_SIZE = 4
# BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 5e-4
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_WORKERS = 4

# ==================== 模型参数 ====================
INPUT_CHANNELS = len(VARIABLES) + N_COVARIATES  # 输入变量数 + 协变量数
# INPUT_CHANNELS = len(VARIABLES)
OUTPUT_CHANNELS = len(TARGET_VARIABLES)  # 输出变量数
# ==================== 路径设置 ====================
PROCESSED_DATA_PATH = './processed_data/v_wind_2022_COV/'
# MODEL_SAVE_PATH = './saved_models/v_wind/unet/'
# FIGURE_SAVE_PATH = './figures/v_wind/unet/'
MODEL_SAVE_PATH = './saved_models/v_wind_2022_revise/unet/'
FIGURE_SAVE_PATH = './figures/v_wind_2022_revise/unet/'

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