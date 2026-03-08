# UNet-based-Weather-Forecasting-on-Surface-Data-from-the-WeatherBench-2-Dataset


本项目基于 **UNet** 模型，利用 **WeatherBench 2 (ERA5_Daily)** 数据集进行表层气象数据预测。主要关注东亚及西太平洋区域（70°E-150°E, 0°N-61.5°N）的短期天气变化。

## 1. 数据源 (Data Source)

- **数据集名称**: ERA5_Daily (WeatherBench 2)
- **原始文件**: `1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr`
- **时间范围**: 2018-01-01 至 2019-12-31 (共 2 年)
- **地理范围**:
  - 经度: 70.0°E ~ 150.0°E
  - 纬度: 0.0°N ~ 61.5°N
- **分辨率**: 1.5°
- **网格尺寸**: 
  - 原始全局: 240 × 121
  - 裁剪后区域: 54 × 42

## 2. 输入与输出 (Input & Output)

> *注：请根据实际代码补充具体的变量名称（如：2m_temperature, total_precipitation 等）*

- **输入变量 (Input Variables)**: 
  - [在此处填写，例如：2m Temperature, U/V Wind Components]
- **输出变量 (Output Variables)**: 
  - [在此处填写，例如：Future 2m Temperature]
- **数据形状 (Shape)**: `(Batch, Time, Lat, Lon, Channels)`

## 3. 样本构建 (Sample Construction)

采用滑动窗口方式构建序列样本：

- **输入时间步 (Input Frames)**: 7 天 (`INPUT_FRAMES=7`)
- **预测时间步 (Prediction Frames)**: 1 天 (`PRED_FRAMES=1`)

### 数据集划分
总有效样本数：**723** (注意：下方训练/验证/测试集数量之和与总数存在差异，建议核对原始统计逻辑，此处按您提供的数据展示)

| 数据集 | 比例 | 时间范围 (估算) | 样本数量 |
| :--- | :--- | :--- | :--- |
| **训练集 (Train)** | 76% | 2018-01 ~ 2019-06 | 1753 |
| **验证集 (Val)** | 8% | 2019-07 ~ 2019-08 | 58 |
| **测试集 (Test)** | 16% | 2019-09 ~ 2019-12 | 116 |

*(注：若总样本数为723，则各分项数量可能存在统计口径差异，请以实际代码生成的 DataLoader 为准)*

## 4. 模型评价结果 (Evaluation Results)

在测试集上的整体表现如下：

| 指标 (Metric) | 数值 (Value) |
| :--- | :--- |
| **MSE** (均方误差) | 1.58 |
| **RMSE** (均方根误差) | 1.26 |
| **MAE** (平均绝对误差) | 0.89 |

## 5. 环境依赖 (Requirements)

```bash
pip install torch
pip install xarray
pip install zarr
# 添加其他需要的库
