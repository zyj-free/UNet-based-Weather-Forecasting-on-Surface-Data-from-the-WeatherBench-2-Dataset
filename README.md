# UNet-based-Weather-Forecasting-on-Surface-Data-from-the-WeatherBench-2-Dataset


本项目基于 **UNet** 模型，利用 **WeatherBench 2 (ERA5_Daily)** 数据集进行表层气象数据预测。主要关注东亚及西太平洋区域（70°E-150°E, 0°N-61.5°N）的短期天气变化。

## 1. 数据源 (Data Source)

- **数据集名称**: ERA5_Daily (WeatherBench 2)
- **原始文件**: `1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr`
- **时间范围**: 2014-09-01 至 2020-12-31
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
  - [10m_u_component_of_wind, 10m_v_component_of_wind, 2m_temperature]
- **输出变量 (Output Variables)**: 
  - [Future 2m Temperature]
- **数据形状 (Shape)**: `(Batch, Time, Lat, Lon, Channels)`

## 3. 样本构建 (Sample Construction)

采用滑动窗口方式构建序列样本：

- **输入时间步 (Input Frames)**: 7 天 (`INPUT_FRAMES=7`)
- **预测时间步 (Prediction Frames)**: 1 天 (`PRED_FRAMES=1`)

### 数据集划分
总有效样本数：**1927** (注意：下方训练/验证/测试集数量之和与总数存在差异，建议核对原始统计逻辑，此处按您提供的数据展示)

| 数据集 | 比例 | 时间范围 (估算) | 样本数量 |
| :--- | :--- | :--- | :--- |
| **训练集 (Train)** | 76% | 2014-09 ~ 2019-05 | 1753 |
| **验证集 (Val)** | 8% | 2019-06 ~ 2019-12 | 58 |
| **测试集 (Test)** | 16% | 2020-01 ~ 2020-12 | 116 |

*(注：若总样本数为723，则各分项数量可能存在统计口径差异，请以实际代码生成的 DataLoader 为准)*

## 4. 模型架构 (Model Architecture)

本项目核心采用 **UNet** 卷积神经网络架构，针对气象网格数据的特点进行了定制化改进，以支持非标准分辨率输入并防止过拟合。

### 4.1 网络结构细节
- **骨干网络**: 经典的 Encoder-Decoder 对称结构，包含 4 层下采样和 4 层上采样。
- **输入处理**: 
  - 输入张量形状: `(Batch, Time=7, Lat, Lon, Channels)`
  - **时空融合**: 在输入层将时间步 (`7 days`) 和通道维度合并，转化为多通道图像输入 `(Batch, 7*Channels, H, W)`。
- **特征提取 (Encoder)**: 
  - 每层包含两个 $3 \times 3$ 卷积块，后接 BatchNorm 和 ReLU 激活函数。
  - 使用 $2 \times 2$ 最大池化进行下采样。
- **瓶颈层 (Bottleneck)**: 深层特征提取，通道数扩展至 1024。
- **特征重建 (Decoder)**: 
  - 使用转置卷积 ($2 \times 2$ ConvTranspose) 进行上采样。
  - **跳跃连接 (Skip Connections)**: 将编码器对应层的特征图与解码器特征图拼接 (Concatenate)，保留高频空间细节。
- **输出层**: $1 \times 1$ 卷积，将通道数映射回预测时间步数 (`PRED_FRAMES=1`)。

### 4.2 关键改进策略
针对气象数据网格尺寸非 2 的幂次方（本项目为 **54×42**）的问题，模型实现了以下自适应机制：
1. **动态偶数填充 (Dynamic Even Padding)**: 在下采样前，自动检测高宽是否为奇数。若是，使用 `reflect` 模式进行单像素填充，确保每次池化后尺寸整除，避免维度丢失报错。
2. **尺寸对齐 (Size Matching)**: 在跳跃连接拼接前，强制使用双线性插值将上采样后的特征图尺寸对齐到编码器对应层的尺寸，消除因舍入误差导致的维度不匹配。
3. **原始尺寸裁剪**: 在最终输出前，裁剪掉为了计算而额外填充的像素，确保输出尺寸与原始输入区域严格一致。

### 4.3 训练策略：早停机制 (Early Stopping)
为防止模型在长时间训练中过拟合，引入了基于验证集损失的早停机制：
- **监控指标**: 验证集均方误差 (Validation MSE)。
- **耐心值 (Patience)**: 当验证集损失在连续 **N** 个 epoch 内未下降时，自动终止训练。
- **恢复权重**: 训练结束时，自动加载验证集损失最低时的模型权重作为最终模型。
- 
## 5. 模型评价结果 (Evaluation Results)

在测试集上的整体表现如下：

| 指标 (Metric) | 数值 (Value) |
| :--- | :--- |
| **MSE** (均方误差) | 1.58 |
| **RMSE** (均方根误差) | 1.26 |
| **MAE** (平均绝对误差) | 0.89 |

## 6. 环境依赖 (Requirements)

```bash
pip install torch
pip install xarray
pip install zarr
# 添加其他需要的库
