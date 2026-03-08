"""
数据准备模块 - 下载、处理、归一化
"""
import xarray as xr
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
import gcsfs
from config import *


class WeatherDataPreparer:
    def __init__(self):
        self.fs = gcsfs.GCSFileSystem()
        self.scalers = {}

    def download_data(self):
        """下载原始数据"""
        print("=" * 50)
        print("开始下载数据...")
        print(f"数据源: {DATA_PATH}")
        print(f"区域: lon[{LON_MIN}, {LON_MAX}], lat[{LAT_MIN}, {LAT_MAX}]")
        print(f"时间: {TIME_START} 到 {TIME_END}")
        print("=" * 50)

        # 打开Zarr格式数据
        ds = xr.open_zarr(DATA_PATH, consolidated=True)

        # 选择数据
        data = ds.sel(
            time=slice(TIME_START, TIME_END),
            latitude=slice(LAT_MIN, LAT_MAX),
            longitude=slice(LON_MIN, LON_MAX),
        )[VARIABLES]

        print(f"\n数据下载完成！")
        print(f"数据维度: {data.dims}")
        print(f"时间点数: {len(data.time)}")
        print(f"空间网格: {len(data.latitude)} x {len(data.longitude)}")
        print(f"变量数: {len(data.data_vars)}")

        return data

    def create_samples(self, data):
        """创建训练样本（滑动窗口）"""
        print("\n" + "=" * 50)
        print("创建训练样本...")

        # 将数据转换为numpy数组
        # 形状: (时间, 纬度, 经度, 变量)
        X_data = []

        # 提取每个变量
        for var_name in VARIABLES:
            var_data = data[var_name].values  # (time, lat, lon)
            X_data.append(var_data)

        # 堆叠变量： (time, lat, lon, channels)
        X = np.stack(X_data, axis=-1)

        # 目标变量：500hPa位势高度
        y = data[TARGET_VARIABLE].values  # (time, lat, lon)

        print(f"原始数据形状: X = {X.shape}, y = {y.shape}")

        # 创建滑动窗口样本
        X_samples = []
        y_samples = []

        total_samples = len(data.time) - INPUT_FRAMES - PRED_FRAMES + 1

        for i in range(total_samples):
            # 输入：从i开始的INPUT_FRAMES个时间步
            X_sample = X[i:i + INPUT_FRAMES]  # (input_frames, lat, lon, channels)

            # 输出：从i+INPUT_FRAMES开始的PRED_FRAMES个时间步
            # 只取目标变量
            y_sample = y[i + INPUT_FRAMES:i + INPUT_FRAMES + PRED_FRAMES]  # (pred_frames, lat, lon)

            X_samples.append(X_sample)
            y_samples.append(y_sample)

        X_samples = np.array(X_samples)
        y_samples = np.array(y_samples)

        print(f"样本创建完成！")
        print(f"样本总数: {len(X_samples)}")
        print(f"输入形状: {X_samples.shape}")
        print(f"输出形状: {y_samples.shape}")

        return X_samples, y_samples

    def normalize_data(self, X, y):
        """数据归一化"""
        print("\n" + "=" * 50)
        print("数据归一化...")

        # 获取原始形状
        n_samples, n_frames, H, W, n_channels = X.shape

        # 重塑为2D数组用于归一化
        X_reshaped = X.reshape(-1, n_channels)
        y_reshaped = y.reshape(-1, 1)

        # 创建并拟合标准化器
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_normalized = X_scaler.fit_transform(X_reshaped).reshape(X.shape)
        y_normalized = y_scaler.fit_transform(y_reshaped).reshape(y.shape)

        # 保存标准化器
        self.scalers['input'] = X_scaler
        self.scalers['target'] = y_scaler

        print(f"归一化完成！")
        print(f"输入 - 均值: {X_scaler.mean_[:5]}...")
        print(f"输入 - 标准差: {X_scaler.scale_[:5]}...")

        return X_normalized, y_normalized

    def split_data(self, X, y):
        """划分训练集和验证集
        遵循 WeatherBench 2 标准：
        1. 严禁随机打乱 (No Random Shuffle)，防止时间序列数据泄露。
        2. 确保：训练集时间 < 验证集时间 < 测试集时间。
        3. 模拟真实场景：用过去的数据预测未来。

        划分策略 (基于总时长 2018-2019):
        训练集 (Train): 前 76% (约 2018.01 - 2019.06)
        验证集 (Val):   中间 8% (约 2019.07 - 2019.08) -> 用于调参和早停
        测试集 (Test):  最后 16% (约 2019.09 - 2019.12) -> 用于最终盲测

        """
        print("\n" + "=" * 50)
        print("划分数据集...")

        n_samples = len(X)

        # --- 配置划分比例 ---
        # 对应：1.5年训练，2个月验证，4个月测试
        train_ratio = 0.76
        val_ratio = 0.08

        # 计算切分索引
        train_end_idx = int(n_samples * train_ratio)
        val_end_idx = int(n_samples * (train_ratio + val_ratio))

        print(f"总样本数: {n_samples}")
        print(f"切分点: 训练截止@{train_end_idx}, 验证截止@{val_end_idx}")


        X_train = X[:train_end_idx]
        y_train = y[:train_end_idx]

        X_val = X[train_end_idx:val_end_idx]
        y_val = y[train_end_idx:val_end_idx]

        X_test = X[val_end_idx:]
        y_test = y[val_end_idx:]

        # 完整性检查
        total_split = len(X_train) + len(X_val) + len(X_test)
        assert total_split == n_samples, f"样本数量不匹配！{total_split} != {n_samples}"

        print(f"\n划分完成:")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_processed_data(self, X_train, X_val, y_train, y_val,X_test, y_test):
        """保存处理后的数据"""
        print("\n" + "=" * 50)
        print("保存处理后的数据...")

        # 保存numpy数组
        np.save(os.path.join(PROCESSED_DATA_PATH, 'X_train.npy'), X_train)
        np.save(os.path.join(PROCESSED_DATA_PATH, 'X_val.npy'), X_val)
        np.save(os.path.join(PROCESSED_DATA_PATH, 'X_test.npy'), X_test)

        np.save(os.path.join(PROCESSED_DATA_PATH, 'y_train.npy'), y_train)
        np.save(os.path.join(PROCESSED_DATA_PATH, 'y_val.npy'), y_val)
        np.save(os.path.join(PROCESSED_DATA_PATH, 'y_test.npy'), y_test)


        # 保存标准化器
        with open(os.path.join(PROCESSED_DATA_PATH, 'scalers.pkl'), 'wb') as f:
            pickle.dump(self.scalers, f)

        print(f"数据已保存到: {PROCESSED_DATA_PATH}")

    def load_processed_data(self):
        """加载处理后的数据"""
        print("\n" + "=" * 50)
        print("加载处理后的数据...")

        X_train = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_train.npy'))
        X_val = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_val.npy'))
        X_test = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_test.npy'))

        y_train = np.load(os.path.join(PROCESSED_DATA_PATH, 'y_train.npy'))
        y_val = np.load(os.path.join(PROCESSED_DATA_PATH, 'y_val.npy'))
        y_test = np.load(os.path.join(PROCESSED_DATA_PATH, 'y_test.npy'))


        with open(os.path.join(PROCESSED_DATA_PATH, 'scalers.pkl'), 'rb') as f:
            self.scalers = pickle.load(f)

        print(f"数据加载完成！")
        print(f"训练集: X {X_train.shape}, y {y_train.shape}")
        print(f"验证集: X {X_val.shape}, y {y_val.shape}")
        print(f"测试集: X {X_test.shape}, y {y_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def run_pipeline(self):
        """运行完整的数据处理流程"""
        print("\n" + "=" * 50)
        print("启动完整数据处理流程")
        print("=" * 50)

        # 1. 下载数据
        data = self.download_data()

        # 2. 创建样本
        X, y = self.create_samples(data)

        # 3. 归一化
        X_norm, y_norm = self.normalize_data(X, y)

        # 4. 划分数据集
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_norm, y_norm)
        # 固定年份划分
        # X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
        #     X, y,
        #     method='fixed_years',
        #     train_years= 17520,  # 2018-01-01 到 2019-12-31 (2年)
        #     val_years= 18768 # 2020-01-01 到 2020-02-29 (2个月)
        #     # 剩余全部是测试集 (2020-03-01 到 2020-12-31)
        # )

        # 5. 保存数据
        self.save_processed_data(X_train, X_val, y_train, y_val, X_test, y_test)

        print("\n" + "=" * 50)
        print("数据处理流程完成！")
        print("=" * 50)

        return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # 测试数据处理
    preparer = WeatherDataPreparer()
    X_train, X_val, X_test, y_train, y_val, y_test = preparer.run_pipeline()

    print(f"\n最终数据形状:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")