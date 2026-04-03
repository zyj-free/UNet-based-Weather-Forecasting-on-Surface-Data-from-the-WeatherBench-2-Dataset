"""
数据准备模块 - 下载、处理、归一化
"""
import os
import sys
# ============ Clash 代理配置 ============
# 根据你的 Clash 实际端口修改
PROXY_HOST = '127.0.0.1'
PROXY_PORT = '7890'  # 如果是混合端口改为 7893

# 设置环境变量（影响大部分 Python 库）
os.environ['HTTP_PROXY'] = f'http://{PROXY_HOST}:{PROXY_PORT}'
os.environ['HTTPS_PROXY'] = f'http://{PROXY_HOST}:{PROXY_PORT}'

print(f"✅ 已设置代理：http://{PROXY_HOST}:{PROXY_PORT}")
import xarray as xr
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
import gcsfs
from config import *
from tqdm import tqdm
import dask
from numpy.lib.stride_tricks import sliding_window_view
import gc  # 垃圾回收
import pandas as pd
from pathlib import Path
import time
# 在文件开头添加
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
        # print(f"目标变量：{TARGET_VARIABLES}")
        print("=" * 50)
        # 打开Zarr格式数据
        ds = xr.open_zarr(DATA_PATH, consolidated=True)
        time_slice = ds.time.sel(time=slice(TIME_START, TIME_END))
        print(f"需要的时间范围：{len(time_slice)} 天")
        
        # ======== 确定要下载的所有变量 ========
        download_vars = list(VARIABLES)
        print(f"🔍 [DEBUG] COVARIATE_RAW_VARS 完整内容: {COVARIATE_RAW_VARS}")
        print(f"🔍 [DEBUG] COVARIATE_RAW_VARS 长度: {len(COVARIATE_RAW_VARS)}")
        
        if ADD_COVARIATES:
            # 确保 COVARIATE_RAW_VARS 已定义
            if 'COVARIATE_RAW_VARS' not in globals():
                raise NameError("未定义 COVARIATE_RAW_VARS。请在 config 中列出需要下载的原始变量名。")
            
            covariate_raw_vars = list(COVARIATE_RAW_VARS)
            
            print(f"ℹ️ 检测到协变量模式开启。将额外下载以下原始变量用于计算协变量：")
            for var in covariate_raw_vars:
                if var not in download_vars:
                    download_vars.append(var)
                    print(f"   + {var}")
                else:
                    print(f"   - {var} (已在基础列表中)")
        
        print(f"\n📥 最终下载的变量列表 (共 {len(download_vars)} 个):")
        for v in download_vars:
            print(f"   - {v}")
        print()
        
        # 选择数据
        data = ds.sel(
            time=slice(TIME_START, TIME_END),
            latitude=slice(LAT_MIN, LAT_MAX),
            longitude=slice(LON_MIN, LON_MAX),
        )[download_vars]

        print(f"\n✅ 数据下载完成！")
        print(f"数据维度: {data.dims}")
        print(f"时间点数: {len(data.time)}")
        print(f"空间网格: {len(data.latitude)} x {len(data.longitude)}")
        print(f"变量数: {len(data.data_vars)}")
        print(f"包含的变量: {list(data.data_vars)}")

        return data


    # def create_samples(self, data):
    #     """创建训练样本（滑动窗口）"""
    #     print("\n" + "=" * 50)
    #     print("创建训练样本...")
    #     X_data = []
    #     # 提取每个变量
    #     for var_name in VARIABLES:
    #         print(f"  正在提取变量 {var_name}")
    #         var_data = data[var_name].values
    #         var_dims = data[var_name].dims
    #         if var_dims == ('time', 'longitude', 'latitude'):
    #             print(f"    ⚠️  检测到维度顺序是 (time, lon, lat)，转置为 (time, lat, lon)")
    #             var_data = var_data.transpose(0, 2, 1)  # (time, lon, lat) -> (time, lat, lon)
    #         elif var_dims == ('time', 'latitude', 'longitude'):
    #             print(f"    ✅ 维度顺序正确: (time, lat, lon)")
    #         else:
    #             print(f"    ⚠️  未知维度顺序: {var_dims}")
    #         X_data.append(var_data)
    #         print(f"    完成，形状: {var_data.shape}，内存占用: {var_data.nbytes / 1024**2:.1f} MB")
    #
    #     # 堆叠变量： (time, lat, lon, channels)
    #     X = np.stack(X_data, axis=-1)
    #     del X_data  # 释放内存
    #     gc.collect()
    #
    #     y = data[TARGET_VARIABLE].values
    #     y_dims = data[TARGET_VARIABLE].dims
    #
    #     # === 修复：确保 y 维度顺序是 (time, lat, lon) ===
    #     if y_dims == ('time', 'longitude', 'latitude'):
    #         print(f"⚠️  y 维度顺序是 (time, lon, lat)，转置为 (time, lat, lon)")
    #         y = y.transpose(0, 2, 1)
    #
    #     print(f"原始数据形状: X = {X.shape}, y = {y.shape}")
    #
    #     # 创建滑动窗口样本
    #     X_samples = []
    #     y_samples = []
    #
    #     total_samples = len(data.time) - INPUT_FRAMES - PRED_FRAMES + 1
    #     if TIME_SAMPLING_STRIDE > 1:
    #         sampled_samples = (total_samples + TIME_SAMPLING_STRIDE - 1) // TIME_SAMPLING_STRIDE
    #         print(f"   采样后样本数：{sampled_samples}")
    #
    #     # 1. 创建输入 X 的滑动窗口
    #     print(f"   创建 X 滑动窗口...")
    #     X_windows = sliding_window_view(X, window_shape=INPUT_FRAMES, axis=0)
    #     # 形状：(total_samples, INPUT_FRAMES, lat, lon, channels)
    #     X_samples = X_windows[:total_samples]
    #
    #     # 2. 创建输出 y 的滑动窗口
    #     print(f"   创建 y 滑动窗口...")
    #     y_windows = sliding_window_view(y, window_shape=PRED_FRAMES, axis=0)
    #     # 形状：(total_samples, PRED_FRAMES, lat, lon)
    #      # 注意：y 的窗口需要偏移 INPUT_FRAMES
    #     y_samples = y_windows[INPUT_FRAMES:INPUT_FRAMES + total_samples]
    #
    #     # 3. 时间采样（如果配置了步长）
    #     if TIME_SAMPLING_STRIDE > 1:
    #         print(f"   应用时间采样步长 {TIME_SAMPLING_STRIDE}...")
    #         X_samples = X_samples[::TIME_SAMPLING_STRIDE]
    #         y_samples = y_samples[::TIME_SAMPLING_STRIDE]
    #
    #         # 释放临时窗口视图
    #     del X_windows, y_windows
    #     gc.collect()
    #
    #
    #     print(f"样本创建完成！")
    #     print(f"样本总数: {len(X_samples)}")
    #     print(f"输入形状: {X_samples.shape}")
    #     print(f"输出形状: {y_samples.shape}")
    #
    #     return X_samples, y_samples

    def create_samples(self, data):
        """创建训练样本（滑动窗口）- 输出维度 (batch, frames, H, W, channels)"""
        import numpy as np
        from numpy.lib.stride_tricks import sliding_window_view
        import gc

        print("\n" + "=" * 50)
        print("创建训练样本...")
        print("=" * 50)

        # ================= 配置 =================
        # 每次读取的时间步数。
        # 如果依然卡顿或内存爆，请减小这个数字 (例如 50, 20)
        # 如果内存充裕，可以增大 (例如 200, 500)
        CHUNK_SIZE = 100
        total_time_steps = len(data.time)
        # =======================================

        # 将数据转换为 numpy 数组
        # 形状：(时间，纬度，经度，变量)
        X_data = []
        y_data = None

        # 提取每个变量
        for var_idx, var_name in enumerate(VARIABLES, 1):
            print(f"\n[{var_idx}/{len(VARIABLES)}] 正在提取变量 {var_name}")

            # 1. 获取 DataArray
            var_arr = data[var_name]
            var_dims = var_arr.dims

            # 2. 统一维度顺序为 (time, latitude, longitude)
            # 统一维度顺序为 (time, latitude, longitude)
            target_dims = ('time', 'latitude', 'longitude')
            if var_dims == target_dims:
                print(f"    ✅  维度正确：{target_dims}")
                # 不需要做任何事
            else:
                # 检查是否包含所有目标维度
                if not set(target_dims).issubset(set(var_dims)):
                    raise ValueError(f"变量 {var_name} 的维度 {var_dims} 缺少必要的维度 {target_dims}")
                
                print(f"    ⚠️  维度转置：{var_dims} -> {target_dims}")
                # 直接传入维度名称元组，xarray 会自动处理重排
                var_arr = var_arr.transpose(*target_dims)

            # 3. 重新分块 (Re-chunking)
            # 原始 Zarr 文件的 chunk 可能很小（例如只有 1 个时间步），导致读取碎片化。
            print(f"    -> 正在重组数据块 (Chunking)...")
            var_chunked = var_arr.chunk({'time': -1})

            # 4. 并行计算并转为 NumPy
            print(f"    -> 正在并行读取并加载到内存 (Computing)...")
            var_data = var_chunked.compute()

            X_data.append(var_data)
            print(f"    ✅ 完成，形状：{var_data.shape}，内存占用：{var_data.nbytes / 1024 ** 2:.1f} MB")

        # 堆叠变量：(time, lat, lon, channels) ⭐ channels 在最后
        print(f"\n合并所有变量...")
        X = np.stack(X_data, axis=-1)  # axis=-1 确保 channels 在最后
        del X_data
        gc.collect()
        print(f"  X 形状：{X.shape} (time, lat, lon, channels)")

        # ================= 处理多目标变量 y =================
        print(f"\n处理多目标变量: {TARGET_VARIABLES}...")
        
        y_data_list = []
        
        for var_name in TARGET_VARIABLES:
            print(f"  -> 提取目标：{var_name}")
            y_arr = data[var_name]
            y_dims = y_arr.dims
            
            # 统一维度
            target_dims = ('time', 'latitude', 'longitude')
            if y_dims != target_dims:
                y_arr = y_arr.transpose(*target_dims)
            
            # 分块并计算
            y_chunked = y_arr.chunk({'time': -1})
            y_np = y_chunked.compute()
            y_data_list.append(y_np)
        
        # 堆叠所有目标变量：(Time, Lat, Lon, Target_Channels)
        # 假设 TARGET_VARIABLES 有 2 个，这里 axis=-1 会生成 (Time, Lat, Lon, 2)
        Y_all = np.stack(y_data_list, axis=-1)
        print(f"  原始 Y 形状：{Y_all.shape} (Time, Lat, Lon, Target_Chan={len(TARGET_VARIABLES)})")

        # ================= 创建滑动窗口样本 =================
        # 计算总样本数
        total_samples = total_time_steps - INPUT_FRAMES - PRED_FRAMES + 1

        if total_samples <= 0:
            raise ValueError(
                f"时间步数不足！数据有 {total_time_steps} 步，但需要 "
                f"INPUT_FRAMES({INPUT_FRAMES}) + PRED_FRAMES({PRED_FRAMES}) = "
                f"{INPUT_FRAMES + PRED_FRAMES} 步"
            )

        print(f"\n预计生成样本数：{total_samples}")
        print("正在使用向量化方法构建滑动窗口...")

        # 1. 处理输入 X
        # 在时间轴 (axis=0) 上创建长度为 INPUT_FRAMES 的滑动窗口
        # 结果形状：(总时间-INPUT_FRAMES+1, INPUT_FRAMES, lat, lon, channels)
        print(f"  创建 X 滑动窗口...")
        X_all_windows = sliding_window_view(X, window_shape=INPUT_FRAMES, axis=0)
        # 截取我们需要的部分 (去掉后面不足以构成预测帧的部分)
        X_samples = X_all_windows[:total_samples]
        if X_samples.shape[1] != INPUT_FRAMES:
            # 检测到 Frames 不在第 2 维，假设它在最后
            X_samples = np.moveaxis(X_samples, -1, 1)
        print(f"    X 窗口形状：{X_samples.shape} (N, T, H, W, C)")

        # 2. 处理输出 y
        # 先在时间轴上创建长度为 PRED_FRAMES 的滑动窗口
        # 结果形状：(总时间-PRED_FRAMES+1, PRED_FRAMES, lat, lon)
        # 2. 处理输出 y
        # 先在时间轴上创建长度为 PRED_FRAMES 的滑动窗口
        print(f"  创建 y 滑动窗口...")
        y_all_windows = sliding_window_view(Y_all, window_shape=PRED_FRAMES, axis=0)
        # y 需要偏移 INPUT_FRAMES，使输入输出对齐
        y_samples = y_all_windows[INPUT_FRAMES: INPUT_FRAMES + total_samples]
        if y_samples.shape[1] != PRED_FRAMES:
            y_samples = np.moveaxis(y_samples, -1, 1)
        print(f"    y 窗口形状：{y_samples.shape} (N, T, H, W, C)")

        # 3. 应用时间采样步长
        if TIME_SAMPLING_STRIDE > 1:
            print(f"\n应用时间采样步长：{TIME_SAMPLING_STRIDE}...")
            original_count = len(X_samples)
            X_samples = X_samples[::TIME_SAMPLING_STRIDE]
            y_samples = y_samples[::TIME_SAMPLING_STRIDE]
            print(f"  采样前：{original_count} 样本 -> 采样后：{len(X_samples)} 样本")


        # ================= ⭐ 关键：保持维度 (N, T, H, W, C) 不做转置 =================
        print(f"\n" + "=" * 50)
        print("✅ 维度确认：保持 (batch, frames, H, W, channels)")
        print("=" * 50)

        # X 保持 (N, T, H, W, C) - 不做转置
        print(f"  X 最终形状：{X_samples.shape}")
        print(f"    - batch (样本数): {X_samples.shape[0]}")
        print(f"    - frames (时间步): {X_samples.shape[1]}")
        print(f"    - H (纬度):       {X_samples.shape[2]}")
        print(f"    - W (经度):       {X_samples.shape[3]}")
        print(f"    - channels (变量): {X_samples.shape[4]}")

        # y 保持 (N, T, H, W, C) - 不做转置
        print(f"  y 最终形状：{y_samples.shape}")
        print(f"    - batch (样本数): {y_samples.shape[0]}")
        print(f"    - frames (时间步): {y_samples.shape[1]}")
        print(f"    - H (纬度):       {y_samples.shape[2]}")
        print(f"    - W (经度):       {y_samples.shape[3]}")
        print(f"    - channels (变量): {y_samples.shape[4]}")

        # ================= 最终信息汇总 =================
        print(f"\n" + "=" * 50)
        print(" 样本创建完成！")
        print("=" * 50)
        print(f"样本总数：{len(X_samples)}")
        print(f"输入形状：{X_samples.shape} (batch, frames, H, W, channels)")
        print(f"输出形状：{y_samples.shape} (batch, frames, H, W)")
        print(f"输入内存：{X_samples.nbytes / 1024 ** 3:.2f} GB")
        print(f"输出内存：{y_samples.nbytes / 1024 ** 3:.2f} GB")
        print(f"总内存：{(X_samples.nbytes + y_samples.nbytes) / 1024 ** 3:.2f} GB")
        print("=" * 50)

        return X_samples, y_samples

    # def normalize_data(self, X, y):
    #     """数据归一化"""
    #     print("\n" + "=" * 50)
    #     print("数据归一化...")
    #
    #     # 获取原始形状
    #     n_samples, n_frames, H, W, n_channels = X.shape
    #
    #
    #     # 重塑为2D数组用于归一化
    #     X_reshaped = X.reshape(-1, n_channels)
    #     y_reshaped = y.reshape(-1, 1)
    #
    #     # 创建并拟合标准化器
    #     X_scaler = StandardScaler()
    #     y_scaler = StandardScaler()
    #
    #     X_normalized = X_scaler.fit_transform(X_reshaped).reshape(X.shape)
    #     y_normalized = y_scaler.fit_transform(y_reshaped).reshape(y.shape)
    #
    #     # 保存标准化器
    #     self.scalers['input'] = X_scaler
    #     self.scalers['target'] = y_scaler
    #
    #     print(f"归一化完成！")
    #     print(f"输入 - 均值: {X_scaler.mean_[:5]}...")
    #     print(f"输入 - 标准差: {X_scaler.scale_[:5]}...")
    #
    #     return X_normalized, y_normalized

    def normalize_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        数据归一化 - 只使用训练集拟合，然后转换所有数据集
        """
        print("\n" + "=" * 50)
        print("数据归一化（只使用训练集统计）...")
        print("=" * 50)

        # 获取原始形状
        n_samples_train, n_frames, H, W, n_channels = X_train.shape

        # 重塑训练集用于拟合
        X_train_reshaped = X_train.reshape(-1, n_channels)
        y_train_reshaped = y_train.reshape(-1, 1)

        # 创建并拟合标准化器（只用训练集）
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        print("正在拟合标准化器（仅使用训练集）...")
        X_scaler.fit(X_train_reshaped)
        y_scaler.fit(y_train_reshaped)

        print(f"输入变量 - 均值范围: [{X_scaler.mean_.min():.3f}, {X_scaler.mean_.max():.3f}]") # type: ignore
        print(f"输入变量 - 标准差范围: [{X_scaler.scale_.min():.3f}, {X_scaler.scale_.max():.3f}]") # type: ignore
        print(f"目标变量 - 均值: {y_scaler.mean_[0]:.3f}") # type: ignore
        print(f"目标变量 - 标准差: {y_scaler.scale_[0]:.3f}") # type: ignore

        # 归一化函数
        def normalize_dataset(data, scaler, is_X=True):
            original_shape = data.shape
            """转换数据集"""
            if is_X:
                reshaped = data.reshape(-1, n_channels)
            else:
                reshaped = data.reshape(-1, 1)
            normalized = scaler.transform(reshaped)
            return normalized.reshape(original_shape)

        print("\n正在转换数据集...")
        # 转换所有数据集
        X_train_norm = normalize_dataset(X_train, X_scaler, True)
        X_val_norm = normalize_dataset(X_val, X_scaler, True)
        X_test_norm = normalize_dataset(X_test, X_scaler, True)

        y_train_norm = normalize_dataset(y_train, y_scaler, False)
        y_val_norm = normalize_dataset(y_val, y_scaler, False)
        y_test_norm = normalize_dataset(y_test, y_scaler, False)

        # 保存标准化器
        self.scalers['input'] = X_scaler
        self.scalers['target'] = y_scaler

        print("\n归一化完成！")
        print(f"训练集: X [{X_train_norm.min():.3f}, {X_train_norm.max():.3f}]")
        print(f"验证集: X [{X_val_norm.min():.3f}, {X_val_norm.max():.3f}]")
        print(f"测试集: X [{X_test_norm.min():.3f}, {X_test_norm.max():.3f}]")

        return X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm
    def split_data(self, X, y,data_obj):
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

        times = data_obj.time.values
        total_samples = len(X)
        stride = TIME_SAMPLING_STRIDE

        sample_start_indices = np.arange(total_samples) * stride

        sample_times = times[sample_start_indices]

        # 2. 计算切分点
        t_train_end = np.datetime64(TIME_END_TRAIN)
        t_val_end = np.datetime64(TIME_END_VAL)

        # searchsorted(side='right') 返回第一个大于目标值的位置，正好作为切分索引
        train_end_idx = np.searchsorted(sample_times, t_train_end, side='right')
        val_end_idx = np.searchsorted(sample_times, t_val_end, side='right')

        # 3. 打印诊断信息
        print(f"总有效样本: {total_samples}")

        def get_time_str(idx):
            if idx < 0 or idx >= total_samples:
                return "N/A"
            return str(times[sample_start_indices[idx]])

        print(f"训练集: [0, {train_end_idx}) -> 末样本起始: {get_time_str(train_end_idx - 1)}")
        print(f"验证集: [{train_end_idx}, {val_end_idx}) -> 末样本起始: {get_time_str(val_end_idx - 1)}")
        print(f"测试集: [{val_end_idx}, end) -> 首样本起始: {get_time_str(val_end_idx)}")

        # 4. 执行切片
        X_train, y_train = X[:train_end_idx], y[:train_end_idx]
        X_val, y_val = X[train_end_idx:val_end_idx], y[train_end_idx:val_end_idx]
        X_test, y_test = X[val_end_idx:], y[val_end_idx:]

        # 5. 统计
        print(f"\n划分结果:")
        print(f"  Train: {len(X_train)} ({len(X_train) / total_samples:.1%})")
        print(f"  Val:   {len(X_val)} ({len(X_val) / total_samples:.1%})")
        print(f"  Test:  {len(X_test)} ({len(X_test) / total_samples:.1%})")

        assert len(X_train) + len(X_val) + len(X_test) == total_samples, "样本总数不匹配!"

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

    # 添加诊断代码到 data_prepare.py
    def analyze_time_periods(self, data):
        """分析不同时间段的统计特性 (支持多目标变量)"""
        times = data.time.values

        # 获取三个时间段的索引
        n_samples = len(times)
        train_end = int(n_samples * 0.76)
        val_end = int(n_samples * 0.84)

        train_times = times[:train_end]
        val_times = times[train_end:val_end]
        test_times = times[val_end:]

        print("\n时间段统计:")
        print(f"训练集: {train_times[0]} 到 {train_times[-1]}")
        print(f"验证集: {val_times[0]} 到 {train_times[-1]}")
        print(f"测试集: {test_times[0]} 到 {test_times[-1]}")

        # ✅ 核心修改：动态获取目标变量列表
        # 优先使用全局变量 TARGET_VARIABLE，如果不存在则报错提示
        try:
            target_vars = TARGET_VARIABLES
        except NameError:
            raise NameError("未找到全局变量 'TARGET_VARIABLE'。请确保在文件顶部已定义它（例如：TARGET_VARIABLE = ['u', 'v']）。")

        # 兼容单变量情况：如果是字符串，转为列表
        if isinstance(target_vars, str):
            target_vars = [target_vars]
        
        print(f"\n正在分析目标变量: {target_vars}")

        # 1. 提取并堆叠所有目标变量
        y_arrays = []
        for var_name in target_vars:
            if var_name not in data.data_vars:
                raise ValueError(f"数据集中找不到变量: '{var_name}'。可用变量: {list(data.data_vars)}")
            y_arrays.append(data[var_name].values)
        
        # 堆叠后形状: (Time, Lat, Lon, Num_Variables)
        y = np.stack(y_arrays, axis=-1)

        # 2. 分变量打印统计信息
        print(f"\n目标变量详细统计:")
        
        for i, var_name in enumerate(target_vars):
            # 提取当前变量的切片 (axis=-1 是变量维度，i 是第几个变量)
            train_y = y[:train_end, ..., i].flatten()
            val_y = y[train_end:val_end, ..., i].flatten()
            test_y = y[val_end:, ..., i].flatten()

            print(f"\n--- 变量 [{i}]: {var_name} ---")
            print(f"训练集 - 均值: {train_y.mean():.4f}, 标准差: {train_y.std():.4f}, 范围: [{train_y.min():.2f}, {train_y.max():.2f}]")
            print(f"验证集 - 均值: {val_y.mean():.4f}, 标准差: {val_y.std():.4f}, 范围: [{val_y.min():.2f}, {val_y.max():.2f}]")
            print(f"测试集 - 均值: {test_y.mean():.4f}, 标准差: {test_y.std():.4f}, 范围: [{test_y.min():.2f}, {test_y.max():.2f}]")


    def run_pipeline(self):
        """运行完整的数据处理流程"""
        print("\n" + "=" * 50)
        print("启动完整数据处理流程")
        print("=" * 50)
        print(f"协变量模式: {'已启用' if ADD_COVARIATES else '已禁用'}")

        # 1. 下载数据
        data = self.download_data()

        # 2. 创建样本
        X, y = self.create_samples(data)

        # 2.5 （可选）添加协变量
        if ADD_COVARIATES:
            print("\n" + "=" * 50)
            print("🔧 协变量处理阶段")
            print("=" * 50)
            
            X = self._augment_with_covariates(X, data)

            expected_channels = len(VARIABLES) + len(COVARIATE_LIST)
            if X.shape[-1] == expected_channels:
                print("✅ 协变量添加完成！")
                print(f"   增强后输入形状：{X.shape}")
            else:
                print("⚠️ 协变量未完全加进去，当前使用的是未增强或部分增强输入。")
                print(f"   当前输入形状：{X.shape}")
                print(f"   期望通道数：{expected_channels}，当前通道数：{X.shape[-1]}")

        # 3. 划分数据集
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y, data)

        # 4. 归一化（只使用训练集统计）
        X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm = self.normalize_data(X_train, X_val, X_test, y_train, y_val, y_test)

        # 5. 保存数据
        self.save_processed_data(X_train_norm, X_val_norm,
                                 y_train_norm, y_val_norm, X_test_norm, y_test_norm)

        print("\n" + "=" * 50)
        print("数据处理流程完成！")
        print("=" * 50)

        return X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm
    
        
    def _augment_with_covariates(self, X_samples, data):
        """
        通用协变量增强函数
        自动读取 COVARIATE_RAW_VARS 列表，计算特征，并对齐到 X_samples 的时空维度
        """
        if not COVARIATE_RAW_VARS:
            print("⚠️ 协变量列表为空，跳过增强。")
            return X_samples

        try:
            print("\n" + "=" * 60)
            print("🔧 开始通用协变量增强处理...")
            print("=" * 60)
            
            N, T, H, W, C_core = X_samples.shape
            print(f"输入基础形状：{X_samples.shape}")
            print(f"待处理协变量原料：{COVARIATE_RAW_VARS}")

            covariate_features = []

            # --- 辅助函数：维度对齐 ---
            def align_spatial_dims(arr, target_h, target_w):
                """确保数组最后两维是 (H, W)"""
                if arr.shape[-2] == target_h and arr.shape[-1] == target_w:
                    return arr
                elif arr.shape[-2] == target_w and arr.shape[-1] == target_h:
                    return np.transpose(arr, list(range(arr.ndim-2)) + [-1, -2])
                else:
                    raise ValueError(f"空间维度无法对齐：{arr.shape[-2:]} vs ({target_h}, {target_w})")

            # --- 辅助函数：创建滑动窗口 ---
            def make_windows(arr_1d_time_series, input_frames):
                """将 (Time, H, W) 转换为 (N, T, H, W)"""
                # arr_1d_time_series 形状应为 (Total_Time, H, W)
                total_t = arr_1d_time_series.shape[0]
                needed_total = N + input_frames + PRED_FRAMES - 1 # 近似估算，实际用 X_samples 的长度反推
                
                # 使用 sliding_window_view
                windows = sliding_window_view(arr_1d_time_series, window_shape=input_frames, axis=0)
                # 截取与 X_samples 对应的部分 (去掉尾部不足以构成预测的部分，且偏移 input_frames 以匹配 y? 
                # 注意：X_samples 已经截取了 [:total_samples]。
                # 这里的 arr_1d_time_series 是完整时间的。我们需要取前 total_samples 个窗口。
                
                total_samples_needed = X_samples.shape[0]
                if windows.shape[0] < total_samples_needed:
                     # 如果因为 stride 导致长度不匹配，这里需要小心。
                     # 假设 create_samples 中已经应用了 stride，那么 data 也是原始时间。
                     # 我们需要对 data 也应用同样的 stride 逻辑，或者在切片时对应。
                     # 简单做法：先做窗口，再按 stride 切片（如果 create_samples 还没切片，但这里 X_samples 已经切了）
                     # 由于 X_samples 已经经过了 TIME_SAMPLING_STRIDE 切片，我们需要对协变量做同样的操作。
                     # 但 sliding_window_view 是在原始时间上做的。
                     # 策略：先做全量窗口，然后 [::STRIDE]
                     pass
                
                final_windows = windows[:total_samples_needed]
                
                # 检查维度顺序 (N, T, H, W)
                if final_windows.ndim == 4 and final_windows.shape[1] == input_frames:
                    return final_windows
                elif final_windows.ndim == 4 and final_windows.shape[-1] == input_frames:
                    return np.moveaxis(final_windows, -1, 1)
                else:
                    # 尝试修复常见维度错误
                    if final_windows.shape[0] != total_samples_needed:
                         # 应用 Stride
                         final_windows = final_windows[::TIME_SAMPLING_STRIDE]
                    
                    if final_windows.shape[1] != input_frames:
                        final_windows = np.moveaxis(final_windows, -1, 1)
                    return final_windows

            # ================= 主循环：处理每个协变量 =================
            for i, var_name in enumerate(COVARIATE_RAW_VARS):
                print(f"\n[{i+1}/{len(COVARIATE_RAW_VARS)}] 处理协变量：{var_name}")
                
                if var_name not in data.data_vars:
                    print(f"  ⚠️ 警告：数据集中不存在 {var_name}，跳过。")
                    continue

                # 1. 读取数据
                var_arr = data[var_name]
                
                # 2. 维度对齐 (Time, Lat, Lon)
                target_dims = ('time', 'latitude', 'longitude')
                if var_arr.dims != target_dims:
                    var_arr = var_arr.transpose(*target_dims)
                
                # 3. 转为 numpy (分块优化)
                var_np = var_arr.chunk({'time': -1}).compute() # (Time, H, W)
                var_np = align_spatial_dims(var_np, H, W)
                
                # 4. 判断变量类型并处理
                # 规则：植被指数 (LAI) 通常视为静态/慢变 -> 归一化
                #       气象通量/状态 (MSLP, Rad, Flux, BLH) -> 动态 -> 异常值 或 归一化
                is_static = "leaf_area_index" in var_name
                
                if is_static:
                    print(f"  -> 识别为静态/慢变变量，执行全局归一化...")
                    # 全局归一化 (0-1 或 Z-score)
                    mean_val = var_np.mean()
                    std_val = var_np.std()
                    if std_val < 1e-6: std_val = 1.0
                    var_norm = (var_np - mean_val) / std_val
                    
                    # 对于静态变量，每个时间步都是一样的。
                    # 我们需要将其扩展为 (N, T, H, W)
                    # 先取第一个时间步代表全场 (或者平均场)
                    static_field = var_norm[0] # (H, W)
                    # 广播到 (N, T, H, W)
                    # 技巧：reshape 为 (1, 1, H, W) 然后 tile
                    static_expanded = np.tile(static_field[np.newaxis, np.newaxis, :, :], (N, T, 1, 1))
                    cov_feature = static_expanded
                    print(f"     形状：{cov_feature.shape}, 范围：[{cov_feature.min():.2f}, {cov_feature.max():.2f}]")

                else:
                    print(f"  -> 识别为动态变量，执行气候态异常值计算 (或归一化)...")
                    # 动态变量处理逻辑
                    # A. 计算气候基准 (Climatology)
                    # 为了简化，这里使用 dayofyear 平均。如果数据时间短，直接用全局均值减也可以。
                    # 这里采用：Anomaly = Value - Climatology(dayofyear)
                    
                    time_coord = data.time
                    dayofyear = time_coord.dt.dayofyear.values
                    
                    # 计算气候态 (只在用于计算 clim 的时间范围内)
                    # 注意：data 对象包含所有时间。我们需要确保 clim 计算不使用未来数据（防止泄露），
                    # 但在预处理阶段，通常允许使用整个历史期计算气候态。
                    clim_group = var_arr.groupby('time.dayofyear').mean('time')
                    clim_values = clim_group.values # (366, H, W) 或 (365, ...)
                    clim_values = align_spatial_dims(clim_values, H, W)
                    
                    # 匹配每一天的气候值
                    # dayofyear 从 1 开始，数组索引从 0 开始
                    indices = dayofyear - 1
                    # 处理闰年情况 (如果 clim 只有 365 天但数据有 366)
                    if clim_values.shape[0] == 365:
                        indices = np.minimum(indices, 364)
                    
                    clim_matched = clim_values[indices] # (Time, H, W)
                    
                    # 计算异常值
                    anomaly = var_np - clim_matched
                    
                    # 可选：平滑 (Moving Average) - 类似原代码
                    # 为了速度，这里暂时省略复杂的逐帧平滑，直接使用标准化
                    # 全局标准化异常值
                    anom_mean = anomaly.mean()
                    anom_std = anomaly.std()
                    if anom_std < 1e-6: anom_std = 1.0
                    var_final = (anomaly - anom_mean) / anom_std
                    
                    # B. 创建滑动窗口以匹配 X_samples
                    # var_final shape: (Total_Time, H, W)
                    cov_feature = make_windows(var_final, T)
                    
                    # 再次检查维度是否匹配 X_samples (N, T, H, W)
                    if cov_feature.shape != (N, T, H, W):
                        # 尝试裁剪或填充以匹配 N
                        if cov_feature.shape[0] > N:
                            cov_feature = cov_feature[:N]
                        elif cov_feature.shape[0] < N:
                            print(f"  ⚠️ 警告：协变量窗口数量 ({cov_feature.shape[0]}) 少于样本数 ({N})，尝试填充...")
                            # 简单重复最后一帧
                            last_frame = cov_feature[-1:]
                            repeat_times = N - cov_feature.shape[0]
                            padding = np.repeat(last_frame, repeat_times, axis=0)
                            cov_feature = np.concatenate([cov_feature, padding], axis=0)
                    
                    print(f"     形状：{cov_feature.shape}, 范围：[{cov_feature.min():.2f}, {cov_feature.max():.2f}]")

                covariate_features.append(cov_feature)

            # ================= 拼接所有协变量 =================
            if not covariate_features:
                print("⚠️ 没有成功处理任何协变量，返回原始数据。")
                return X_samples

            print(f"\n正在拼接 {len(covariate_features)} 个协变量通道...")
            # 堆叠在通道轴 (axis=-1 之前，需要先确保它们是 (N, T, H, W))
            # 当前 cov_feature 是 (N, T, H, W)。我们需要加一个通道维变成 (N, T, H, W, 1)
            cov_stack = np.stack(covariate_features, axis=4)
            
            print(f"  协变量堆叠形状：{cov_stack.shape} (N, T, H, W, Cov_Channels)")

            # 最终拼接：X_samples (N, T, H, W, C_core) + Cov (N, T, H, W, C_cov)
            X_augmented = np.concatenate([X_samples, cov_stack], axis=-1)

            print(f"\n✅ 协变量增强完成！")
            print(f"   原始通道：{C_core}")
            print(f"   新增通道：{cov_stack.shape[-1]}")
            print(f"   最终输入形状：{X_augmented.shape}")
            
            # 更新全局变量以便模型知道新的通道数 (可选，取决于您的模型初始化逻辑)
            global INPUT_CHANNELS
            INPUT_CHANNELS = X_augmented.shape[-1]
            
            return X_augmented

        except Exception as e:
            print(f"\n❌ 协变量增强失败：{e}")
            import traceback
            traceback.print_exc()
            print("-> 回退到无协变量模式")
            return X_samples

    # def _augment_with_covariates(self, X_samples, data):
    #     """
    #     为输入数据添加协变量（修复维度对齐问题）
    #     """
    #     try:
    #         print("\n" + "=" * 50)
    #         print("计算协变量...")
    #         print("=" * 50)
            
    #         N, T, H, W, C = X_samples.shape
            
    #         # --- 辅助函数：确保维度顺序为 (..., H, W) ---
    #         def align_spatial_dims(arr, target_h, target_w):
    #             """
    #             检查数组最后两维，如果是 (W, H) 则转置为 (H, W)。
    #             如果已经是 (H, W) 则保持不变。
    #             """
    #             current_h, current_w = arr.shape[-2], arr.shape[-1]
    #             if current_h == target_h and current_w == target_w:
    #                 return arr
    #             elif current_h == target_w and current_w == target_h:
    #                 # 转置最后两个维度: (... , W, H) -> (... , H, W)
    #                 # axes: [0, 1, ..., -2, -1] -> [0, 1, ..., -1, -2]
    #                 new_axes = list(range(arr.ndim))
    #                 new_axes[-2], new_axes[-1] = new_axes[-1], new_axes[-2]
    #                 return np.transpose(arr, new_axes)
    #             else:
    #                 raise ValueError(f"维度无法对齐：当前 {arr.shape[-2:]} vs 目标 ({target_h}, {target_w})")
        
    #         # ======== 1. 计算气候基准 ========
    #         print("\n1️⃣  计算气候基准平均值...")
    #         # 注意：这里选择的时间范围要覆盖用于计算 climatology 的历史数据
    #         clim_data = data.sel(time=slice(None, TIME_END_VAL))
    #         if len(clim_data.time) == 0:
    #             clim_data = data
    #             print("   [fallback] 未找到测试期前历史数据，改用当前可用时间范围")
            
    #         # 按日期 (dayofyear) 分组计算气候平均
    #         mslp_clim = clim_data['mean_sea_level_pressure'].groupby('time.dayofyear').mean('time')
    #         temp_clim = clim_data['2m_temperature'].groupby('time.dayofyear').mean('time')
    #         dewpt_clim = clim_data['2m_dewpoint_temperature'].groupby('time.dayofyear').mean('time')
    #         humid_clim = temp_clim - dewpt_clim
            
    #         print(f"   ✅ 气候基准计算完成")
            
    #         # ======== 2. 提取 MSLP 异常 ========
    #         print("\n2️⃣  提取MSLP异常...")
    #         mslp_raw = data['mean_sea_level_pressure'].values
    #         # 【关键修复】对齐维度
    #         mslp_data = align_spatial_dims(mslp_raw, H, W)
            
    #         dayofyear = data.time.dt.dayofyear.values
            
    #         # 同样需要对齐气候基准的维度
    #         mslp_clim_values = align_spatial_dims(mslp_clim.values, H, W)
    #         mslp_clim_matched = mslp_clim_values[dayofyear - 1]
            
    #         # 计算异常值并平滑
    #         mslp_anom = mslp_data - mslp_clim_matched
    #         mslp_anom_smooth = np.zeros_like(mslp_anom)
    #         for i in range(len(mslp_anom)):
    #             start = max(0, i - 3)
    #             end = min(len(mslp_anom), i + 4)
    #             mslp_anom_smooth[i] = mslp_anom[start:end].mean(axis=0)
            
    #         mslp_anom_norm = mslp_anom_smooth / (mslp_anom_smooth.std() + 1e-6)
    #         print(f"   MSLP异常范围: [{mslp_anom_norm.min():.2f}, {mslp_anom_norm.max():.2f}]")
            
    #         # ======== 3. 提取相对湿度异常 ========
    #         print("\n3️⃣  提取相对湿度异常...")
    #         temp_raw = data['2m_temperature'].values
    #         dewpt_raw = data['2m_dewpoint_temperature'].values
            
    #         # 【关键修复】对齐维度
    #         temp_data = align_spatial_dims(temp_raw, H, W)
    #         dewpt_data = align_spatial_dims(dewpt_raw, H, W)
            
    #         humid_idx = temp_data - dewpt_data
            
    #         humid_clim_values = align_spatial_dims(humid_clim.values, H, W)
    #         humid_clim_matched = humid_clim_values[dayofyear - 1]
            
    #         humid_anom = humid_idx - humid_clim_matched
    #         humid_anom_norm = humid_anom / (humid_anom.std() + 1e-6)
    #         print(f"   湿度异常范围: [{humid_anom_norm.min():.2f}, {humid_anom_norm.max():.2f}]")
            
    #         # ======== 4. 拼接为最终协变量 (N, H, W, 2) ========
    #         # 此时 mslp_anom_norm 和 humid_anom_norm 的形状都是 (N, H, W)
    #         cov_stack = np.stack([mslp_anom_norm, humid_anom_norm], axis=-1)
    #         print(f"   协变量堆叠形状: {cov_stack.shape} (N, H, W, 2)")
            
    #         # ======== 5. 按输入窗口对齐协变量 ========
    #         print("\n4️⃣  扩展输入维度...")
    #         print(f"   原始输入: {X_samples.shape} (N, T, H, W, {C})")
            
    #         total_samples = cov_stack.shape[0] - INPUT_FRAMES - PRED_FRAMES + 1
    #         if total_samples <= 0:
    #             raise ValueError(
    #                 f"协变量时间步不足: time={cov_stack.shape[0]}, need={INPUT_FRAMES + PRED_FRAMES}"
    #             )

    #         # 创建滑动窗口
    #         cov_windows = sliding_window_view(cov_stack, window_shape=INPUT_FRAMES, axis=0)
    #         # 截取有效部分
    #         cov_samples = cov_windows[:total_samples]
            
    #         # 检查维度顺序，sliding_window_view 可能会把窗口维放在最后或最前，取决于版本和实现
    #         # 标准行为: (New_Time, Window, H, W, Channels)
    #         if cov_samples.shape[1] != INPUT_FRAMES:
    #              # sliding_window_view 常把窗口维放在最后，这里移回到第 2 维
    #              cov_samples = np.moveaxis(cov_samples, -1, 1)
    #              print(f"   已调整协变量窗口维顺序: {cov_samples.shape}")
    #         target_h, target_w = X_samples.shape[2], X_samples.shape[3]
    #         current_h, current_w = cov_samples.shape[2], cov_samples.shape[3]

    #         if current_h == target_w and current_w == target_h:
    #             print(f"   ⚠️ 检测到协变量空间维度反转 ({current_h}, {current_w}) vs 目标 ({target_h}, {target_w})")
    #             print(f"   🔧 正在执行转置修正：axes (0,1,3,2,4)...")
    #             # 转置：保持 N(0), T(1) 不变，交换 H(2)<->W(3)，保持 C(4) 不变
    #             cov_samples = np.transpose(cov_samples, (0, 1, 3, 2, 4))
    #             print(f"   ✅ 修正后协变量形状：{cov_samples.shape}")
    #         elif current_h != target_h or current_w != target_w:
    #             # 如果既不是匹配的，也不是反转的，说明有严重的数据形状错误
    #             raise ValueError(
    #                 f"协变量空间维度严重不匹配！\n"
    #                 f"目标 (H, W): ({target_h}, {target_w})\n"
    #                 f"当前 (H, W): ({current_h}, {current_w})\n"
    #                 f"无法通过简单转置修复。"
    #             )
    #         else:
    #             print(f"   ✅ 维度已对齐，无需修正。")
        

    #         print(f"   协变量窗口: {cov_samples.shape} (N, T, H, W, 2)")

    #         # ======== 6. 最终拼接 ========
    #         X_augmented = np.concatenate([X_samples, cov_samples], axis=-1)

            
    #         print(f"   增强后输入: {X_augmented.shape} (N, T, H, W, {C+2})")
    #         print("\n   ✅ 协变量添加成功!")
            
    #         return X_augmented
            
    #     except Exception as e:
    #         print(f"\n⚠️  协变量提取失败: {e}")
    #         print(f"   继续使用无协变量的输入")
    #         import traceback
    #         traceback.print_exc()
    #         return X_samples

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
