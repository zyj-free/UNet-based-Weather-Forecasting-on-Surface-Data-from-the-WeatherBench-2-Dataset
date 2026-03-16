"""
数据准备模块 - 下载、处理、归一化
"""
import os
import sys
import pickle
import gc
import numpy as np
from sklearn.preprocessing import StandardScaler
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
        print("=" * 50)
    #     cache_dir = Path(LOCAL_CACHE_PATH)
    #     cache_dir.mkdir(parents=True, exist_ok=True)
    #     cache_filename = f"era5_{TIME_START}_to_{TIME_END}_{int(LAT_MIN)}-{int(LAT_MAX)}_{int(LON_MIN)}-{int(LON_MAX)}.nc"
    #     cache_file = cache_dir / cache_filename
    #     # ==================== 检查缓存 ====================
    #     if cache_file.exists():
    #         print(f"\n发现本地缓存，直接加载...")
    #         print(f"   缓存文件：{cache_file}")
    #
    #         start_time = time.time()
    #         data = xr.open_dataset(cache_file)
    #         load_time = time.time() - start_time
    #
    #         print(f"   加载耗时：{load_time:.2f} 秒")
    #         print(f"\n数据加载完成！")
    #         print(f"数据维度：{data.dims}")
    #         print(f"时间点数：{len(data.time)}")
    #         print(f"空间网格：{len(data.latitude)} x {len(data.longitude)}")
    #         print(f"变量数：{len(data.data_vars)}")
    #
    #         return data
    #
    #     # ==================== 首次下载 ====================
    #     print(f"\n缓存不存在，从 GCS 下载数据...")
    #     print(f"   首次运行较慢，后续将使用缓存（快 10-20 倍）")
    #     print("-" * 50)
    #
    #     start_time = time.time()
    #
    #     # 打开 Zarr 格式数据（不立即加载）
    #     print("\n[1/5] 打开远程数据集...")
    #     ds = xr.open_zarr(
    #         DATA_PATH,
    #         consolidated=True,
    #         storage_options={'timeout': 300}
    #     )
    #
    #     # 选择时间范围
    #     print(f"[2/5] 选择时间范围...")
    #     time_slice = ds.time.sel(time=slice(TIME_START, TIME_END))
    #     print(f"   需要的时间范围：{len(time_slice)} 天")
    #
    #     # 选择空间范围和变量
    #     print(f"[3/5] 选择空间范围和变量...")
    #     print(f"   纬度：{LAT_MIN} 到 {LAT_MAX}")
    #     print(f"   经度：{LON_MIN} 到 {LON_MAX}")
    #     print(f"   变量：{VARIABLES}")
    #
    #     # 注意：latitude 要从大到小切片（因为纬度是递减的）
    #     data = ds.sel(
    #         time=time_slice,
    #         latitude=slice(LAT_MIN, LAT_MAX),  # 从大到小！
    #         longitude=slice(LON_MIN, LON_MAX)
    #     )[VARIABLES]
    #
    #     # 加载到内存（最慢的步骤）
    #     print(f"\n[4/5] 加载数据到内存...")
    #     print(f"   这可能需要 10-30 分钟，请耐心等待...")
    #     data = data.load()
    #
    #     # 保存到本地缓存
    #     print(f"\n[5/5] 保存到本地缓存...")
    #     # 改为：
    #     import os
    #
    #     # 如果文件存在，先删除
    #     if cache_file.exists():
    #         try:
    #             cache_file.unlink()
    #             print(f"删除旧缓存文件：{cache_file.name}")
    #         except PermissionError:
    #             print(f"无法删除旧文件，尝试覆盖...")
    #
    #     # 保存时添加 mode='w' 明确写入模式
    #     data.to_netcdf(cache_file, format='NETCDF4', engine='netcdf4', mode='w')
    #
    #     elapsed = time.time() - start_time
    #     cache_size = sum(f.stat().st_size for f in cache_file.rglob('*')) / 1024 / 1024
    #
    #     print(f"\n数据下载完成！")
    #     print(f"   总耗时：{elapsed / 60:.1f} 分钟")
    #     print(f"   缓存大小：{cache_size:.1f} MB")
    #     print(f"   缓存位置：{cache_file}")
    #     print(f"\n数据维度：{data.dims}")
    #     print(f"时间点数：{len(data.time)}")
    #     print(f"空间网格：{len(data.latitude)} x {len(data.longitude)}")
    #     print(f"变量数：{len(data.data_vars)}")
    #     print("-" * 50)
    #     print(f"下次运行将直接加载缓存，只需几秒！")
    #     print("=" * 50)

        # return data

        # 打开Zarr格式数据
        ds = xr.open_zarr(DATA_PATH, consolidated=True)
        time_slice = ds.time.sel(time=slice(TIME_START, TIME_END))
        print(f"需要的时间范围：{len(time_slice)} 天")
        ds_subset = ds.sel(
            time=time_slice,
            latitude=slice(LAT_MAX, LAT_MIN),  # 注意：纬度从大到小
            longitude=slice(LON_MIN, LON_MAX)
        )[VARIABLES]
        # 选择数据
        data = ds.sel(
            time=slice(TIME_START, TIME_END),
            latitude=slice(LAT_MIN, LAT_MAX),
            # latitude=slice(LAT_MAX, LAT_MIN),
            longitude=slice(LON_MIN, LON_MAX),
        )[VARIABLES]

        print(f"\n数据下载完成！")
        print(f"数据维度: {data.dims}")
        print(f"时间点数: {len(data.time)}")
        print(f"空间网格: {len(data.latitude)} x {len(data.longitude)}")
        print(f"变量数: {len(data.data_vars)}")

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
            if var_dims == ('time', 'longitude', 'latitude'):
                print(f"    ⚠️  维度转置：(time, lon, lat) -> (time, lat, lon)")
                var_arr = var_arr.transpose('time', 'latitude', 'longitude')
            elif var_dims == ('time', 'latitude', 'longitude'):
                print(f"    ✅  维度正确：(time, lat, lon)")
            else:
                # 通用转置
                target_dims = ('time', 'latitude', 'longitude')
                order = [var_dims.index(d) for d in target_dims]
                print(f"    ⚠️  维度转置：{var_dims} -> {target_dims}")
                var_arr = var_arr.transpose(*order)

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

        # 处理目标变量 y
        print(f"\n处理目标变量 {TARGET_VARIABLE}...")
        y_arr = data[TARGET_VARIABLE]
        y_dims = y_arr.dims

        # 统一维度顺序(Batch, 1, H, W)
        if y_dims == ('time', 'longitude', 'latitude'):
            print(f"  ⚠️  维度转置：(time, lon, lat) -> (time, lat, lon)")
            y_arr = y_arr.transpose('time', 'latitude', 'longitude')
        elif y_dims == ('time', 'latitude', 'longitude'):
            print(f"  ✅  维度正确：(time, lat, lon)")
        else:
            target_dims = ('time', 'latitude', 'longitude')
            order = [y_dims.index(d) for d in target_dims]
            y_arr = y_arr.transpose(*order)

        y_chunked = y_arr.chunk({'time': -1})
        y = y_chunked.compute()
        print(f"  y 形状：{y.shape} (time, lat, lon)")

        print(f"\n原始数据形状：X = {X.shape}, y = {y.shape}")

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
        y_all_windows = sliding_window_view(y, window_shape=PRED_FRAMES, axis=0)
        # y 需要偏移 INPUT_FRAMES，使输入输出对齐
        y_samples = y_all_windows[INPUT_FRAMES: INPUT_FRAMES + total_samples]
        if y_samples.shape[1] != PRED_FRAMES:
            y_samples = np.moveaxis(y_samples, -1, 1)
        print(f"    y 窗口形状：{y_samples.shape} (N, T, H, W)")

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

        # y 保持 (N, T, H, W) - 不做转置
        print(f"  y 最终形状：{y_samples.shape}")
        print(f"    - batch (样本数): {y_samples.shape[0]}")
        print(f"    - frames (时间步): {y_samples.shape[1]}")
        print(f"    - H (纬度):       {y_samples.shape[2]}")
        print(f"    - W (经度):       {y_samples.shape[3]}")

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
    def get_fold_split_indices(self, data_obj, fold_index):
        """
        辅助函数：计算当前 Fold 的切分索引，不实际切片数据以节省内存
        """
        times = data_obj.time.values
        stride = TIME_SAMPLING_STRIDE
        # 估算样本总数 (用于创建 sample_times)
        # 注意：这里需要与 create_samples 中的逻辑完全一致
        total_time_steps = len(times)
        total_samples_raw = total_time_steps - INPUT_FRAMES - PRED_FRAMES + 1
        if TIME_SAMPLING_STRIDE > 1:
            total_samples = (total_samples_raw + TIME_SAMPLING_STRIDE - 1) // TIME_SAMPLING_STRIDE
        else:
            total_samples = total_samples_raw

        sample_start_indices = np.arange(total_samples) * stride
        sample_start_indices = np.clip(sample_start_indices, 0, len(times) - 1)
        sample_times = times[sample_start_indices]

        val_year = CV_VAL_YEARS[fold_index]

        t_train_end = np.datetime64(f"{val_year - 1}-12-31")
        t_val_end = np.datetime64(f"{val_year}-12-31")
        t_test_start = np.datetime64(f"{TEST_YEAR}-01-01")
        t_test_end = np.datetime64(f"{TEST_YEAR}-12-31")

        train_end_idx = np.searchsorted(sample_times, t_train_end, side='right')
        val_end_idx = np.searchsorted(sample_times, t_val_end, side='right')
        test_start_idx = np.searchsorted(sample_times, t_test_start, side='left')
        test_end_idx = np.searchsorted(sample_times, t_test_end, side='right')

        test_start_idx = max(test_start_idx, val_end_idx)

        return train_end_idx, val_end_idx, test_start_idx, test_end_idx

    # def normalize_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
    #     """
    #     数据归一化 - 只使用训练集拟合，然后转换所有数据集
    #     """
    #     print("\n" + "=" * 50)
    #     print("数据归一化（只使用训练集统计）...")
    #     print("=" * 50)
    #
    #     # 获取原始形状
    #     n_samples_train, n_frames, H, W, n_channels = X_train.shape
    #
    #     # 重塑训练集用于拟合
    #     X_train_reshaped = X_train.reshape(-1, n_channels)
    #     y_train_reshaped = y_train.reshape(-1, 1)
    #
    #     # 创建并拟合标准化器（只用训练集）
    #     X_scaler = StandardScaler()
    #     y_scaler = StandardScaler()
    #
    #     print("正在拟合标准化器（仅使用训练集）...")
    #     X_scaler.fit(X_train_reshaped)
    #     y_scaler.fit(y_train_reshaped)
    #
    #     print(f"输入变量 - 均值范围: [{X_scaler.mean_.min():.3f}, {X_scaler.mean_.max():.3f}]")
    #     print(f"输入变量 - 标准差范围: [{X_scaler.scale_.min():.3f}, {X_scaler.scale_.max():.3f}]")
    #     print(f"目标变量 - 均值: {y_scaler.mean_[0]:.3f}")
    #     print(f"目标变量 - 标准差: {y_scaler.scale_[0]:.3f}")
    #
    #     # 归一化函数
    #     def normalize_dataset(data, scaler, is_X=True):
    #         original_shape = data.shape
    #         """转换数据集"""
    #         if is_X:
    #             reshaped = data.reshape(-1, n_channels)
    #         else:
    #             reshaped = data.reshape(-1, 1)
    #         normalized = scaler.transform(reshaped)
    #         return normalized.reshape(original_shape)
    #
    #     print("\n正在转换数据集...")
    #     # 转换所有数据集
    #     X_train_norm = normalize_dataset(X_train, X_scaler, True)
    #     X_val_norm = normalize_dataset(X_val, X_scaler, True)
    #     X_test_norm = normalize_dataset(X_test, X_scaler, True)
    #
    #     y_train_norm = normalize_dataset(y_train, y_scaler, False)
    #     y_val_norm = normalize_dataset(y_val, y_scaler, False)
    #     y_test_norm = normalize_dataset(y_test, y_scaler, False)
    #
    #     # 保存标准化器
    #     self.scalers['input'] = X_scaler
    #     self.scalers['target'] = y_scaler
    #
    #     print("\n归一化完成！")
    #     print(f"训练集: X [{X_train_norm.min():.3f}, {X_train_norm.max():.3f}]")
    #     print(f"验证集: X [{X_val_norm.min():.3f}, {X_val_norm.max():.3f}]")
    #     print(f"测试集: X [{X_test_norm.min():.3f}, {X_test_norm.max():.3f}]")
    #
    #     return X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm

    def normalize_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        数据归一化 - 核心修复点
        只在 X_train/y_train 上 fit，然后 transform 所有集合
        """
        print("\n" + "=" * 50)
        print("数据归一化")
        print("=" * 50)
        n_channels = X_train.shape[-1]

        # Reshape for scaler
        X_train_2d = X_train.reshape(-1, n_channels)
        y_train_2d = y_train.reshape(-1, 1)

        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        # FIT ONLY ON TRAIN
        X_scaler.fit(X_train_2d)
        y_scaler.fit(y_train_2d)

        print(f"输入变量 - 均值范围: [{X_scaler.mean_.min():.3f}, {X_scaler.mean_.max():.3f}]")
        print(f"输入变量 - 标准差范围: [{X_scaler.scale_.min():.3f}, {X_scaler.scale_.max():.3f}]")
        print(f"目标变量 - 均值: {y_scaler.mean_[0]:.3f}")
        print(f"目标变量 - 标准差: {y_scaler.scale_[0]:.3f}")

        # Transform all
        def transform_set(data, scaler, is_X=True):
            orig_shape = data.shape
            flat = data.reshape(-1, n_channels if is_X else 1)
            norm = scaler.transform(flat)
            return norm.reshape(orig_shape)

        X_train_n = transform_set(X_train, X_scaler, True)
        X_val_n = transform_set(X_val, X_scaler, True)
        X_test_n = transform_set(X_test, X_scaler, True)

        y_train_n = transform_set(y_train, y_scaler, False)
        y_val_n = transform_set(y_val, y_scaler, False)
        y_test_n = transform_set(y_test, y_scaler, False)

        # 保存 scalers 供后续推理使用 (这里只保存最后一个 fold 的，或者你可以选择保存所有)
        self.scalers['input'] = X_scaler
        self.scalers['target'] = y_scaler
        print("\n归一化完成！")
        print(f"训练集: X [{X_train_n.min():.3f}, {X_train_n.max():.3f}]")
        print(f"验证集: X [{X_val_n.min():.3f}, {X_val_n.max():.3f}]")
        print(f"测试集: X [{X_test_n.min():.3f}, {X_test_n.max():.3f}]")

        return X_train_n, X_val_n, X_test_n, y_train_n, y_val_n, y_test_n

    def split_data(self, X, y,data_obj):
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
        print(f"Val Y - Mean: {y_val.mean():.4f}, Std: {y_val.std():.4f}")
        print(f"Test Y - Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")

        ratio = y_val.std() / y_train.std()
        if ratio < 0.7:
            print(f"⚠️ 警告：验证集波动性仅为训练集的 {ratio:.2%}，验证集过于简单！")
        return X_train, X_val, X_test, y_train, y_val, y_test

    # def save_processed_data(self, X_train, X_val, y_train, y_val,X_test, y_test):
    #     """保存处理后的数据"""
    #     print("\n" + "=" * 50)
    #     print("保存处理后的数据...")
    #
    #     # 保存numpy数组
    #     np.save(os.path.join(PROCESSED_DATA_PATH, 'X_train.npy'), X_train)
    #     np.save(os.path.join(PROCESSED_DATA_PATH, 'X_val.npy'), X_val)
    #     np.save(os.path.join(PROCESSED_DATA_PATH, 'X_test.npy'), X_test)
    #
    #     np.save(os.path.join(PROCESSED_DATA_PATH, 'y_train.npy'), y_train)
    #     np.save(os.path.join(PROCESSED_DATA_PATH, 'y_val.npy'), y_val)
    #     np.save(os.path.join(PROCESSED_DATA_PATH, 'y_test.npy'), y_test)
    #
    #
    #     # 保存标准化器
    #     with open(os.path.join(PROCESSED_DATA_PATH, 'scalers.pkl'), 'wb') as f:
    #         pickle.dump(self.scalers, f)
    #
    #     print(f"数据已保存到: {PROCESSED_DATA_PATH}")

    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, fold_index=None):
        """保存处理后的数据，支持添加后缀以区分 Fold"""
        print("\n" + "=" * 50)
        print("保存处理后的数据...")
        if fold_index is not None and fold_index != "":
            # 强制转为整数，防止传入字符串 "1" 导致格式错误
            try:
                idx = int(fold_index)
                suffix = f"_fold_{idx}"
            except ValueError:
                # 如果传入的是非数字字符串，保留旧逻辑作为兼容，但打印警告
                suffix = f"_{fold_index}"
                print(f"⚠️ 警告：检测到非整数 fold_index '{fold_index}'，建议统一使用整数。")
        else:
            suffix = ""

        print(f"   -> 生成后缀: '{suffix}'")
        print(f"   -> 目标文件示例: X_train{suffix}.npy")

        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

        np.save(os.path.join(PROCESSED_DATA_PATH, f'X_train{suffix}.npy'), X_train)
        np.save(os.path.join(PROCESSED_DATA_PATH, f'X_val{suffix}.npy'), X_val)
        np.save(os.path.join(PROCESSED_DATA_PATH, f'X_test{suffix}.npy'), X_test)
        np.save(os.path.join(PROCESSED_DATA_PATH, f'y_train{suffix}.npy'), y_train)
        np.save(os.path.join(PROCESSED_DATA_PATH, f'y_val{suffix}.npy'), y_val)
        np.save(os.path.join(PROCESSED_DATA_PATH, f'y_test{suffix}.npy'), y_test)

        with open(os.path.join(PROCESSED_DATA_PATH, f'scalers{suffix}.pkl'), 'wb') as f:
            pickle.dump(self.scalers, f)

        print(f"✅ 数据已保存 (Fold {suffix if suffix else 'Default'})")
    # def load_processed_data(self):
    #     """加载处理后的数据"""
    #     print("\n" + "=" * 50)
    #     print("加载处理后的数据...")
    #
    #     X_train = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_train.npy'))
    #     X_val = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_val.npy'))
    #     X_test = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_test.npy'))
    #
    #     y_train = np.load(os.path.join(PROCESSED_DATA_PATH, 'y_train.npy'))
    #     y_val = np.load(os.path.join(PROCESSED_DATA_PATH, 'y_val.npy'))
    #     y_test = np.load(os.path.join(PROCESSED_DATA_PATH, 'y_test.npy'))
    #
    #
    #     with open(os.path.join(PROCESSED_DATA_PATH, 'scalers.pkl'), 'rb') as f:
    #         self.scalers = pickle.load(f)
    #
    #     print(f"数据加载完成！")
    #     print(f"训练集: X {X_train.shape}, y {y_train.shape}")
    #     print(f"验证集: X {X_val.shape}, y {y_val.shape}")
    #     print(f"测试集: X {X_test.shape}, y {y_test.shape}")
    #
    #     return X_train, X_val, X_test, y_train, y_val, y_test

    # 添加诊断代码到 data_prepare.py
    def analyze_time_periods(self, data):
        """分析不同时间段的统计特性"""
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
        print(f"验证集: {val_times[0]} 到 {val_times[-1]}")
        print(f"测试集: {test_times[0]} 到 {test_times[-1]}")

        # 计算每个时间段目标变量的统计值
        y = data[TARGET_VARIABLE].values

        train_y = y[:train_end].flatten()
        val_y = y[train_end:val_end].flatten()
        test_y = y[val_end:].flatten()

        print(f"\n目标变量统计:")
        print(f"训练集 - 均值: {train_y.mean():.3f}, 标准差: {train_y.std():.3f}")
        print(f"验证集 - 均值: {val_y.mean():.3f}, 标准差: {val_y.std():.3f}")
        print(f"测试集 - 均值: {test_y.mean():.3f}, 标准差: {test_y.std():.3f}")


    def run_pipeline(self):
        """运行完整的数据处理流程"""
        print("\n" + "=" * 50)
        print("启动完整数据处理流程")
        print("=" * 50)

        # 1. 下载数据
        data = self.download_data()

        # 2. 创建样本
        X, y = self.create_samples(data)



        # 4. 划分数据集
        # X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_norm, y_norm)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y,data)

        # 4. 归一化（只使用训练集统计）
        X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm = self.normalize_data(X_train, X_val, X_test, y_train, y_val, y_test)


        # 5. 保存数据
        # self.save_processed_data(X_train, X_val, y_train, y_val, X_test, y_test)

        self.save_processed_data(X_train_norm, X_val_norm,
                                 y_train_norm, y_val_norm,X_test_norm, y_test_norm)

        print("\n" + "=" * 50)
        print("数据处理流程完成！")
        print("=" * 50)

        # return X_train, X_val, X_test, y_train, y_val, y_test
        return X_train_norm, X_val_norm, X_test_norm,y_train_norm, y_val_norm, y_test_norm

    def process_fold(self, X_all, y_all, sample_times, fold_index):
        """
        【核心逻辑】处理单个 Fold：仅进行切分、归一化、保存
        不再重复下载或创建样本，直接使用传入的全量数据
        """
        print("传统单次分割")
        val_year = CV_VAL_YEARS[fold_index]
        train_end_year = val_year - 1

        print(f"\n  正在处理 Fold {fold_index + 1} (验证年份: {val_year})...")

        # 1. 计算切分索引 (逻辑与 create_samples 严格对应)
        t_train_end = np.datetime64(f"{train_end_year}-12-31")
        t_val_end = np.datetime64(f"{val_year}-12-31")
        t_test_start = np.datetime64(f"{TEST_YEAR}-01-01")
        t_test_end = np.datetime64(f"{TEST_YEAR}-12-31")

        train_end_idx = np.searchsorted(sample_times, t_train_end, side='right')
        val_end_idx = np.searchsorted(sample_times, t_val_end, side='right')
        test_start_idx = np.searchsorted(sample_times, t_test_start, side='left')
        test_end_idx = np.searchsorted(sample_times, t_test_end, side='right')

        test_start_idx = max(test_start_idx, val_end_idx)

        if train_end_idx >= val_end_idx or val_end_idx >= test_start_idx:
            print(f"警告：Fold {fold_index + 1} 数据集划分可能重叠或为空！")
            print(f"Indices: TrainEnd={train_end_idx}, ValEnd={val_end_idx}, TestStart={test_start_idx}")

        X_train, y_train = X_all[:train_end_idx], y_all[:train_end_idx]
        X_val, y_val = X_all[train_end_idx:val_end_idx], y_all[train_end_idx:val_end_idx]
        X_test, y_test = X_all[test_start_idx:test_end_idx], y_all[test_start_idx:test_end_idx]

        if len(X_val) == 0:
            raise ValueError(f"Fold {fold_index + 1} 验证集为空！检查年份配置。")

            # 3. 独立归一化 (关键：仅用 Train Fit)
        n_channels = X_train.shape[-1]
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        # Fit
        x_scaler.fit(X_train.reshape(-1, n_channels))
        y_scaler.fit(y_train.reshape(-1, 1))

        # Transform
        def transform_3d(arr, scaler):
            shape = arr.shape
            return scaler.transform(arr.reshape(-1, shape[-1])).reshape(shape)

        def transform_2d(arr, scaler):
            shape = arr.shape
            return scaler.transform(arr.reshape(-1, 1)).reshape(shape)

        X_train_n = transform_3d(X_train, x_scaler)
        X_val_n = transform_3d(X_val, x_scaler)
        X_test_n = transform_3d(X_test, x_scaler)

        y_train_n = transform_2d(y_train, y_scaler)
        y_val_n = transform_2d(y_val, y_scaler)
        y_test_n = transform_2d(y_test, y_scaler)

        current_fold_num = fold_index + 1
        print(f"   -> 准备保存为 Fold {current_fold_num} (后缀将自动生成 '_fold_{current_fold_num}')")

        self.save_processed_data(
            X_train_n, X_val_n, X_test_n,
            y_train_n, y_val_n, y_test_n,
            fold_index=current_fold_num  # <--- 直接传整数 1, 2, 3...
        )

        return len(X_train_n), len(X_val_n), len(X_test_n)

    def run_rolling_cv_pipeline(self):
        print("\n" + "=" * 70)
        print("    启动滚动交叉验证数据准备流程 (Rolling CV Pipeline)")
        print(f"   时间范围: {TIME_START} - {TIME_END}")
        print(f"   验证年份: {CV_VAL_YEARS}")
        print(f"   测试年份: {TEST_YEAR}")
        print("=" * 70)

        # 1. 准备全量数据 (只做一次)
        print("\n[1/2] 下载并构建全量样本 (此步骤仅执行一次)...")
        data = self.download_data()
        X_all, y_all = self.create_samples(data)

        total_samples = X_all.shape[0]
        stride = TIME_SAMPLING_STRIDE
        original_times = data.time.values
        indices = np.arange(total_samples) * stride
        indices = np.clip(indices, 0, len(original_times) - 1)
        sample_times = original_times[indices]

        # Step 2: 【循环】处理每个 Fold
        results = []
        print("\n[2/2] 开始循环切分、归一化并保存各 Fold 数据...")

        for i, year in enumerate(CV_VAL_YEARS):
            try:
                n_tr, n_va, n_te = self.process_fold(X_all, y_all, sample_times, i)

                results.append({
                    'fold': i + 1,
                    'val_year': year,
                    'train': n_tr,
                    'val': n_va,
                    'test': n_te,
                    'status': 'Success'
                })
                print(f"   Fold {i + 1} 完成 | Train:{n_tr}, Val:{n_va}, Test:{n_te}")

            except Exception as e:
                print(f"   Fold {i + 1} 失败: {e}")
                results.append({'fold': i + 1, 'status': f'Failed: {e}'})
                break  # 遇到错误立即停止

        # 汇总
        print("\n" + "=" * 70)
        print("处理完成汇总")
        print("=" * 70)
        for r in results:
            if r['status'] == 'Success':
                print(f"Fold {r['fold']} ({r['val_year']}): T={r['train']}, V={r['val']}, Te={r['test']}")
            else:
                print(f"Fold {r['fold']}: {r['status']}")

        return results

    # 保留原有的 load 方法，但增加加载指定 fold 的功能
    def load_processed_data(self, fold_index=None):
        """加载处理后的数据"""
        suffix = f"_fold_{fold_index}" if fold_index is not None else ""

        files = [
            f'X_train{suffix}.npy', f'X_val{suffix}.npy', f'X_test{suffix}.npy',
            f'y_train{suffix}.npy', f'y_val{suffix}.npy', f'y_test{suffix}.npy'
        ]

        paths = [os.path.join(PROCESSED_DATA_PATH, f) for f in files]

        # 检查文件
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"缺失文件: {missing}\n请先运行 run_rolling_cv_pipeline()")

        # 加载数据
        data_list = [np.load(p, mmap_mode='r') for p in paths]

        # 加载 Scaler
        scaler_path = os.path.join(PROCESSED_DATA_PATH, f'scalers{suffix}.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scalers = pickle.load(f)
        else:
            self.scalers = None
            print("  未找到对应的 Scaler 文件，将无法进行反归一化。")

        print(f" 已加载 Fold {fold_index} 数据 (内存映射模式)。")
        return tuple(data_list)
if __name__ == "__main__":
    # 测试数据处理
    preparer = WeatherDataPreparer()
    # X_train, X_val, X_test, y_train, y_val, y_test = preparer.run_pipeline()
    if USE_ROLLING_CV:
        print(">>> 模式：滚动交叉验证 (Rolling CV)")
        preparer.run_rolling_cv_pipeline()

        print("\n>>> 验证数据加载...")
        for i in range(1, len(CV_VAL_YEARS) + 1):
            X_tr, X_va, X_te, y_tr, y_va, y_te = preparer.load_processed_data(i)
            print(f"Fold {i}: Train={X_tr.shape}, Val={X_va.shape}, Test={X_te.shape}")

    else:
        # 否则执行旧流程
        print(">>> 模式：单次传统划分")
        X_train, X_val, X_test, y_train, y_val, y_test = preparer.run_pipeline()

        print(f"\n最终数据形状:")
        print(f"X_train: {X_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}")
        print(f"y_val: {y_val.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_test: {y_test.shape}")

    print("\n所有检查完成！")