# import torch
#
# print("=" * 50)
# print("PyTorch GPU验证")
# print("=" * 50)
#
# print(f"PyTorch版本: {torch.__version__}")
# print(f"CUDA可用: {torch.cuda.is_available()}")
#
# if torch.cuda.is_available():
#     print(f"CUDA版本: {torch.version.cuda}")
#     print(f"GPU数量: {torch.cuda.device_count()}")
#     print(f"当前GPU: {torch.cuda.get_device_name(0)}")
#
#     # 测试GPU计算
#     x = torch.randn(3, 3).cuda()
#     y = torch.randn(3, 3).cuda()
#     z = torch.matmul(x, y)
#     print(f"\n✅ GPU计算测试成功!")
#     print(f"结果: \n{z}")
# else:
#     print("\n❌ GPU仍然不可用")


# test_gcs.py
# import gcsfs
# import time
#
# # 你的数据路径
# BUCKET_PATH = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
# # 提取桶名 (weatherbench2)
# bucket_name = BUCKET_PATH.split('/')[2]
#
# print(f"正在测试连接到 GCS 桶: {bucket_name} ...")
# start_time = time.time()
#
# try:
#     # 初始化文件系统 (匿名访问，因为 weatherbench2 是公开桶)
#     fs = gcsfs.GCSFileSystem(token="anon")
#
#     print("✅ GCS 文件系统对象创建成功")
#     print("正在尝试列出桶中的前 3 个条目...")
#
#     # 尝试列出文件 (设置 timeout 防止无限等待)
#     # 注意：对于巨大的 zarr 目录，ls 可能很慢，我们只取前几个
#     items = fs.ls(bucket_name, detail=False, maxdepth=1)
#
#     elapsed = time.time() - start_time
#     print(f"\n🎉 连接成功！耗时: {elapsed:.2f} 秒")
#     print(f"找到的前 3 个条目示例:")
#     for item in items[:3]:
#         print(f"  - {item}")
#
# except Exception as e:
#     elapsed = time.time() - start_time
#     print(f"\n❌ 连接失败！耗时: {elapsed:.2f} 秒")
#     print(f"错误类型: {type(e).__name__}")
#     print(f"错误详情: {e}")
#
#     # 常见错误建议
#     if "Timeout" in str(e) or "Connection" in str(e):
#         print("\n💡 建议: 网络连接超时。国内访问 GCS 通常不稳定，建议使用 cdsapi 下载本地。")
#     elif "Forbidden" in str(e) or "403" in str(e):
#         print("\n💡 建议: 权限被拒绝。虽然该桶通常是公开的，但可能需要配置 credentials。")
#     elif "NotFound" in str(e) or "404" in str(e):
#         print("\n💡 建议: 路径错误或桶不存在。")
#


def check_data_source_limits():
    """检查数据源的实际范围"""

    # ERA5数据的全球范围
    global_lat = [-90, 90]  # 纬度 -90°N 到 90°N
    global_lon = [0, 360]  # 经度 0°E 到 360°E

    # 但这个特定数据集的网格是 240×121
    # 121个纬度点：从90°N到-90°N，步长1.5°
    # 240个经度点：从0°E到360°E，步长1.5°

    print("数据源信息:")
    print(f"  全球经度点数: 240")
    print(f"  全球纬度点数: 121")
    print(f"  经度步长: 1.5°")
    print(f"  纬度步长: 1.5°")
    print(f"  纬度范围: 90°N 到 -90°N")

    # 你的请求
    requested_lat = [0, 100]
    requested_lon = [70, 166]

    print(f"\n你的请求:")
    print(f"  经度: {requested_lon}")
    print(f"  纬度: {requested_lat}")

    # 实际能获取的
    # 纬度最大90°，所以100°会被截断到90°
    actual_lat_max = min(100, 90)
    actual_lat_min = max(0, -90)

    print(f"\n实际获取:")
    print(f"  纬度: [{actual_lat_min}, {actual_lat_max}]")

    # 计算实际点数
    lat_points = int((actual_lat_max - actual_lat_min) / 1.5) + 1
    lon_points = int((166 - 70) / 1.5) + 1

    print(f"  纬度点数: {lat_points} (从0°到90°)")
    print(f"  经度点数: {lon_points}")


check_data_source_limits()