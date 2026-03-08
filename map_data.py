"""
quick_region_view.py
快速查看选中的区域位置
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def show_my_region():
    """显示我选中的区域"""

    # ===== 你的选中区域 =====
    LON_MIN, LON_MAX = 70.0, 150
    LAT_MIN, LAT_MAX = 0.0, 61.5

    #54x42
    # LON_MIN, LON_MAX = 70.0, 141.5
    # LAT_MIN, LAT_MAX = 3, 52
    #32x48
    # ========================

    # 创建图形
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # 设置地图范围（显示整个东亚）
    ax.set_extent([60, 180, -10, 80], crs=ccrs.PlateCarree())

    # 添加基础地图
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle='-')
    ax.add_feature(cfeature.OCEAN, alpha=0.3, color='lightblue')
    ax.add_feature(cfeature.LAND, alpha=0.2, color='lightgreen')
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.5, color='lightblue')

    # 绘制你的区域（红色半透明）
    import matplotlib.patches as mpatches
    rect = mpatches.Rectangle(
        (LON_MIN, LAT_MIN),
        LON_MAX - LON_MIN,
        LAT_MAX - LAT_MIN,
        facecolor='red',
        alpha=0.3,
        edgecolor='red',
        linewidth=3,
        transform=ccrs.PlateCarree(),
        label=f'你的区域: {LON_MIN}°E-{LON_MAX}°E, {LAT_MIN}°N-{LAT_MAX}°N'
    )
    ax.add_patch(rect)

    # 标记中国边界
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='blue')

    # 标记主要城市
    cities = {
        '北京': (116.4, 39.9),
        '上海': (121.5, 31.2),
        '广州': (113.3, 23.1),
        '香港': (114.2, 22.3),
        '台北': (121.5, 25.0),
        '拉萨': (91.1, 29.6),
        '乌鲁木齐': (87.6, 43.8),
        '哈尔滨': (126.5, 45.8),
        '漠河': (122.5, 53.5),
        '喀什': (75.9, 39.5),
        '抚远': (134.3, 48.4)
    }

    for city, (lon, lat) in cities.items():
        ax.plot(lon, lat, 'ro', markersize=4, transform=ccrs.PlateCarree())
        ax.text(lon+1, lat, city, fontsize=8, transform=ccrs.PlateCarree())

    # 添加网格线
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False

    # 添加图例
    ax.legend(loc='upper left')

    # 标题
    plt.title('你的选中区域位置', fontsize=16)

    # 保存图片
    plt.savefig('my_region.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "="*50)
    print("你的区域信息:")
    print("="*50)
    print(f"经度范围: {LON_MIN}°E 到 {LON_MAX}°E")
    print(f"纬度范围: {LAT_MIN}°N 到 {LAT_MAX}°N")
    print(f"区域大小: {LON_MAX-LON_MIN:.1f}° × {LAT_MAX-LAT_MIN:.1f}°")

    # 计算网格点
    lon_points = int((LON_MAX - LON_MIN) / 1.5) + 1
    lat_points = int((LAT_MAX - LAT_MIN) / 1.5) + 1
    print(f"网格点数: {lon_points} × {lat_points}")

    # 简单描述
    print("\n位置描述:")
    if LON_MIN <= 73:
        print("✅ 覆盖中国最西部（喀什73°E）")
    else:
        print(f"❌ 缺中国西部 {73-LON_MIN:.1f}°")

    if LON_MAX >= 135:
        print("✅ 覆盖中国最东部（抚远135°E）")
    else:
        print(f"❌ 缺中国东部 {135-LON_MAX:.1f}°")

    if LAT_MIN <= 3:
        print("✅ 覆盖中国最南部（南海3°N）")
    else:
        print(f"❌ 缺中国南部 {LAT_MIN-3:.1f}°")

    if LAT_MAX >= 53.5:
        print("✅ 覆盖中国最北部（漠河53.5°N）")
    else:
        print(f"❌ 缺中国北部 {53.5-LAT_MAX:.1f}°")

    print("="*50)

if __name__ == "__main__":
    show_my_region()