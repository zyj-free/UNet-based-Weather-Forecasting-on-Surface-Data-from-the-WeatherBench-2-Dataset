# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd

# # --- 1. 数据准备 ---
# # 你的实验数据
# data = {
#     'Model': [
#         '1_LightweightUNet',
#         '2_LightSharedWeight',
#         '3_LightIndepWeight',
#         '4_LightSharedWeight_History',
#         '5_LightIndepWeight_History'
#     ],
#     'Best Val RMSE': [0.519,0.513,0.515,0.505,0.499],
#     'Test RMSE': [0.511, 0.505, 0.506, 0.497, 0.492],
#     'Test MAE': [0.356, 0.356, 0.360, 0.350, 0.346],
#     'Test R2': [0.737, 0.737, 0.731, 0.746, 0.751],
#     'Extreme RMSE': [0.94695, 0.94770, 0.96944, 0.91061, 0.88484]
# }

# df = pd.DataFrame(data)

# # --- 2. 样式设置 (美化) ---
# # 设置风格：白色网格，无干扰
# sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)

# # 颜色定义 (使用专业色系，如深蓝、蓝绿、金色)
# colors = sns.color_palette("deep") # 使用 seaborn 默认的深色系，保证色盲友好
# # 或者自定义一组更优雅的颜色
# custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # 经典分类色

# # 创建画布 (2行2列布局)
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# fig.suptitle('Model Performance Comparison Report', fontsize=20, fontweight='bold', y=0.98)

# # --- 3. 绘制子图 ---

# # 1. 测试集 RMSE 对比 (柱状图)
# bars1 = axes[0, 0].bar(df['Model'], df['Test RMSE'], color=custom_colors, alpha=0.85, edgecolor='black', linewidth=0.8)
# axes[0, 0].set_title('Test RMSE Comparison', fontsize=14, pad=15)
# axes[0, 0].set_ylabel('RMSE', fontsize=12)
# axes[0, 0].tick_params(axis='x', rotation=15) # 旋转X轴标签防止重叠

# # 在柱子上方添加数值标签
# for bar in bars1:
#     height = bar.get_height()
#     axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
#                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# # 2. 验证集 RMSE 对比 (折线图 + 散点)
# axes[0, 1].plot(df['Model'], df['Best Val RMSE'], marker='o', markersize=8,
#                 linewidth=2.5, color='#d62728', label='Validation RMSE')
# axes[0, 1].set_title('Validation RMSE Trend', fontsize=14, pad=15)
# axes[0, 1].set_ylabel('RMSE', fontsize=12)
# axes[0, 1].tick_params(axis='x', rotation=15)
# axes[0, 1].grid(True, alpha=0.3)
# axes[0, 1].legend()

# # 3. 极端值 RMSE 对比 (横向柱状图)
# bars3 = axes[1, 0].barh(df['Model'], df['Extreme RMSE'], color=sns.light_palette("navy", reverse=True),
#                         edgecolor='grey', linewidth=0.5)
# axes[1, 0].set_title('Extreme RMSE (Worst Case)', fontsize=14, pad=15)
# axes[1, 0].set_xlabel('Extreme RMSE', fontsize=12)

# # 4. R2 分数对比 (面积图模拟)
# axes[1, 1].fill_between(range(len(df)), df['Test R2'], alpha=0.6, color='#8c564b', label='R² Score')
# axes[1, 1].plot(range(len(df)), df['Test R2'], marker='s', color='#8c564b', markersize=6)
# axes[1, 1].set_title('R² Score Comparison', fontsize=14, pad=15)
# axes[1, 1].set_ylabel('R²')
# axes[1, 1].set_xticks(range(len(df)))
# axes[1, 1].set_xticklabels(df['Model'], rotation=15)
# axes[1, 1].legend()

# # --- 4. 通用布局优化 ---
# plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 为大标题留出空间

# # --- 5. 保存高清图片 (适合论文插入) ---
# # dpi=300 保证清晰度，bbox_inches='tight' 防止标签被裁剪
# plt.savefig('model_performance_benchmark.png', dpi=300, bbox_inches='tight', facecolor='white')
# print("📊 图表已生成并保存为 'model_performance_benchmark.png'")

# # 显示图表
# plt.show()

# # --- 6. 额外：参数与精度散点图 (如果需要分析模型复杂度与性能的关系) ---
# plt.figure(figsize=(10, 6))
# # 假设参数量 (这里用索引代替，或者你需要提供具体的参数量数据)
# param_estimates = [1.0, 1.1, 1.5, 1.6, 1.7] # 示例数据，实际请替换为真实的 Params 数量

# scatter = plt.scatter(param_estimates, df['Test RMSE'], s=150, c=df['Test R2'], cmap='viridis', alpha=0.8, edgecolors='w', linewidth=2)
# for i, txt in enumerate(df['Model']):
#     plt.annotate(txt.split('_')[1], (param_estimates[i], df['Test RMSE'][i]), fontsize=10, ha='center')

# plt.colorbar(scatter, label='R² Score')
# plt.title('Model Complexity (Params) vs Performance (RMSE)', fontsize=16, pad=20)
# plt.xlabel('Estimated Model Size (Params)', fontsize=12)
# plt.ylabel('Test RMSE', fontsize=12)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('complexity_vs_performance.png', dpi=300, bbox_inches='tight')
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # ================= 1. 数据准备 =================
# # 模型名称 (用于X轴)
# models = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']
# full_names = [
#     'LightweightUNet', 
#     'L.SharedWeight', 
#     'L.IndepWeight', 
#     'L.Shared_Hist', 
#     'L.Indep_Hist'
# ]

# # 原始数据录入
# # 顺序: RMSE, R²
# raw_rmse = np.array([0.5053, 0.5052, 0.5113, 0.4972, 0.4930])
# raw_r2 = np.array([0.7379, 0.7348, 0.7317, 0.7463, 0.7507])

# # 计算 MSE (RMSE的平方)
# mse_data = raw_rmse ** 2

# # 设置绘图风格
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans'] # 防止中文乱码
# plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
# plt.figure(figsize=(10, 12)) # 设置画布大小 (宽, 高)

# # ================= 2. 绘制折线图 (上方) =================
# ax1 = plt.subplot(2, 1, 1) # 2行1列，第1个子图

# # 创建双Y轴
# ax2 = ax1.twinx()

# # 绘制 MSE 折线 (左轴)
# line1 = ax1.plot(models, mse_data, color='#D62728', marker='o', linewidth=2, markersize=8, label='MSE')
# ax1.set_ylabel('MSE (Mean Squared Error)', fontsize=12, color='#D62728')
# ax1.tick_params(axis='y', labelcolor='#D62728')
# ax1.grid(True, linestyle='--', alpha=0.6)
# ax1.set_title('Performance Comparison: MSE and $R^2$ across Models', fontsize=14, fontweight='bold', pad=20)

# # 绘制 R² 折线 (右轴)
# line2 = ax2.plot(models, raw_r2, color='#1F77B4', marker='s', linewidth=2, markersize=8, label='$R^2$ Score')
# ax2.set_ylabel('$R^2$ (Coefficient of Determination)', fontsize=12, color='#1F77B4')
# ax2.tick_params(axis='y', labelcolor='#1F77B4')
# # 设置R2的y轴范围，让变化更明显 (例如从0.72开始)
# ax2.set_ylim(0.72, 0.76) 

# # 合并图例
# lines = line1 + line2
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc='upper left', fontsize=11)

# # ================= 3. 绘制柱状图 (下方) =================
# ax3 = plt.subplot(2, 1, 2) # 2行1列，第2个子图

# # 绘制 MSE 柱状图
# bars = ax3.bar(models, mse_data, color='#FF9999', edgecolor='#D62728', alpha=0.7, width=0.6)

# ax3.set_xlabel('Model Variants', fontsize=12)
# ax3.set_ylabel('MSE Value', fontsize=12)
# ax3.set_title('MSE Error Distribution by Model', fontsize=14, fontweight='bold', pad=20)
# ax3.grid(axis='y', linestyle='--', alpha=0.6)

# # 在柱状图上方添加具体数值标签
# for bar in bars:
#     yval = bar.get_height()
#     ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.0005, round(yval, 4), ha='center', va='bottom', fontsize=10)

# # 调整布局，防止重叠
# plt.tight_layout()

# # 显示图表
# plt.show()

# ===========================================
# 多个模型对比
# ===========================================
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.gridspec as gridspec

# # ================= 1. 数据准备 =================
# models = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']

# # 模型全称映射
# model_full_names = [
#     'LightweightUNet',
#     'LightweightSharedWeightUNet',
#     'LightweightIndependentWeightUNet',
#     'LightweightSharedWeightUNet_History',
#     'LightweightIndependentWeightUNet_History'
# ]

# # 原始数据
# rmse_data = np.array([0.5053, 0.5052, 0.5113, 0.4972, 0.4930])
# r2_data   = np.array([0.7379, 0.7348, 0.7317, 0.7463, 0.7507])
# extreme_data = np.array([0.9470, 0.9477, 0.9694, 0.9106, 0.8848])

# # 计算 MSE
# mse_data = rmse_data ** 2

# # ================= 2. 画布与布局设置 (核心修改) =================
# fig = plt.figure(figsize=(14, 9))

# # 修改点：将画布分为3行
# # 第1行(图表):高度4, 第2行(空白):高度0.8, 第3行(表格):高度1
# # hspace=0 确保行之间紧密相连，利用空白行控制间距
# gs = gridspec.GridSpec(3, 1, height_ratios=[4, 0.8, 1], hspace=0)

# # 图表放在第0行
# ax1 = fig.add_subplot(gs[0])

# # ================= 3. 绘制图表 (保持不变) =================

# # --- 左轴: MSE (红色折线) ---
# color_mse = '#D62728'
# line1 = ax1.plot(models, mse_data, color=color_mse, marker='o', linewidth=2.5, markersize=8, label='MSE')
# ax1.set_ylabel('MSE (Mean Squared Error)', fontsize=12, color=color_mse, fontweight='bold')
# ax1.tick_params(axis='y', labelcolor=color_mse, labelsize=10)
# ax1.set_ylim(0.24, 0.265)
# ax1.grid(True, linestyle='--', alpha=0.4, axis='y', zorder=0)

# # --- 中轴: R² (蓝色折线) ---
# ax2 = ax1.twinx()
# color_r2 = '#1F77B4'
# line2 = ax2.plot(models, r2_data, color=color_r2, marker='s', linewidth=2.5, markersize=8, label='$R^2$ Score')
# ax2.set_ylabel('$R^2$ Score', fontsize=12, color=color_r2, fontweight='bold', rotation=270, labelpad=15)
# ax2.tick_params(axis='y', labelcolor=color_r2, labelsize=10)
# ax2.set_ylim(0.72, 0.76)

# # --- 右轴: 极端天气 (绿色柱状图) ---
# ax3 = ax1.twinx()
# color_ext = '#2CA02C'
# ax3.spines['right'].set_position(('outward', 60))

# bar_width = 0.4
# index = np.arange(len(models))
# bars = ax3.bar(index, extreme_data, width=bar_width, color=color_ext, alpha=0.2, edgecolor=color_ext, linewidth=1.2, label='Extreme Weather Index')
# ax3.set_ylabel('Extreme Weather Index', fontsize=12, color=color_ext, fontweight='bold', rotation=270, labelpad=15)
# ax3.tick_params(axis='y', labelcolor=color_ext, labelsize=10)
# ax3.set_xticks(index)
# ax3.set_xticklabels(models, fontsize=11)
# ax3.set_ylim(0, 1.1)

# # --- 添加柱状图数值标签 ---
# for bar in bars:
#     height = bar.get_height()
#     ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
#              f'{height:.3f}',
#              ha='center', va='bottom', fontsize=10, fontweight='bold', color=color_ext)

# # --- 图例 ---
# lines = line1 + line2 + [bars]
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc='upper left', frameon=True, fontsize=11, shadow=False, ncol=3)

# # 标题
# plt.suptitle('Combined Performance Metrics on Test Set', fontsize=15, fontweight='bold', y=0.98)

# # ================= 4. 底部表格绘制 (核心修改) =================
# # 修改点：表格放在第2行 (gs[2])，跳过中间的 gs[1]
# ax_table = fig.add_subplot(gs[2])
# ax_table.axis('off')

# # 准备表格数据
# table_data = [[models[i], model_full_names[i]] for i in range(len(models))]
# columns = ["Model ID", "Full Architecture Name"]

# # 创建表格
# the_table = ax_table.table(cellText=table_data,
#                            colLabels=columns,
#                            colWidths=[0.15, 0.85],
#                            loc='center',
#                            cellLoc='center')

# # 美化表格样式
# the_table.set_fontsize(11)
# the_table.scale(1, 2)

# # 设置表头样式
# for j in range(len(columns)):
#     the_table[(0, j)].set_facecolor('#1F77B4')
#     the_table[(0, j)].set_text_props(color='w', weight='bold')

# # 设置单元格边框
# for key, cell in the_table.get_celld().items():
#     cell.set_linewidth(0.8)
#     cell.set_edgecolor('gray')

# # ================= 5. 最终调整 =================
# # 保持原有的边距设置，确保图表内部不被压缩
# plt.subplots_adjust(left=0.12, right=0.78, top=0.92, bottom=0.25)

# plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec

# ================= 1. 数据准备 =================
data = {
    'model_name': [
        'Model A1', 'Model A2', 'Model A3', 
        'Model A4', 'Model A5', 'Model B1', 
        'Model B2', 'Model B3', 'Model B4', 'Model B5'
    ],
    'rmse': [
        0.49540936946868896, 0.4954076111316681, 0.4958805441856384, 
        0.49587342143058777, 0.49542102217674255, 0.48628780245780945, 
        0.48628339171409607, 0.4863823652267456, 0.4863966703414917, 
        0.4863009750843048
    ],
    'latency_ms': [
        3190.5970099993283, 834.805434998998, 1304.3291049994878, 
        3133.174324997526, 742.0273900017492, 3089.8460499971407, 
        844.391829999222, 1215.6624500028556, 3052.8258249993087, 
        749.463804998959
    ],
    'model_size_mb': [
        0.484161376953125, 0.484161376953125, 0.2420806884765625, 
        0.2420806884765625, 0.24208831787109375, 0.48664093017578125, 
        0.48664093017578125, 0.24332046508789062, 0.24332046508789062, 
        0.24332809448242188
    ]
}

df = pd.DataFrame(data)

# 模型全称映射
full_names = {
    'Model A1': 'Shared Weight FP32 Baseline',
    'Model A2': 'Shared Weight Channels Last Only',
    'Model A3': 'Shared Weight Uniform Int8 Quantization',
    'Model A4': 'Shared Weight State-Aware Int8 Quantization',
    'Model A5': 'Shared Weight FP16',
    'Model B1': 'Independent Weight FP32 Baseline',
    'Model B2': 'Independent Weight Channels Last Only',
    'Model B3': 'Independent Weight Uniform Int8 Quantization',
    'Model B4': 'Independent Weight State-Aware Int8 Quantization',
    'Model B5': 'Independent Weight FP16'
}

# ================= 2. 画布布局 (GridSpec) =================
fig = plt.figure(figsize=(14, 10))
# 定义网格：3行1列
# 第1行：图表 (高度 6)
# 第2行：空白间隙 (高度 1) -> 解决压住横轴的问题
# 第3行：表格 (高度 3)
gs = gridspec.GridSpec(3, 1, height_ratios=[6, 1, 3], hspace=0.05)

ax_main = fig.add_subplot(gs[0])
ax_table = fig.add_subplot(gs[2])
ax_table.axis('off') 

# ================= 3. 绘制三个柱状图 =================
x = np.arange(len(df))
width = 0.25 

r1 = x - width
r2 = x
r3 = x + width

# --- 配色方案优化 (现代商务风) ---
# RMSE: 深藏青色
color_rmse = '#2C3E50' 
# Latency: 钢蓝色
color_lat = '#3498DB' 
# Size: 珊瑚橙色
color_size = '#E67E22' 

# 1. RMSE (左侧柱子)
bars1 = ax_main.bar(r1, df['rmse'], color=color_rmse, width=width, label='RMSE', alpha=0.9, edgecolor='white', linewidth=1)

# 2. Latency (中间柱子)
ax_lat = ax_main.twinx()
bars2 = ax_lat.bar(r2, df['latency_ms'], color=color_lat, width=width, label='Latency', alpha=0.7, edgecolor='white', linewidth=1)

# 3. Model Size (右侧柱子)
ax_size = ax_main.twinx()
ax_size.spines['right'].set_position(('outward', 60)) 
bars3 = ax_size.bar(r3, df['model_size_mb'], color=color_size, width=width, label='Model Size', alpha=0.7, edgecolor='white', linewidth=1)

# 设置 X 轴
ax_main.set_xlabel('Model Variants', fontsize=12, fontweight='bold', labelpad=10)
ax_main.set_xticks(x)
ax_main.set_xticklabels(df['model_name'], rotation=0, fontsize=10)

# 设置 Y 轴标签颜色对应
ax_main.set_ylabel('RMSE', fontsize=12, color=color_rmse, fontweight='bold')
ax_main.tick_params(axis='y', labelcolor=color_rmse)

# 【关键修改】设置 RMSE 轴从 0.3 开始，放大差异
ax_main.set_ylim(0.4, max(df['rmse']) * 1.15) 
# 设置 Latency 轴从 500 开始，放大差异
ax_lat.set_ylim(500, max(df['latency_ms']) * 1.15)
# 设置 Model Size 轴从 0.2 开始，放大差异
ax_size.set_ylim(0.1, max(df['model_size_mb']) * 1.15)

ax_lat.set_ylabel('Latency (ms)', fontsize=12, color=color_lat, fontweight='bold', labelpad=10)
ax_lat.tick_params(axis='y', labelcolor=color_lat)



ax_size.set_ylabel('Size (MB)', fontsize=12, color=color_size, fontweight='bold', labelpad=10)
ax_size.tick_params(axis='y', labelcolor=color_size)


# 标题
plt.suptitle('Performance Comparison: RMSE, Latency & Model Size', fontsize=15, fontweight='bold', y=0.93)

# 图例合并
lines_labels = [ax_main.get_legend_handles_labels(), ax_lat.get_legend_handles_labels(), ax_size.get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax_main.legend(lines, labels, loc='upper left', ncol=3, frameon=True, shadow=False, fontsize=10)

# 网格 (仅针对主Y轴，样式微调)
ax_main.grid(True, axis='y', linestyle='--', alpha=0.2, zorder=0)

# ================= 4. 添加数值标签 =================
def add_labels(bars, ax, fmt="{:.3f}", rotation=90, v_offset=0, color='black'):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + v_offset,
                fmt.format(height),
                ha='center', va='bottom', fontsize=8, rotation=rotation, fontweight='bold', color=color)

# 为三组柱子添加标签
add_labels(bars1, ax_main, "{:.3f}", v_offset=0.002, color='#2C3E50') # RMSE
add_labels(bars2, ax_lat, "{:.0f}", v_offset=100, color='#3498DB')     # Latency
add_labels(bars3, ax_size, "{:.3f}", v_offset=0.005, color='#E67E22') # Size

# ================= 5. 绘制底部表格 =================
table_data = [[name, full_names[name]] for name in df['model_name']]
columns = ["Model ID", "Full Architecture Name"]

the_table = ax_table.table(cellText=table_data,
                           colLabels=columns,
                           colWidths=[0.2, 0.8],
                           loc='center',
                           cellLoc='center')

the_table.set_fontsize(10)
the_table.scale(1, 1.5)

# 表格样式美化
for j in range(len(columns)):
    the_table[(0, j)].set_facecolor('#2C3E50') # 深藏青表头，与RMSE呼应
    the_table[(0, j)].set_text_props(color='w', weight='bold')
    the_table[(0, j)].set_edgecolor('white')

for i in range(1, len(table_data) + 1):
    face_color = '#F8F9FA' if i % 2 == 0 else '#FFFFFF'
    the_table[(i, 0)].set_facecolor(face_color)
    the_table[(i, 1)].set_facecolor(face_color)
    the_table[(i, 0)].set_edgecolor('#DDDDDD')
    the_table[(i, 1)].set_edgecolor('#DDDDDD')

the_table.auto_set_column_width(col=1)
the_table._bbox = [0.1, 0.1, 0.8, 0.8] 

# ================= 6. 最终边距调整 =================
plt.subplots_adjust(left=0.1, right=0.82, top=0.88, bottom=0.25)

plt.show()