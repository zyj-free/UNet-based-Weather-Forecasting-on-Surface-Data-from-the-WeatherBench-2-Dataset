"""
评估和可视化脚本
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from config import *
from models import SimpleCNN, UNet
from dataset import create_data_loaders
from data_prepare import WeatherDataPreparer
import pickle

class Evaluator:
    def __init__(self, model, test_loader, scalers, device):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.scalers = scalers
        self.device = device

    def evaluate(self):
        """评估模型性能"""
        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                output = self.model(batch_X)

                all_predictions.append(output.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        # 合并所有批次
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # 反归一化
        y_scaler = self.scalers['target']
        predictions_original = y_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).reshape(predictions.shape)
        targets_original = y_scaler.inverse_transform(
            targets.reshape(-1, 1)
        ).reshape(targets.shape)

        # 计算指标
        mse = mean_squared_error(targets_original.flatten(), predictions_original.flatten())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets_original.flatten(), predictions_original.flatten())

        # 计算每个时间步的RMSE
        time_rmse = []
        for t in range(predictions_original.shape[1]):
            t_rmse = np.sqrt(mean_squared_error(
                targets_original[:, t].flatten(),
                predictions_original[:, t].flatten()
            ))
            time_rmse.append(t_rmse)

        print("\n" + "="*50)
        print("模型评估结果")
        print("="*50)
        print(f"整体 MSE: {mse:.2f}")
        print(f"整体 RMSE: {rmse:.2f}")
        print(f"整体 MAE: {mae:.2f}")
        print("\n各时间步RMSE:")
        for t, rmse_t in enumerate(time_rmse):
            print(f"  第{t+1}步 ({(t+1)*TIME_STEP_HOURS}小时): {rmse_t:.2f}")

        return predictions_original, targets_original

    def plot_predictions(self, predictions, targets, num_samples=3):
        """绘制预测结果对比"""
        n_samples = min(num_samples, len(predictions))
        n_steps = predictions.shape[1]

        # 创建图形：n_samples 行，n_steps * 2 列 (预测+真实)
        fig, axes = plt.subplots(n_samples, n_steps * 2,
                                 figsize=(4 * n_steps * 2, 4 * n_samples))

        # 【修复核心】强制将 axes 转换为二维数组，防止维度塌陷
        # 如果 n_samples=1 或 n_steps*2=1，axes 可能会变成 1D 数组或标量
        if n_samples == 1 and n_steps * 2 == 1:
            axes = np.array([[axes]])
        elif n_samples == 1:
            axes = axes.reshape(1, -1)
        elif n_steps * 2 == 1:
            axes = axes.reshape(-1, 1)

        # 现在 axes 一定是二维的 [row, col]

        for i in range(n_samples):
            # 随机选择一个样本
            idx = np.random.randint(len(predictions))

            for t in range(n_steps):
                # 预测值 (列索引: t)
                ax_pred = axes[i, t]
                # 确保数据是 2D (H, W)，如果是单通道可能需要 squeeze
                pred_data = predictions[idx, t]
                if pred_data.ndim == 3: pred_data = pred_data.squeeze()

                im_pred = ax_pred.imshow(pred_data, cmap='viridis')
                ax_pred.set_title(f'Sample {idx + 1} - Pred t+{t + 1}')
                ax_pred.axis('off')
                plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

                # 真实值 (列索引: t + n_steps)
                ax_true = axes[i, t + n_steps]
                true_data = targets[idx, t]
                if true_data.ndim == 3: true_data = true_data.squeeze()

                im_true = ax_true.imshow(true_data, cmap='viridis')
                ax_true.set_title(f'Sample {idx + 1} - True t+{t + 1}')
                ax_true.axis('off')
                plt.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_SAVE_PATH, 'predictions_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.show()

    def plot_error_maps(self, predictions, targets, num_samples=3):
        """绘制误差图"""
        n_samples = min(num_samples, len(predictions))
        n_steps = predictions.shape[1]

        # 创建图形：n_samples 行，n_steps 列
        fig, axes = plt.subplots(n_samples, n_steps,
                                 figsize=(4 * n_steps, 4 * n_samples))

        # 【修复核心】强制将 axes 转换为二维数组
        if n_samples == 1 and n_steps == 1:
            axes = np.array([[axes]])
        elif n_samples == 1:
            axes = axes.reshape(1, -1)
        elif n_steps == 1:
            axes = axes.reshape(-1, 1)

        for i in range(n_samples):
            idx = np.random.randint(len(predictions))

            for t in range(n_steps):
                error = predictions[idx, t] - targets[idx, t]
                if error.ndim == 3: error = error.squeeze()

                # 现在可以安全地使用 [i, t] 索引
                ax = axes[i, t]

                max_err = np.abs(error).max()
                # 避免 max_err 为 0 导致颜色条错误
                vmin, vmax = (-max_err, max_err) if max_err > 0 else (-1, 1)

                im = ax.imshow(error, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                ax.set_title(f'Sample {idx + 1} - Error t+{t + 1}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_SAVE_PATH, 'error_maps.png'),
                    dpi=150, bbox_inches='tight')
        plt.show()

    def plot_geographic_predictions(self, predictions, targets,
                                   lat_range=(LAT_MIN, LAT_MAX),
                                   lon_range=(LON_MIN, LON_MAX)):
        """在地图上绘制预测结果"""
        # 创建经纬度网格
        lats = np.linspace(lat_range[1], lat_range[0], predictions.shape[2])
        lons = np.linspace(lon_range[0], lon_range[1], predictions.shape[3])

        fig = plt.figure(figsize=(20, 8))

        for t in range(min(4, predictions.shape[1])):
            # 选择一个样本
            idx = np.random.randint(len(predictions))

            # 预测
            ax1 = fig.add_subplot(2, 4, t+1, projection=ccrs.PlateCarree())
            self._plot_geographic(ax1, lons, lats, predictions[idx, t],
                                 f'Prediction t+{t+1}')

            # 真实值
            ax2 = fig.add_subplot(2, 4, t+5, projection=ccrs.PlateCarree())
            self._plot_geographic(ax2, lons, lats, targets[idx, t],
                                 f'True t+{t+1}')

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_SAVE_PATH, 'geographic_predictions.png'),
                   dpi=150, bbox_inches='tight')
        plt.show()

    def _plot_geographic(self, ax, lons, lats, data, title):
        """辅助函数：在地图上绘制数据"""
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS, alpha=0.5)

        im = ax.contourf(lons, lats, data, transform=ccrs.PlateCarree(),
                         cmap='viridis', levels=20)
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05)

        ax.set_title(title)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

def main():
    """主评估函数"""
    # 加载数据
    print("加载数据...")
    preparer = WeatherDataPreparer()
    X_train, X_val, X_test, y_train, y_val, y_test = preparer.load_processed_data()

    # 使用验证集作为测试集
    _, _, test_loader = create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test)  # 只取val_loader

    # 加载模型
    print("\n加载模型...")
    model = UNet()  # 或 SimpleCNN()

    # 加载训练好的权重
    model_path = os.path.join(MODEL_SAVE_PATH, 'best_model.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型加载成功: {model_path}")
    else:
        print(f"警告: 未找到模型文件 {model_path}")
        return

    # 创建评估器
    evaluator = Evaluator(model, test_loader, preparer.scalers, DEVICE)

    # 评估
    predictions, targets = evaluator.evaluate()

    # 可视化
    print("\n生成可视化结果...")
    evaluator.plot_predictions(predictions, targets, num_samples=3)
    evaluator.plot_error_maps(predictions, targets, num_samples=3)
    evaluator.plot_geographic_predictions(predictions, targets)

    print(f"\n可视化结果已保存到: {FIGURE_SAVE_PATH}")

if __name__ == "__main__":
    main()