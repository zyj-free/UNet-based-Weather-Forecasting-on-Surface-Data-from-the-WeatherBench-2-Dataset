"""
训练脚本
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import time
import json
from config import *
from models import SimpleCNN, UNet
from dataset import create_data_loaders
from data_prepare import WeatherDataPreparer
from torch.optim import AdamW

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class Trainer:
    def __init__(self, model, train_loader, val_loader, device,learning_rate=1e-3, weight_decay=1e-5,fold_suffix=""):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.fold_suffix = fold_suffix  # 新增
        # 损失函数和优化器
        self.criterion = nn.MSELoss()

        # self.optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        #
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        # )
        # 优化器：建议使用 AdamW (带权重衰减的Adam)，比纯Adam泛化性更好
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # 学习率调度器：当验证集损失不再下降时，自动降低学习率
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.early_stopping = EarlyStopping(patience=15, min_delta=0.001)

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_metric = float('inf')

        # TensorBoard
        log_dir = os.path.join(MODEL_SAVE_PATH, f'logs{fold_suffix}')
        self.writer = SummaryWriter(log_dir)
        # self.writer = SummaryWriter(os.path.join(MODEL_SAVE_PATH, 'logs'))

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch_X, batch_y in self.train_loader:
            # print("原始 batch_X 形状:", batch_X.shape)
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)


            self.optimizer.zero_grad()

            # 前向传播
            output = self.model(batch_X)
            loss = self.criterion(output, batch_y)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                output = self.model(batch_X)
                loss = self.criterion(output, batch_y)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, epochs=EPOCHS):
        """完整训练流程"""
        print("\n" + "=" * 50)
        print("开始训练...")
        print(f"模型: {self.model.__class__.__name__}")
        print(f"设备: {self.device}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        print(f"批次大小: {BATCH_SIZE}")
        # print(f"学习率: {LEARNING_RATE}")
        print(f"初始学习率: {self.optimizer.param_groups[0]['lr']}")
        # 打印早停配置
        if hasattr(self, 'early_stopping'):
            print(f"早停策略: Patience={self.early_stopping.patience}, Min Delta={self.early_stopping.min_delta}")
        else:
            print("警告: 未检测到 early_stopping 实例，将运行完整 epochs。")

        print("=" * 50)

        start_time = time.time()
        stopped_epoch = epochs

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            if hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.step(val_loss)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']


            # 记录到TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_metric = val_loss
                self.save_model('best_model.pth')

            early_stop_triggered = False
            if hasattr(self, 'early_stopping') and self.early_stopping is not None:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    early_stop_triggered = True
                    stopped_epoch = epoch + 1
                    print(f"\n 早停触发 @ Epoch {epoch + 1}")
                    print(
                        f"   原因: 验证集损失连续 {self.early_stopping.patience} 次未改善 (最小增量: {self.early_stopping.min_delta})")
                    print(f"   当前 Val Loss: {val_loss:.6f}, 最佳 Val Loss: {self.early_stopping.best_loss:.6f}")
                    break  # 跳出训练循环
            # 打印进度
            if (epoch + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Time: {elapsed:.1f}s")

        # 保存最终模型
        self.save_model('final_model.pth')

        # 绘制损失曲线
        # self.plot_losses()
        fold_data = self.plot_losses()
        print("\n训练完成!")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        print(f"总训练时间: {time.time() - start_time:.1f}s")

        self.writer.close()
        return fold_data

    def save_model(self, filename):
        """保存模型"""
        if self.fold_suffix:
            # 去掉 .pth 后缀，插入 fold_suffix，再加回 .pth
            name, ext = os.path.splitext(filename)
            final_filename = f"{name}{self.fold_suffix}{ext}"
        else:
            final_filename = filename
        # model_path = os.path.join(MODEL_SAVE_PATH, filename)
        model_path = os.path.join(MODEL_SAVE_PATH, final_filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, model_path)
        print(f"模型已保存: {model_path}")

    def load_model(self, filename):
        """加载模型"""
        model_path = os.path.join(MODEL_SAVE_PATH, filename)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"模型已加载: {model_path}")

    def save_loss_data(self):
        """将当前 Fold 的 loss 数据追加保存到公共 JSON 文件中"""
        json_path = os.path.join(FIGURE_SAVE_PATH, 'all_folds_losses.json')

        # 准备当前折的数据
        current_fold_data = {
            'fold_suffix': self.fold_suffix if self.fold_suffix else 'default',
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }

        all_data = []

        # 如果文件存在，读取已有数据
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
            except:
                all_data = []

        # 追加当前数据
        all_data.append(current_fold_data)

        # 写回文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=4)

        print(f"损失数据已追加保存至: {json_path}")


    def plot_losses(self):
        """绘制损失曲线"""
        # plt.figure(figsize=(10, 6))
        # plt.plot(self.train_losses, label='Train Loss', alpha=0.8)
        # plt.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        # plt.xlabel('Epoch')
        # plt.ylabel('MSE Loss')
        # plt.title(f'{self.model.__class__.__name__} Training History')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.savefig(os.path.join(FIGURE_SAVE_PATH, 'training_history.png'), dpi=150, bbox_inches='tight')
        # plt.show()
        pass

def main():
    """主训练函数"""
    # 加载数据
    print("加载数据...")

    preparer = WeatherDataPreparer()
    X_train, X_val, X_test, y_train, y_val, y_test = preparer.load_processed_data()

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test)

    # 选择模型
    print("\n选择模型:")
    print("1. SimpleCNN")
    print("2. UNet")
    choice = input("请输入选择 (1或2): ").strip()

    if choice == '1':
        model = SimpleCNN()
        print("使用 SimpleCNN 模型")
    else:
        model = UNet()
        print("使用 UNet 模型")

    # 创建训练器并训练
    trainer = Trainer(model, train_loader, val_loader, DEVICE)
    trainer.train()


if __name__ == "__main__":
    main()