"""
PyTorch数据集类
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import BATCH_SIZE


class WeatherDataset(Dataset):
    """气象数据集类"""

    def __init__(self, X, y):
        """
        Args:
            X: 输入数据，形状 (样本数, 输入时间步, 高度, 宽度, 通道)
            y: 输出数据，形状 (样本数, 输出时间步, 高度, 宽度)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test):
    """创建数据加载器"""

    # 创建数据集
    train_dataset = WeatherDataset(X_train, y_train)
    val_dataset = WeatherDataset(X_val, y_val)
    test_dataset = WeatherDataset(X_test, y_test)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows下设置为0
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"\n数据加载器创建完成!")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据集
    from data_prepare import WeatherDataPreparer

    # 加载数据
    preparer = WeatherDataPreparer()
    X_train, X_val, X_test, y_train, y_val, y_test = preparer.load_processed_data()

    # 创建数据加载器
    train_loader, val_loader,test_loader = create_data_loaders(X_train, X_val, y_train, y_val)

    # 测试一个batch
    for batch_X, batch_y in train_loader:
        print(f"\nBatch形状:")
        print(f"  X: {batch_X.shape}")
        print(f"  y: {batch_y.shape}")
        break