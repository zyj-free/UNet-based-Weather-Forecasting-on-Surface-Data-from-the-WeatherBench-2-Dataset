"""
PyTorch数据集类
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import BATCH_SIZE,INPUT_FRAMES,INPUT_CHANNELS,PRED_FRAMES,H,W,NUM_WORKERS


class WeatherDataset(Dataset):
    """气象数据集类"""
    def __init__(self, X, y):
        """
        Args:
            X: 输入数据，形状 (样本数, 输入时间步, 高度, 宽度, 通道)
            y: 输出数据，形状 (样本数, 输出时间步, 高度, 宽度)
        """
        # === 维度验证 ===
        assert len(X.shape) == 5, f"X 应该是 5 维，实际是 {len(X.shape)} 维"
        assert len(y.shape) == 4, f"y 应该是 4 维，实际是 {len(y.shape)} 维"
        assert X.shape[1] == INPUT_FRAMES, f"X 时间步应该是 {INPUT_FRAMES}，实际是 {X.shape[1]}"
        assert X.shape[2] == H, f"X 高度应该是 {H}，实际是 {X.shape[2]}"
        assert X.shape[3] == W, f"X 宽度应该是 {W}，实际是 {X.shape[3]}"
        assert X.shape[4] == INPUT_CHANNELS, f"X 通道应该是 {INPUT_CHANNELS}，实际是 {X.shape[4]}"
        assert y.shape[1] == PRED_FRAMES, f"y 时间步应该是 {PRED_FRAMES}，实际是 {y.shape[1]}"
        assert y.shape[2] == H, f"y 高度应该是 {H}，实际是 {y.shape[2]}"
        assert y.shape[3] == W, f"y 宽度应该是 {W}，实际是 {y.shape[3]}"

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test):
    """创建数据加载器"""

    # === 维度诊断 ===
    print("\n" + "=" * 50)
    print("数据维度诊断")
    print("=" * 50)
    print(f"X_train: {X_train.shape} 期望：(N, {INPUT_FRAMES}, {H}, {W}, {INPUT_CHANNELS})")
    print(f"y_train: {y_train.shape} 期望：(N, {PRED_FRAMES}, {H}, {W})")
    print(f"X_val:   {X_val.shape}")
    print(f"y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")
    # 创建数据集
    train_dataset = WeatherDataset(X_train, y_train)
    val_dataset = WeatherDataset(X_val, y_val)
    test_dataset = WeatherDataset(X_test, y_test)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # num_workers=NUM_WORKERS,  # Windows下设置为0
        num_workers=0,#unet fnn mlp
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        # num_workers=NUM_WORKERS,
        num_workers=0,  # unet fnn mlp
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        # num_workers=NUM_WORKERS,
        num_workers=0,  # unet fnn mlp
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"\n数据加载器创建完成!")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    print(f"Worker 数量: {NUM_WORKERS}")  # 加一行打印确认

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据集
    from data_prepare import WeatherDataPreparer

    # 加载数据
    preparer = WeatherDataPreparer()
    X_train, X_val, X_test, y_train, y_val, y_test = preparer.load_processed_data()

    # 创建数据加载器
    train_loader, val_loader,test_loader = create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test)

    # 测试一个batch
    for batch_X, batch_y in train_loader:
        print(f"\nBatch形状:")
        print(f"  X: {batch_X.shape}")
        print(f"  y: {batch_y.shape}")
        break