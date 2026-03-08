"""
主程序入口 - 整合所有模块
"""
import os
import argparse
from data_prepare import WeatherDataPreparer
from dataset import create_data_loaders
from train import Trainer
from models import SimpleCNN, UNet
from config import *


def setup_environment():
    """设置运行环境"""
    print("\n" + "=" * 50)
    print("Weather Forecasting System")
    print("=" * 50)
    print(f"PyTorch设备: {DEVICE}")
    print(f"数据路径: {PROCESSED_DATA_PATH}")
    print(f"模型保存路径: {MODEL_SAVE_PATH}")
    print(f"图像保存路径: {FIGURE_SAVE_PATH}")

    # 创建必要的目录
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(FIGURE_SAVE_PATH, exist_ok=True)


def prepare_data(force=False):
    """准备数据"""
    data_path = os.path.join(PROCESSED_DATA_PATH, 'X_train.npy')

    if os.path.exists(data_path) and not force:
        print("\n找到已处理的数据文件，跳过数据准备...")
        preparer = WeatherDataPreparer()

        X_train, X_val, X_test, y_train, y_val, y_test = preparer.load_processed_data()
        return X_train, X_val, X_test, y_train, y_val, y_test, preparer.scalers
    else:
        print("\n开始数据准备流程...")
        preparer = WeatherDataPreparer()
        X_train, X_val, X_test, y_train, y_val, y_test = preparer.run_pipeline()
        return X_train, X_val, X_test, y_train, y_val, y_test, preparer.scalers


def train_model(X_train, X_val, X_test, y_train, y_val, y_test, model_type='unet'):
    """训练模型"""
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    # 选择模型
    if model_type.lower() == 'cnn':
        model = SimpleCNN()
        print("\n使用 SimpleCNN 模型")
    else:
        model = UNet()
        print("\n使用 UNet 模型")

    # 创建训练器并训练
    trainer = Trainer(model, train_loader, val_loader, DEVICE)
    trainer.train()

    return trainer


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Weather Forecasting System')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['prepare', 'train', 'full'],
                        help='运行模式: prepare(仅准备数据), train(仅训练), full(完整流程)')
    parser.add_argument('--model', type=str, default='unet',
                        choices=['cnn', 'unet'],
                        help='模型类型: cnn 或 unet')
    parser.add_argument('--force', action='store_true',
                        help='强制重新处理数据')

    args = parser.parse_args()

    # 设置环境
    setup_environment()

    if args.mode == 'prepare':
        # 仅准备数据
        prepare_data(force=args.force)
        print("\n数据准备完成！")

    elif args.mode == 'train':
        # 仅训练模型（假设数据已准备好）
        try:
            preparer = WeatherDataPreparer()
            X_train, X_val, X_test, y_train, y_val, y_test = preparer.load_processed_data()
            train_model(X_train, X_val, X_test, y_train, y_val, y_test, args.model)
        except FileNotFoundError:
            print("\n错误: 未找到处理后的数据文件。请先运行 'python main.py --mode prepare'")

    else:  # full mode
        # 完整流程
        print("\n" + "=" * 50)
        print("运行完整流程")
        print("=" * 50)

        # 1. 准备数据
        X_train, X_val, X_test, y_train, y_val, y_test, scalers = prepare_data(force=args.force)

        # 2. 训练模型
        trainer = train_model(X_train, X_val, X_test, y_train, y_val, y_test, args.model)

        print("\n" + "=" * 50)
        print("完整流程完成！")
        print(f"模型保存位置: {MODEL_SAVE_PATH}")
        print(f"可视化结果保存位置: {FIGURE_SAVE_PATH}")
        print("=" * 50)


if __name__ == "__main__":
    main()