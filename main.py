"""
主程序入口 - 整合所有模块
"""
import os
import argparse
from data_prepare import WeatherDataPreparer
from dataset import create_data_loaders
from train import Trainer
from models import SimpleCNN, UNet,FNN,MLP,ConvLSTM,ConvLSTMCell,SwinTransformerModel
from config import *
import numpy as np
import matplotlib.pyplot as plt

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

def prepare_data_single_fold(preparer, fold_index=None):
    """
    【修改点】加载单个 Fold 的数据
    fold_index: None (传统模式) 或 1, 2, 3... (滚动验证模式)
    """
    if fold_index is None:
        # 传统模式：加载 X_train.npy
        file_check = os.path.join(PROCESSED_DATA_PATH, 'X_train.npy')
        if not os.path.exists(file_check):
            raise FileNotFoundError("未找到传统数据文件，请先运行数据准备。")
        return preparer.load_processed_data() # 调用原有无参版本
    else:
        # 滚动验证模式：加载 X_train_fold_1.npy 等
        file_check = os.path.join(PROCESSED_DATA_PATH, f'X_train_fold_{fold_index}.npy')
        if not os.path.exists(file_check):
            raise FileNotFoundError(f"未找到 Fold {fold_index} 的数据文件。")
        return preparer.load_processed_data(fold_index=fold_index) # 调用带参版本

def train_model(X_train, X_val, X_test, y_train, y_val, y_test, model_type='unet',fold_index=None, scalers=None):
    """训练模型"""
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    # 选择模型
    if model_type.lower() == 'cnn':
        model = SimpleCNN()
        print("\n使用 SimpleCNN 模型")
    elif model_type.lower() == 'unet':
        model = UNet()
        print("\n使用 UNet 模型")
    elif model_type.lower() == 'fnn':
        model = FNN()
        print("\n使用 FNN 模型")
    elif model_type.lower() == 'mlp':
        model = MLP()
        print("\n使用 MLP 模型")
    elif model_type.lower() == 'convlstm':
        model = ConvLSTM()
        print("\n使用 convLSTM 模型")
    elif model_type.lower() == 'swin_transformer':
        model = SwinTransformerModel()
        print("\n使用 Swin Transformer 模型")
    else:
        # 默认使用UNet
        model = UNet()
        print(f"\n未知模型类型 '{model_type}'，默认使用 UNet 模型")

    fold_suffix = f"_fold_{fold_index}" if fold_index is not None else ""
    # 创建训练器并训练
    # trainer = Trainer(model, train_loader, val_loader, DEVICE)
    trainer = Trainer(model, train_loader, val_loader, DEVICE, fold_suffix=fold_suffix)
    result_data = trainer.train()
    trainer.best_metric = trainer.best_val_loss
    return trainer


def plot_all_folds_comparison(all_train_curves, all_val_curves, model_name, save_path):
    """
    【新增功能】将所有 Fold 的损失曲线画在一张图上
    """
    if not all_train_curves or not all_val_curves:
        return

    plt.figure(figsize=(12, 8))
    # 定义颜色循环
    colors = plt.cm.tab10.colors

    max_len = max([len(v) for v in all_val_curves])

    for i, (t_loss, v_loss) in enumerate(zip(all_train_curves, all_val_curves)):
        label_name = f'Fold {i + 1}'
        color = colors[i % len(colors)]

        # 画验证集 (实线，粗一点)
        plt.plot(v_loss, label=f'{label_name} (Val)', color=color, linestyle='-', linewidth=2)
        # 画训练集 (虚线，细一点，淡一点)
        plt.plot(t_loss, color=color, linestyle='--', linewidth=1, alpha=0.4)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title(f'{model_name.upper()} Cross-Validation Loss Comparison (All Folds)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已生成对比总图: {save_path}")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Weather Forecasting System')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['prepare', 'train', 'full'],
                        help='运行模式: prepare(仅准备数据), train(仅训练), full(完整流程)')
    parser.add_argument('--model', type=str, default='unet',
                        choices=['cnn', 'unet','fnn','mlp','convlstm', 'swin_transformer'],
                        help='模型类型: cnn, unet, fnn, mlp, convlstm, swin_transformer')
    parser.add_argument('--force', action='store_true',
                        help='强制重新处理数据')

    args = parser.parse_args()

    # 设置环境
    setup_environment()
    preparer = WeatherDataPreparer()

    is_cv_mode = USE_ROLLING_CV and len(CV_VAL_YEARS) > 0

    if is_cv_mode:
        total_folds = len(CV_VAL_YEARS)
        fold_list = list(range(1, total_folds + 1))
        print(f"\n[自动检测] 检测到滚动交叉验证配置 (CV_VAL_YEARS={CV_VAL_YEARS})")
        print(f"   将自动处理 {total_folds} 个 Folds.")
    else:
        fold_list = []  # 传统模式不需要列表
        print(f"\n[自动检测] 未启用滚动交叉验证，运行传统单套数据模式。")

    if args.mode == 'prepare':
        print("\n>>> 开始数据准备...")

        if is_cv_mode:
            print(f"   [CV 模式] 正在生成 {total_folds} 组 Fold 数据文件...")
            # 调用你的滚动管道，生成 x_train_fold_1.npy 等
            preparer.run_rolling_cv_pipeline()
            print("   ✅ 所有 Fold 数据已保存！")
        else:
            print("   [传统模式] 正在生成单套数据文件...")
            # 调用原有逻辑，生成 x_train.npy 等
            prepare_data(force=args.force)
            print("   ✅ 传统数据已保存！")

        # =========================================================
        # 模式 2: train (仅训练模型)
        # =========================================================
    elif args.mode == 'train':
        print("\n>>> 开始训练模型...")

        if is_cv_mode:
            all_metrics = []
            all_train_losses = []
            all_val_losses = []
        for k in fold_list:
                print(f"\n{'-' * 40}")
                print(f"   >> 训练 Fold {k}/{total_folds} (验证年份: {CV_VAL_YEARS[k - 1]})")
                print(f"{'-' * 40}")

                try:
                    # 加载带后缀的文件: x_train_fold_k.npy
                    X_train, X_val, X_test, y_train, y_val, y_test = preparer.load_processed_data(fold_index=k)

                    trainer = train_model(
                        X_train, X_val, X_test,
                        y_train, y_val, y_test,
                        args.model,
                        fold_index=k,  # 告诉模型保存为 model_unet_fold_k.pth
                        scalers=preparer.scalers
                    )

                    all_train_losses.append(trainer.train_losses)
                    all_val_losses.append(trainer.val_losses)

                    if k == 1:
                        model_name = trainer.model.__class__.__name__

                    if hasattr(trainer, 'best_metric'):
                        all_metrics.append(trainer.best_metric)
                        print(f"   -> Fold {k} 完成，Loss: {trainer.best_metric:.4f}")
                    else:
                        print(f"   -> Fold {k} 训练完成。")

                except Exception as e:

                    print(f"\n Fold {k} 训练失败：{e}")

                    import traceback

                    traceback.print_exc()

                    break

                if len(all_val_losses) == len(fold_list):
                    print("\n" + "=" * 50)
                    print("🎉 所有 Fold 训练完毕，正在生成对比总图...")
                    save_path = os.path.join(FIGURE_SAVE_PATH, f'cv_{args.model}_summary.png')

                    # 调用绘图函数
                    plot_all_folds_comparison(all_train_losses, all_val_losses, model_name, save_path)

                    print(f"平均指标：{np.mean(all_metrics):.4f}")
                    print("=" * 50)

                else:
                    # 传统模式 (保持不变)
                    print("   [传统模式] 加载单套数据...")
                    try:
                        X_train, X_val, X_test, y_train, y_val, y_test = preparer.load_processed_data(fold_index=None)
                        train_model(X_train, X_val, X_test, y_train, y_val, y_test, args.model, fold_index=None,
                                    scalers=preparer.scalers)
                        print("   ✅ 传统模式训练完成！")
                    except FileNotFoundError:
                        print("\n❌ 错误：未找到数据文件。")
        # =========================================================
        # 模式 3: full (完整流程)
        # =========================================================
    else:  # args.mode == 'full'
        print("\n" + "=" * 50)
        print("运行完整流程")
        print("=" * 50)

        if is_cv_mode:
            # --- CV 完整流程 ---
            print("\n[步骤 1/2] 准备所有 Fold 数据...")
            preparer.run_rolling_cv_pipeline()

            print("\n[步骤 2/2] 开始循环训练...")
            all_metrics = []
            all_train_losses = []
            all_val_losses = []
            for k in fold_list:
                print(f"\n   >> 训练 Fold {k}...")
                try:
                    X_train, X_val, X_test, y_train, y_val, y_test = preparer.load_processed_data(fold_index=k)

                    trainer = train_model(X_train, X_val, X_test, y_train, y_val, y_test, args.model, fold_index=k,
                                          scalers=preparer.scalers)

                    # ✅ 收集数据
                    all_train_losses.append(trainer.train_losses)
                    all_val_losses.append(trainer.val_losses)
                    if k == 1: model_name = trainer.model.__class__.__name__

                    if hasattr(trainer, 'best_metric'):
                        all_metrics.append(trainer.best_metric)
                except Exception as e:
                    print(f"   ❌ Fold {k} 失败：{e}")
                    break

                # ✅ 循环结束后，只画这一张总图
            if len(all_val_losses) == len(fold_list):
                print("\n" + "=" * 50)
                print("🎉 完整流程结束，正在生成对比总图...")
                save_path = os.path.join(FIGURE_SAVE_PATH, f'cv_{args.model}_full_summary.png')
                plot_all_folds_comparison(all_train_losses, all_val_losses, model_name, save_path)
                print(f"平均指标：{np.mean(all_metrics):.4f}")
                print("=" * 50)
            else:
                # 传统模式 (保持不变)
                print("\n[步骤 1/2] 准备传统数据...")
                X_train, X_val, X_test, y_train, y_val, y_test, scalers = prepare_data(force=args.force)
                print("\n[步骤 2/2] 训练模型...")
                train_model(X_train, X_val, X_test, y_train, y_val, y_test, args.model, fold_index=None,
                            scalers=scalers)
                print("\n✅ 完整流程 (传统) 完成！")



            print("\n" + "=" * 50)
            print("完整流程 (传统) 完成！")
            print(f"模型保存位置：{MODEL_SAVE_PATH}")
            print(f"可视化结果保存位置：{FIGURE_SAVE_PATH}")
            print("=" * 50)
    # if args.mode == 'prepare':
    #     # 仅准备数据
    #     prepare_data(force=args.force)
    #     print("\n数据准备完成！")
    #
    # elif args.mode == 'train':
    #     # 仅训练模型（假设数据已准备好）
    #     try:
    #     #     preparer = WeatherDataPreparer()
    #     #     X_train, X_val, X_test, y_train, y_val, y_test = preparer.load_processed_data()
    #     #     train_model(X_train, X_val, X_test, y_train, y_val, y_test, args.model)
    #     # except FileNotFoundError:
    #     #     print("\n错误: 未找到处理后的数据文件。请先运行 'python main.py --mode prepare'")
    #
    #
    # else:  # full mode
    #     # 完整流程
    #     print("\n" + "=" * 50)
    #     print("运行完整流程")
    #     print("=" * 50)
    #
    #     # 1. 准备数据
    #     X_train, X_val, X_test, y_train, y_val, y_test, scalers = prepare_data(force=args.force)
    #
    #     # 2. 训练模型
    #     trainer = train_model(X_train, X_val, X_test, y_train, y_val, y_test, args.model)

        print("\n" + "=" * 50)
        print("完整流程完成！")
        print(f"模型保存位置: {MODEL_SAVE_PATH}")
        print(f"可视化结果保存位置: {FIGURE_SAVE_PATH}")
        print("=" * 50)


if __name__ == "__main__":
    main()
    """
    ==================================================
划分数据集...
总样本数: 2307
切分点: 训练截止@1753, 验证截止@1937

划分完成:

==================================================
数据归一化（只使用训练集统计）...
==================================================
正在拟合标准化器（仅使用训练集）...
输入变量 - 均值范围: [-0.198, 287.183]
输入变量 - 标准差范围: [2.915, 15.832]
目标变量 - 均值: -0.197
目标变量 - 标准差: 2.916

正在转换数据集...

归一化完成！
训练集: X [-7.522, 8.135]
验证集: X [-6.378, 7.758]
测试集: X [-7.403, 6.930]

==================================================
保存处理后的数据...
数据已保存到: ./processed_data_v_wind/

==================================================
数据处理流程完成！
==================================================

最终数据形状:
X_train: (1753, 7, 54, 42, 3)
y_train: (1753, 1, 54, 42)
X_val: (184, 7, 54, 42, 3)
y_val: (184, 1, 54, 42)
X_test: (370, 7, 54, 42, 3)
y_test: (370, 1, 54, 42)
    """