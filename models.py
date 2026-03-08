"""
CNN和UNet模型定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_CHANNELS, INPUT_FRAMES, PRED_FRAMES


class SimpleCNN(nn.Module):
    """简单的CNN模型"""

    def __init__(self, input_channels=INPUT_CHANNELS, input_frames=INPUT_FRAMES,
                 output_frames=PRED_FRAMES):
        super(SimpleCNN, self).__init__()

        self.input_frames = input_frames
        self.input_channels = input_channels
        self.output_frames = output_frames

        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels * input_frames, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # 解码器
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),  # 跳跃连接
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),  # 跳跃连接
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 输出层
        self.output = nn.Conv2d(64, output_frames, 3, padding=1)

    def forward(self, x):
        # x shape: (batch, frames, H, W, channels)
        batch, frames, H, W, channels = x.shape

        # 重塑：合并frames和channels维度
        x = x.permute(0, 1, 4, 2, 3).reshape(batch, frames * channels, H, W)

        # 编码
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)

        # 解码
        d2 = self.up2(e3)
        # 防止尺寸不匹配（虽然SimpleCNN只有两层，但也加上保护）
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # 输出
        out = self.output(d1)

        return out


class UNet(nn.Module):
    """UNet模型 (支持任意输入尺寸)"""

    def __init__(self, input_channels=INPUT_CHANNELS, input_frames=INPUT_FRAMES,
                 output_frames=PRED_FRAMES):
        super(UNet, self).__init__()

        self.input_frames = input_frames
        self.input_channels = input_channels
        self.output_frames = output_frames

        # 编码器
        self.enc1 = self.conv_block(input_channels * input_frames, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # 池化
        self.pool = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = self.conv_block(512, 1024)

        # 解码器
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # 输出层
        self.output = nn.Conv2d(64, output_frames, 1)

    def conv_block(self, in_channels, out_channels):
        """卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 记录原始尺寸，用于最后裁剪
        original_h, original_w = x.shape[2], x.shape[3]

        batch, frames, H, W, channels = x.shape

        # 重塑
        x = x.permute(0, 1, 4, 2, 3).reshape(batch, frames * channels, H, W)

        # --- 辅助函数：如果是奇数，Padding 成偶数 ---
        def make_even(tensor):
            h, w = tensor.shape[2], tensor.shape[3]
            pad_h = h % 2
            pad_w = w % 2
            if pad_h or pad_w:
                # (left, right, top, bottom)
                return F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
            return tensor

        # --- 辅助函数：确保解码器特征与编码器特征尺寸一致 ---
        def match_size(dec_feat, enc_feat):
            if dec_feat.shape[2:] != enc_feat.shape[2:]:
                return F.interpolate(dec_feat, size=enc_feat.shape[2:], mode='bilinear', align_corners=False)
            return dec_feat

        # === 编码器 (Encoder) ===
        # 处理输入
        x = make_even(x)
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        # 处理第2层
        p1 = make_even(p1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        # 处理第3层
        p2 = make_even(p2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        # 处理第4层
        p3 = make_even(p3)
        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        # 处理瓶颈层输入
        p4 = make_even(p4)
        b = self.bottleneck(p4)

        # === 解码器 (Decoder) ===
        d4 = self.up4(b)
        d4 = match_size(d4, e4)  # 【关键】强制对齐尺寸
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = match_size(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = match_size(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = match_size(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.output(d1)

        # 裁剪回原始输入尺寸 (去掉为了凑偶数而 Pad 的部分)
        return out[:, :, :original_h, :original_w]


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建测试输入 (使用报错的尺寸 54x42 进行测试)
    batch_size = 2
    H, W = 54, 42  # 之前报错的尺寸

    test_input = torch.randn(batch_size, INPUT_FRAMES, H, W, INPUT_CHANNELS).to(device)

    # 测试CNN模型
    cnn_model = SimpleCNN().to(device)
    try:
        cnn_output = cnn_model(test_input)
        print(f"CNN模型输出形状: {cnn_output.shape}")
    except Exception as e:
        print(f"CNN模型报错: {e}")

    # 测试UNet模型
    unet_model = UNet().to(device)
    try:
        unet_output = unet_model(test_input)
        print(f"\nUNet模型输出形状: {unet_output.shape}")
        print(f"UNet模型参数数: {sum(p.numel() for p in unet_model.parameters()):,}")
        print("UNet 成功处理了 54x42 的输入！")
    except Exception as e:
        print(f"\nUNet模型报错: {e}")