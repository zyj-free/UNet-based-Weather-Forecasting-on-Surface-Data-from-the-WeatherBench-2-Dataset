"""
CNN和UNet模型定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_CHANNELS, INPUT_FRAMES, PRED_FRAMES,H,W


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
                 output_frames=PRED_FRAMES,dropout=0.1):
        super(UNet, self).__init__()

        self.input_frames = input_frames
        self.input_channels = input_channels
        self.output_frames = output_frames
        self.dropout = dropout

        # 编码器
        # self.enc1 = self.conv_block(input_channels * input_frames, 64)
        # self.enc2 = self.conv_block(64, 128)
        # self.enc3 = self.conv_block(128, 256)
        # self.enc4 = self.conv_block(256, 512)
        # 编码器（不加 Dropout，保证信息流畅通）
        self.enc1 = self.conv_block(input_channels * input_frames, 64, dropout=0)
        self.enc2 = self.conv_block(64, 128, dropout=0)
        self.enc3 = self.conv_block(128, 256, dropout=0)
        self.enc4 = self.conv_block(256, 512, dropout=0)

        # 池化
        self.pool = nn.MaxPool2d(2)

        # 瓶颈层
        # self.bottleneck = self.conv_block(512, 1024)
        self.bottleneck = self.conv_block(512, 1024, dropout=0)
        # # 解码器
        # self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        # self.dec4 = self.conv_block(1024, 512)
        #
        # self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        # self.dec3 = self.conv_block(512, 256)
        #
        # self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        # self.dec2 = self.conv_block(256, 128)
        #
        # self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        # self.dec1 = self.conv_block(128, 64)
        # 解码器（加 Dropout，防止过拟合）
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64, dropout=dropout)

        # 输出层
        self.output = nn.Conv2d(64, output_frames, 1)

    def conv_block(self, in_channels, out_channels,dropout=0):
        """卷积块"""
        # return nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, 3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(dropout),  # 添加Dropout
        #     nn.Conv2d(out_channels, out_channels, 3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(dropout)  # 添加Dropout
        # )
        if dropout > 0:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            )
        else:
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
        # print("模型输入 x 形状:", x.shape)

        # batch, frames, H, W, channels = x.shape

        # batch, lon, lat, channels, time_steps = x.shape
        # # 重塑
        # # x = x.permute(0, 1, 4, 2, 3).reshape(batch, frames * channels, H, W)
        # x = x.permute(0, 3, 4, 2, 1).reshape(batch, channels * time_steps, lat, lon)
        # original_h, original_w = lat, lon

        batch, frames, H, W, channels = x.shape

        # 重塑：(N, T*C, H, W)
        x = x.permute(0, 1, 4, 2, 3).reshape(batch, frames * channels, H, W)

        # 记录原始尺寸
        original_h, original_w = H, W
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

        out = out[:, :, :original_h, :original_w]
        # 步骤 A: 先交换空间维度 (Lat, Lon) -> (Lon, Lat)
        #         (8, 1, 80, 102) -> (8, 1, 102, 80)
        # out = out.permute(0, 1, 3, 2).contiguous()

        # 步骤 B: 再移动通道维度到最后一位 (B, C, H, W) -> (B, H, W, C)
        #         (8, 1, 102, 80) -> (8, 102, 80, 1)
        # out = out.permute(0, 2, 3, 1).contiguous()

        # 裁剪回原始输入尺寸 (去掉为了凑偶数而 Pad 的部分)
        return out


# ==============================================================================
# 1. FNN / MLP (简单基线)
# 特点：忽略空间局部性，将图像展平为向量。仅作为性能下限参考。
# ==============================================================================
class FNN(nn.Module):
    """
    前馈神经网络 (FNN) - 最简单的基线模型
    将输入展平后通过全连接层直接预测输出
    INPUT_FRAMES, PRED_FRAMES

    FNN：将整个输入展平成一个长向量，需要知道确切的输入维度

    输入维度 = INPUT_FRAMES × INPUT_CHANNELS × H × W
    输出维度 = PRED_FRAMES × H × W
    """

    def __init__(self, input_channels=INPUT_CHANNELS, input_frames=INPUT_FRAMES,
                 output_frames=PRED_FRAMES, hidden_dims=[512, 1024, 512]):
        super(FNN, self).__init__()

        self.input_frames = input_frames
        self.input_channels = input_channels
        self.output_frames = output_frames
        self.h, self.w = H, W

        # 输入维度
        input_dim = input_frames * input_channels * H * W
        output_dim = output_frames * H * W  # 输出单通道

        # 构建全连接层
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, frames, H, W, channels)
        batch_size = x.shape[0]

        # 展平所有维度除了batch
        x = x.reshape(batch_size, -1)

        # 前向传播
        x = self.network(x)

        # 重塑为输出格式 (batch, pred_frames, H, W)
        x = x.reshape(batch_size, self.output_frames, self.h, self.w)

        return x


class MLP(nn.Module):
    """
    多层感知机 (MLP) - 修正版
    输入: (Batch, Time, H, W, Channels)
    输出: (Batch, Pred_Time, H, W)
    """

    def __init__(self, input_channels=INPUT_CHANNELS, input_frames=INPUT_FRAMES,
                 output_frames=PRED_FRAMES, hidden_dim=128):
        super(MLP, self).__init__()

        self.input_frames = input_frames
        self.input_channels = input_channels
        self.output_frames = output_frames
        self.h, self.w = H, W

        # 每个位置的输入特征维度
        input_dim = input_frames * input_channels

        print(f"[*] MLP 初始化:")
        print(f"    - 输入帧数: {input_frames}")
        print(f"    - 输入通道: {input_channels}")
        print(f"    - 输入特征维度 (T*C): {input_dim}")
        print(f"    - 隐藏层维度: {hidden_dim}")
        print(f"    - 输出帧数: {output_frames}")
        # 共享的MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_frames)  # 输出预测帧数
        )

    # def forward(self, x):
    #     # x shape: (batch, frames, H, W, channels)
    #     batch_size = x.shape[0]
    #
    #     # 合并frames和channels维度
    #     x = x.reshape(batch_size, self.input_frames * self.input_channels, self.h, self.w)
    #
    #     # 重排为 (batch, H*W, features)
    #     x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.input_frames * self.input_channels)
    #
    #     # 对每个位置应用MLP
    #     x = self.mlp(x)  # (batch, H*W, output_frames)
    #
    #     # 重塑回空间格式
    #     x = x.reshape(batch_size, self.h, self.w, self.output_frames)
    #     x = x.permute(0, 3, 1, 2)  # (batch, output_frames, H, W)
    #
    #     return x

    def forward(self, x):
        """
        输入 x: (Batch, Channels, Time, Height, Width)
               例如：(2, 7, 80, 102, 3)
        """
        batch_size = x.shape[0]

        B, C, T, H, W = x.shape
        H = x.shape[2]
        W = x.shape[3]

        # 步骤 1: 重排维度 (B, T, H, W, C) -> (B, H, W, T, C)
        # 将空间维度移到前面，方便后续展平
        x = x.permute(0, 2, 3, 1, 4).contiguous()
        x = x.view(batch_size, H * W, self.input_frames * self.input_channels)

        x = self.mlp(x)
        x = x.view(batch_size, H, W, self.output_frames)
        x = x.permute(0, 3, 1, 2)

        return x

# ==============================================================================
# 2. ConvLSTM (时空序列基准)
# 特点：显式建模时间依赖性，结合卷积提取空间特征。
# ==============================================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
                参数:
                    input_dim: 输入通道数
                    hidden_dim: 隐藏状态通道数
                    kernel_size: 卷积核大小
                    bias: 是否使用偏置
        """
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 【修复点 1】强制确保 kernel_size 是元组 (h, w)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        """
                前向传播
                参数:
                    x: 当前时间步的输入 [batch, input_dim, height, width]
                    cur_state: 当前状态 (h, c)
                               h: [batch, hidden_dim, height, width]
                               c: [batch, hidden_dim, height, width]
                返回:
                    h_next: 下一个隐藏状态
                    c_next: 下一个细胞状态
                """
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # 激活函数
        i = torch.sigmoid(cc_i)  # 输入门
        f = torch.sigmoid(cc_f)  # 遗忘门
        o = torch.sigmoid(cc_o)  # 输出门
        g = torch.tanh(cc_g)  # 细胞更新

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        state = (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                 torch.zeros(batch_size, self.hidden_dim, height, width, device=device))
        return state


class ConvLSTM(nn.Module):
    def __init__(self, input_channels=INPUT_CHANNELS, input_frames=INPUT_FRAMES,
                 output_frames=PRED_FRAMES, hidden_dim=64, kernel_size=3, num_layers=2):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 输入投影层：将 (Channels * Frames) 映射到 hidden_dim
        # 注意：ConvLSTM通常逐帧输入。我们将输入视为序列。
        # 这里简化处理：先将所有帧通道合并，或者逐帧输入。
        # 策略：逐帧输入，输入通道为 input_channels
        # 确保hidden_dims长度匹配num_layers

        # 【修复点 2】这里接收的 kernel_size 可能是 int，传给 Cell 前会被 Cell 处理，
        # 但为了保险，我们在这里也可以预处理一下，或者依赖 Cell 内部的修复。
        # 关键在于 input_proj 的 kernel_size 也应该是元组如果它是整数的话
        kp = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        # 输入投影：将 input_channels 映射到 hidden_dim
        # 注意：输入数据在 forward 中会被转置为 (B, C, H, W)
        self.input_proj = nn.Conv2d(input_channels, hidden_dim, kernel_size=kp, padding=(kp[0] // 2, kp[1] // 2))

        # 构建多层 ConvLSTM Cells
        cells = []
        for i in range(num_layers):
            # 第一层输入维度是 hidden_dim (经过 input_proj 后)
            # 后续层输入维度也是 hidden_dim (上一层的输出)
            cells.append(ConvLSTMCell(input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=kernel_size))

        self.cells = nn.ModuleList(cells)

        # 输出投影：从 hidden_dim 映射到 output_frames
        self.output_proj = nn.Conv2d(hidden_dim, output_frames, kernel_size=1)
    def forward(self, x):
        # x shape: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        device = x.device

        # 重排为 (B, T, C, H, W) 以符合Conv习惯
        x = x.permute(0, 1, 4, 2, 3).contiguous()

        # 初始化状态
        states = []
        for i in range(self.num_layers):
            states.append(self.cells[i].init_hidden(B, H, W, device))

        last_h = None

        for t in range(T):
            input_t = x[:, t, :, :, :]  # (B, C, H, W)

            # 第一步：投影输入通道到 hidden_dim
            if t == 0:
                # 第一帧需要投影
                current_input = self.input_proj(input_t)
            else:
                # 后续帧同样需要投影 (权重共享)
                current_input = self.input_proj(input_t)

            # 逐层传递
            for i in range(self.num_layers):
                h, c = self.cells[i](current_input, states[i])
                states[i] = (h, c)
                # 当前层的输出 h 成为下一层的输入
                current_input = h

            # 记录最后一层的输出，如果需要序列输出可以在这里收集
            last_h = current_input

        # 使用最后一个时间步、最后一层的隐藏状态进行预测
        # 输出形状: (B, output_frames, H, W)
        out = self.output_proj(last_h)

        return out


# ==============================================================================
# 4. Swin Transformer (SwinUNETR 变体)
# 特点：基于Window Attention，擅长捕捉长距离依赖，气象大模型的主流 backbone。
# 注意：完整Swin很大，这里实现一个简化的 Swin Block 用于演示结构。
# 实际生产环境建议调用 timm 库或 monai.networks.nets.SwinUNETR
# ==============================================================================
class WindowAttention(nn.Module):
    """窗口注意力机制 (支持 Mask)"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        # x: (B * num_windows, N, C), N = window_size^2
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask shape: (num_windows, N, N) or (B*num_windows, N, N)
            nW = mask.shape[0]
            # 将 mask 扩展到 batch 维度
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (B * num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (B * num_windows, window_size, window_size, C)
        window_size: int
        H, W: original height and width
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 如果 shift_size > 0，通常设为 window_size // 2
        if self.shift_size > 0:
            self.shift_size = self.window_size // 2
        else:
            self.shift_size = 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # --- 1. Padding ---
        # 计算需要填充的大小，使 H, W 能被 window_size 整除
        pad_l = 0
        pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

        if pad_r > 0 or pad_b > 0:
            # Pad to (B, C, H+pad_b, W+pad_r) then permute back to (B, H', W', C)
            x = x.permute(0, 3, 1, 2)
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            x = x.permute(0, 2, 3, 1)

        Hp, Wp = x.shape[1], x.shape[2]

        # --- 2. Cyclic Shift ---
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # --- 3. Partition Windows ---
        x_windows = window_partition(shifted_x, self.window_size)
        # x_windows: (B * num_windows, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # --- 4. Attention (with Mask if shifted) ---
        attn_mask = None
        if self.shift_size > 0:
            # 生成 Mask (简化版逻辑，确保不同窗口不交互)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        attn_windows = self.attn(x_windows, mask=attn_mask)

        # --- 5. Reverse Windows ---
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # --- 6. Reverse Cyclic Shift ---
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # --- 7. Unpad ---
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # --- 8. Residual Connection ---
        # 此时 x 的形状严格为 (B, H, W, C)，与 shortcut  reshape 后一致
        x = x.view(B, H * W, C)
        x = shortcut + x

        # --- 9. MLP ---
        x = x + self.mlp(self.norm2(x))

        return x


class SwinTransformerModel(nn.Module):
    def __init__(self, input_channels=INPUT_CHANNELS, input_frames=INPUT_FRAMES,
                 output_frames=PRED_FRAMES, embed_dim=96, depths=[2, 2, 2],
                 num_heads=[3, 6, 12], window_size=7, mlp_ratio=4.):
        super().__init__()
        self.input_frames = input_frames
        self.input_channels = input_channels
        self.output_frames = output_frames
        self.embed_dim = embed_dim
        self.window_size = window_size

        # 1. Patch Embedding
        # 输入: (B, T*C, H, W) -> (B, Dim, H/4, W/4)
        # 注意：您的数据 H=54, W=42. PatchEmbed(stride=4) -> 13x10
        # self.patch_embed = nn.Conv2d(input_channels * input_frames, embed_dim, kernel_size=4, stride=4)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(input_channels * input_frames, embed_dim, kernel_size=3, padding=1),
            nn.LayerNorm([embed_dim, H, W])  # 如果需要
        )
        # 2. Layers (Swin Blocks)
        # 为了简化，这里不使用 PatchMerging (下采样)，保持分辨率不变，只增加深度
        # 这样避免了复杂的维度管理，适合中等分辨率气象预测
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            for _ in range(depths[i_layer]):
                # 所有层使用相同的 dim (因为没有 PatchMerging)
                # 如果未来加入 PatchMerging，dim 需要翻倍
                block = SwinBlock(
                    dim=embed_dim,
                    num_heads=num_heads[i_layer % len(num_heads)],
                    window_size=window_size,
                    shift_size=0 if (_ % 2 == 0) else window_size // 2,  # 交替 shift
                    mlp_ratio=mlp_ratio
                )
                self.layers.append(block)

        # 3. Decoder / Head
        # 当前特征图尺寸: H/4, W/4. 通道: embed_dim
        # 我们需要上采样回 H, W
        # 步骤: (H/4, W/4) -> (H/2, W/2) -> (H, W)

        self.up1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2)

        # 最终输出卷积
        self.head = nn.Conv2d(embed_dim // 4, output_frames, kernel_size=1)

        # 如果插值后尺寸仍有微小差异，用这个对齐
        self.final_interp = True

    def forward(self, x):
        B, T, H, W, C = x.shape
        original_H, original_W = H, W

        # 1. Merge T and C, Permute to (B, T*C, H, W)
        x = x.permute(0, 1, 4, 2, 3).reshape(B, T * C, H, W)

        # 2. Patch Embed
        x = self.patch_embed(x)  # (B, Dim, H', W')
        B, C_enc, H_enc, W_enc = x.shape

        # 3. Flatten for Transformer: (B, L, C)
        x = x.flatten(2).transpose(1, 2)

        # 4. Pass through Swin Blocks
        # 由于没有下采样，H_enc, W_enc 保持不变
        for block in self.layers:
            x = block(x, H_enc, W_enc)

        # 5. Reshape back to (B, C, H, W)
        x = x.transpose(1, 2).view(B, C_enc, H_enc, W_enc)

        # 6. Decode (Upsample)
        x = self.up1(x)
        x = self.up2(x)

        # 7. Output Projection
        out = self.head(x)

        # 8. Resize to original input size (H, W)
        if out.shape[-2:] != (original_H, original_W):
            out = F.interpolate(out, size=(original_H, original_W), mode='bilinear', align_corners=False)

        return out

if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建测试输入 (使用报错的尺寸 54x42 进行测试)
    batch_size = 2
    H, W = 54, 42  # 之前报错的尺寸

    test_input = torch.randn(batch_size, INPUT_FRAMES, H, W, INPUT_CHANNELS).to(device)

    print(f"Input shape: {test_input.shape}")

    # 测试所有模型
    models_to_test = [
        ('FNN', FNN),
        ('MLP', MLP),
        ('ConvLSTM', ConvLSTM),
        ('UNet', UNet),
        # ('Swin Transformer', SwinTransformer)
    ]

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