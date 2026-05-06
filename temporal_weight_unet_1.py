"""
最小化时间加权UNet - 快速验证时间权重的有效性
在原有UNet基础上，仅添加可学习的时间权重，参数增加最少
0403
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from models import UNet
from config import INPUT_CHANNELS, INPUT_FRAMES, PRED_FRAMES, OUTPUT_CHANNELS


@dataclass
class HardwareAwareQuantConfig:
    enabled: bool = False
    activation_bits: int = 8
    weight_bits: int = 8
    sensitive_ratio: float = 0.25
    sensitive_dtype: torch.dtype = torch.float16
    use_channels_last: bool = True
    calibration_momentum: float = 0.9
try:
    import mamba_ssm
    print("✅ mamba_ssm库已正确安装")
    # 检查关键组件
    from mamba_ssm import Mamba
    print(f"可用模型类: {Mamba.__name__}")
except ImportError as e:
    print(f"❌ 导入失败: {str(e)}")
    # 尝试定位问题
    import os
    print(f"当前路径: {os.getcwd()}")
    print(f"Python路径: {os.environ.get('PYTHONPATH', '未设置')}")

class MinimalTimeAwareUNet(nn.Module):
    """
    最小化时间加权UNet
    
    改进方向：为不同时间步赋予不同的权重
    参数增加：仅 INPUT_FRAMES 个权重参数（7个浮点数）
    预期改进：+3-5% RMSE 改善
    """
    
    def __init__(self, input_channels=INPUT_CHANNELS, input_frames=INPUT_FRAMES,
                 output_frames=PRED_FRAMES, output_channels=OUTPUT_CHANNELS, dropout=0.1):
        super(MinimalTimeAwareUNet, self).__init__()
        
        self.input_frames = input_frames
        self.input_channels = input_channels
        self.output_frames = output_frames
        self.output_channels = output_channels
        
        # 复用标准UNet
        self.backbone = UNet(
            input_channels=input_channels,
            input_frames=input_frames,
            output_frames=output_frames,
            output_channels=output_channels,
            dropout=dropout,
        )
        
        initial_weights = torch.linspace(1.0, 2.0, steps=input_frames)
        self.temporal_weights = nn.Parameter(initial_weights)
        
    def forward(self, x):
        """
        x: (batch, frames, H, W, channels)
        """
        batch, frames, H, W, channels = x.shape

        weights = F.softmax(self.temporal_weights, dim=0)  # shape: (frames,)

        x_weighted = x * weights.view(1, frames, 1, 1, 1)
        
        output = self.backbone(x_weighted)
        
        return output
    
    def get_temporal_weights(self):
        """返回学习到的时间权重（用于可视化/分析）"""
        weights = F.softmax(self.temporal_weights, dim=0)
        return weights.detach().cpu().numpy()


class ChannelTimeAwareUNet(nn.Module):
    def __init__(self, input_channels=INPUT_CHANNELS, input_frames=INPUT_FRAMES,
                 output_frames=PRED_FRAMES, output_channels=OUTPUT_CHANNELS, dropout=0.1):
        super(ChannelTimeAwareUNet, self).__init__()
        
        self.input_frames = input_frames
        self.input_channels = input_channels
        self.output_frames = output_frames
        self.output_channels = output_channels
        
        # 复用标准UNet
        self.backbone = UNet(
            input_channels=input_channels,
            input_frames=input_frames,
            output_frames=output_frames,
            output_channels=output_channels,
            dropout=dropout,
        )

        initial_weights = torch.linspace(1.0, 2.0, steps=input_frames)
        self.temporal_weights = nn.Parameter(initial_weights.unsqueeze(0).repeat(input_channels, 1))

    def forward(self, x):
        """
        x: (batch, frames, H, W, channels)
        """
        batch, frames, H, W, channels = x.shape
        
        weights = F.softmax(self.temporal_weights, dim=1)  # shape: (channels, frames)
        
        x = x.permute(0, 4, 1, 2, 3)
        

        x_weighted = x * weights.view(1, channels, frames, 1, 1)
        x_weighted = x_weighted.permute(0, 2, 3, 4, 1)  # (batch, frames, H, W, channels)
        
        output = self.backbone(x_weighted)
        
        return output
    
    def get_temporal_weights(self):
        """返回学习到的时间权重"""
        weights = F.softmax(self.temporal_weights, dim=1)
        return weights.detach().cpu().numpy()
    


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：大幅减少参数量"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        self.norm = nn.GroupNorm(1, out_ch) # 加入归一化稳定训练
        self.act = nn.SiLU()
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x
class MambaBlock(nn.Module):
    """
    Mamba 时间特征提取块
    输入: (B, T, H*W, C) -> 展平空间维度以适应序列模型
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.quant_config = HardwareAwareQuantConfig()
        self.register_buffer("activation_scale_ema", torch.tensor(1.0), persistent=False)

        try:
            from mamba_ssm import Mamba
            self.HAS_MAMBA = True
        except ImportError:
            self.HAS_MAMBA = False

        if self.HAS_MAMBA:
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            # 模拟层：仅用于代码不报错，无实际 Mamba 效果
            # self.mamba = nn.Linear(dim, dim)
            # 模拟层：使用LSTM或GRU作为替代，或者简单的线性层
            self.mamba = nn.GRU(dim, dim, batch_first=True, bidirectional=False)
            
            
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def enable_state_aware_quantization(
        self,
        activation_bits=8,
        weight_bits=8,
        sensitive_ratio=0.25,
        sensitive_dtype=torch.float16,
        calibration_momentum=0.9,
    ):
        self.quant_config = HardwareAwareQuantConfig(
            enabled=True,
            activation_bits=activation_bits,
            weight_bits=weight_bits,
            sensitive_ratio=sensitive_ratio,
            sensitive_dtype=sensitive_dtype,
            use_channels_last=True,
            calibration_momentum=calibration_momentum,
        )
        return self

    @staticmethod
    def _fake_quantize_tensor(x, num_bits=8, dim=None):
        if num_bits is None or num_bits >= 16:
            return x
        qmax = float(2 ** (num_bits - 1) - 1)
        if dim is None:
            scale = x.detach().abs().amax().clamp_min(1e-6) / qmax
        else:
            scale = x.detach().abs().amax(dim=dim, keepdim=True).clamp_min(1e-6) / qmax
        return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale

    def _state_aware_activation_quant(self, x):
        if not self.quant_config.enabled:
            return x

        temporal_diff = x[:, 1:] - x[:, :-1] if x.shape[1] > 1 else x
        temporal_score = temporal_diff.abs().mean(dim=(0, 1, 2))
        channel_score = x.abs().mean(dim=(0, 1, 2))
        importance = temporal_score + 0.5 * channel_score

        sensitive_mask = torch.zeros(self.dim, device=x.device, dtype=torch.bool)
        keep_channels = int(self.dim * self.quant_config.sensitive_ratio)
        if keep_channels > 0:
            topk = torch.topk(importance, k=keep_channels, largest=True).indices
            sensitive_mask[topk] = True

        scale_now = x.detach().abs().mean()
        self.activation_scale_ema.mul_(self.quant_config.calibration_momentum).add_(
            scale_now * (1.0 - self.quant_config.calibration_momentum)
        )
        base_scale = self.activation_scale_ema.clamp_min(1e-6)

        quantized = self._fake_quantize_tensor(
            x / base_scale,
            num_bits=self.quant_config.activation_bits,
            dim=(0, 1, 2),
        ) * base_scale

        output = quantized
        if sensitive_mask.any():
            output = output.clone()
            output[..., sensitive_mask] = x[..., sensitive_mask].to(self.quant_config.sensitive_dtype).to(x.dtype)
        return output

    def quantize_internal_weights(self):
        if not self.quant_config.enabled:
            return self
        for param in self.mamba.parameters():
            if param.ndim >= 2:
                param.data = self._fake_quantize_tensor(param.data, num_bits=self.quant_config.weight_bits)
        return self

    def forward(self, x):
        # x: (B, T, L, C) where L = H*W
        B, T, L, C = x.shape
        
        # 归一化
        x = self.norm(x)
        x = self._state_aware_activation_quant(x)
        
        # 重塑为 (B*L, T, C) 以便 Mamba 处理时间序列
        x = x.reshape(B * L, T, C)
        
        # Mamba 处理
        x = self.mamba(x)
        if isinstance(x, tuple):
            x = x[0]
        
        # 恢复形状
        x = x.reshape(B, T, L, C)
        return x
# class FastDepthwiseSeparableConv(nn.Module):
#     """更快、更部署友好、不掉精度"""
#     def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
#         super().__init__()
#         self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch)
#         self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
#         self.norm = nn.GroupNorm(1, out_ch)
#         self.act = nn.SiLU()

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.norm(x)
#         return self.act(x)
# ==================================================================
# 时空混合块 (Spatial-Temporal Block)
# 这才是真正融合了 Mamba 和 CNN 的轻量化单元
# ==================================================================
class STBlock(nn.Module):
    """时空特征提取块
    1. 先用 2D CNN 提取空间特征
    2. 再用 Mamba 提取时间特征
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 1. 轻量化空间提取
        self.spatial = DepthwiseSeparableConv(in_ch, out_ch)
        # 2. 时间提取 (Mamba)
        self.temporal = MambaBlock(dim=out_ch) if out_ch > 0 else None # Bottleneck 可能不需要

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B * T, H, W, C).permute(0, 3, 1, 2).contiguous() # (B*T, C, H, W)
        x = self.spatial(x) # (B*T, out_ch, H, W)
        x = x.view(B, T, -1, H, W).permute(0, 1, 3, 4, 2).contiguous() # (B, T, H, W, out_ch)
        
        # --- 时间混合 ---
        if self.temporal is not None:
            # 将空间维度 H*W 展平作为 Sequence 的 Length
            _, T_, H_, W_, C_ = x.shape
            x = x.view(B, T_, H_ * W_, C_) # (B, T, H*W, C_)
            x = self.temporal(x) # (B, T, H*W, C_)
            # 恢复空间形状
            x = x.view(B, T_, H_, W_, C_)
        
        return x
# ================= 1. 轻量化 UNet (基准) =================

class LightweightUNet(nn.Module):
    """
    1. 轻量化 UNet
    - 使用深度可分离卷积
    - 使用 Mamba 处理时间维度
    """
    def __init__(self, input_channels, output_channels,input_frames=7, hidden_dim=32):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        self.input_frames = input_frames
        
        # --- 编码器 ---
        # 空间投影
        self.enc_proj = nn.Conv2d(input_channels, hidden_dim, 1)
        
        self.down = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=2, stride=2)
        # Mamba 时间混合器
        # self.mamba_block = MambaBlock(dim=hidden_dim)
        self.bottleneck = STBlock(hidden_dim*2, hidden_dim*2)

        self.up = nn.ConvTranspose2d(hidden_dim*2, hidden_dim, 2, 2)
        
        # 解码卷积 (输入是 上采样+跳跃连接 = hidden_dim + hidden_dim)
        self.dec_conv = DepthwiseSeparableConv(hidden_dim*2, hidden_dim)
        
        # 输出头
        self.out_conv = nn.Conv2d(hidden_dim, output_channels, 1)

    def forward(self, x):
        B, T, H, W, C = x.shape
        hidden_dim = self.hidden_dim
        x_flat = x.view(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
        x_proj = self.enc_proj(x_flat) # (B*T, hidden_dim, H, W)
        x_down = self.down(x_proj) # (B*T, hidden_dim*2, H/2, W/2)
        H_down = x_down.shape[2]
        W_down = x_down.shape[3]
        x_btl = x_down.view(B, T, -1, H_down, W_down).permute(0, 1, 3, 4, 2).contiguous()
        x_btl = self.bottleneck(x_btl) # (B, T, H/2, W/2, hidden_dim*2)
        x_up_in = x_btl.view(B * T, H_down, W_down, -1).permute(0, 3, 1, 2).contiguous()
        x_up = self.up(x_up_in) # (B*T, hidden_dim, H, W)
        
        x_cat = torch.cat([x_up, x_proj], dim=1) # (B*T, hidden_dim*2, H, W)
        x_dec = self.dec_conv(x_cat) # (B*T, hidden_dim, H, W)
        out = self.out_conv(x_dec) # (B*T, output_channels, H, W)

        out = out.view(B, T, self.output_channels, H, W).permute(0, 1, 3, 4, 2).contiguous()

        out = out[:, -1:, :, :, :] # (B, 1, H, W, output_channels)
        
        return out
class EnhancedHistoryWeightEstimator(nn.Module):
    """
    增强版历史权重估计网络
    改进点：
    1. 使用 Global Average Pooling 提取全局时空特征，比简单的 Conv3d 更稳定。
    2. 增加了 MLP 层，增强非线性拟合能力。
    3. 引入 Dropout 防止过拟合。
    """
    def __init__(self, input_channels, input_frames, output_dim='shared'):
        super().__init__()
        self.input_frames = input_frames
        self.output_dim = output_dim
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        
        self.global_pool = nn.AdaptiveAvgPool3d((input_frames, 1, 1))
        
        out_features = 1 if output_dim == 'shared' else input_channels
        
        self.weight_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, out_features)
        )
        
    def forward(self, x):
        B, T, H, W, C = x.shape
        
        x = x.permute(0, 4, 1, 2, 3).contiguous()
 
        feats = self.feature_extractor(x) # (B, 64, T, H, W)

        feats = self.global_pool(feats) # (B, 64, T, 1, 1)

        feats = feats.squeeze(-1).squeeze(-1).permute(0, 2, 1) # (B, T, 64)
        weights = self.weight_head(feats) # (B, T, 1) 或 (B, T, C)
    

        if self.output_dim == 'shared':
            weights = weights.squeeze(-1) 
        else:
            weights = weights.permute(0,2,1) 

        weights = F.softmax(weights, dim=-1)
        
        return weights

class LightweightSharedWeightUNet(LightweightUNet):
    """
    2. 轻量化共享 UNet
    - 继承自 LightweightUNet
    - 增加一个共享的可学习时间权重参数 (所有通道共用)
    """
    def __init__(self, input_channels, output_channels, input_frames=7, hidden_dim=32):
        super().__init__(input_channels, output_channels, input_frames=input_frames, hidden_dim=hidden_dim)

        self.register_parameter(
            "shared_weights", 
            nn.Parameter(torch.ones(1, input_frames)) # (1, T)
        )
    def forward(self, x):
        B, T, H, W, C = x.shape
        
        # 应用共享权重
        weights = F.softmax(self.shared_weights, dim=-1) # (1, T)
        w = weights.view(1, T, 1, 1, 1)
        x = x * w
        
        return super().forward(x)

# ================= 3. 轻量化独立权重 UNet (权重是可学习变量) =================

class LightweightIndependentWeightUNet(LightweightUNet):
    """
    3. 轻量化独立权重 UNet
    - 继承自 LightweightUNet
    - 每个通道拥有独立的可学习时间权重
    """
    def __init__(self, input_channels, output_channels, input_frames=7, hidden_dim=32):
        super().__init__(input_channels, output_channels, input_frames=input_frames, hidden_dim=hidden_dim)
        
        self.register_parameter(
            "independent_weights", 
            nn.Parameter(torch.ones(input_channels, input_frames)) # (C, T)
        )
    def forward(self, x):
        B, T, H, W, C = x.shape
        
        # 1. 预测权重 (B, C, T)
        weights = F.softmax(self.independent_weights, dim=-1) # (C, T)
        
        # 2. 应用权重 (B, T, 1, 1, C)
        w = weights.permute(1, 0).view(1, T, 1, 1, C)
        x = x * w
        
        return super().forward(x)

class LightweightSharedWeightUNet_History(LightweightUNet):
    """
    轻量化共享UNet_History
    - 使用EnhancedHistoryWeightEstimator网络动态学习共享权重
    - 权重是网络预测的，不再是固定可学习参数
    """
    def __init__(self, input_channels, output_channels, input_frames=7, hidden_dim=32):
        super().__init__(input_channels, output_channels, input_frames=input_frames, hidden_dim=hidden_dim)

        self.weight_estimator = EnhancedHistoryWeightEstimator(
            input_channels, 
            input_frames, 
            output_dim='shared'
        )
    
    def forward(self, x):
        B, T, H, W, C = x.shape

        weights = self.weight_estimator(x)  # (B, T)

        weights = weights.view(B, T, 1, 1, 1) # (B, T, 1, 1, 1)
        x = x * weights+x

        
        # 3. 通过UNet处理
        return super().forward(x)
    
class LightweightIndependentWeightUNet_History(LightweightUNet):
    """
    轻量化独立权重UNet_History
    - 使用EnhancedHistoryWeightEstimator网络动态学习独立权重
    - 每个通道有独立的权重，由网络预测
    """
    def __init__(self, input_channels, output_channels, input_frames=7, hidden_dim=32):
        super().__init__(input_channels, output_channels, input_frames=input_frames, hidden_dim=hidden_dim)
        
        # ★ 核心：使用历史权重估计网络替代固定权重
        self.weight_estimator = EnhancedHistoryWeightEstimator(
            input_channels, 
            input_frames, 
            output_dim='independent'
        )
    
    def forward(self, x):
        B, T, H, W, C = x.shape
        weights = self.weight_estimator(x)  # (B, C, T)
        weights = weights.permute(0, 2, 1).view(B, T, 1, 1, C)
        x= x*weights+x
        return super().forward(x)

def prepare_model_for_hardware_deployment(
    model,
    activation_bits=8,
    weight_bits=8,
    sensitive_ratio=0.25,
    sensitive_dtype=torch.float16,
    use_channels_last=True,
    quantize_weights=True,
):
    model.eval()

    if use_channels_last:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                module.to(memory_format=torch.channels_last)

    for module in model.modules():
        if isinstance(module, MambaBlock):
            module.enable_state_aware_quantization(
                activation_bits=activation_bits,
                weight_bits=weight_bits,
                sensitive_ratio=sensitive_ratio,
                sensitive_dtype=sensitive_dtype,
            )
            if quantize_weights:
                module.quantize_internal_weights()

    return model


def build_hardware_deployment_variant(
    model_class,
    input_channels,
    output_channels,
    input_frames=7,
    hidden_dim=32,
    activation_bits=8,
    weight_bits=8,
    sensitive_ratio=0.25,
    sensitive_dtype=torch.float16,
    use_channels_last=True,
):
    model = model_class(
        input_channels=input_channels,
        output_channels=output_channels,
        input_frames=input_frames,
        hidden_dim=hidden_dim,
    )
    return prepare_model_for_hardware_deployment(
        model,
        activation_bits=activation_bits,
        weight_bits=weight_bits,
        sensitive_ratio=sensitive_ratio,
        sensitive_dtype=sensitive_dtype,
        use_channels_last=use_channels_last,
    )


if __name__ == "__main__":
    import torch
    import traceback
    # 测试模型输入输出维度是否正确
    # ================= 配置测试参数 =================
    BATCH_SIZE = 4
    INPUT_FRAMES = 7
    H, W = 80, 102
    INPUT_CHANNELS = 11
    OUTPUT_CHANNELS = 2
    HIDDEN_DIM = 32
    dummy_input = torch.randn(BATCH_SIZE, INPUT_FRAMES, H, W, INPUT_CHANNELS)
    
    expected_shape = (BATCH_SIZE, 1, H, W, OUTPUT_CHANNELS)

    print("=" * 70)
    print(f"🚀 开始模型输入输出形状测试")

    print("=" * 70)
    print(f"输入形状: {tuple(dummy_input.shape)}")
    print(f"期望输出: {expected_shape} (注意时间维度为1)")
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("-" * 70)

    models_to_test = [
        ("1. LightweightUNet (基准)", LightweightUNet),
        ("2. LightweightSharedWeightUNet", LightweightSharedWeightUNet),
        ("3. LightweightIndependentWeightUNet", LightweightIndependentWeightUNet),
        ("4. LightweightSharedWeightUNet_History", LightweightSharedWeightUNet_History),
        ("5. LightweightIndependentWeightUNet_History", LightweightIndependentWeightUNet_History),
    ]

    passed_count = 0
    failed_count = 0

    for model_name, model_class in models_to_test:
        try:
            model = model_class(
                input_channels=INPUT_CHANNELS, 
                output_channels=OUTPUT_CHANNELS, 
                input_frames=INPUT_FRAMES, 
                hidden_dim=HIDDEN_DIM
            )
            
            if torch.cuda.is_available():
                model.cuda()
                dummy_input_gpu = dummy_input.cuda()
            else:
                dummy_input_gpu = dummy_input

            model.eval()
            with torch.no_grad():
                output = model(dummy_input_gpu)
            
            output_shape = tuple(output.shape)
            
            if output_shape == expected_shape:
                status = "✅ 通过"
                passed_count += 1
            else:
                status = "⚠️ 形状不匹配"
                failed_count += 1
            
            print(f"{status} | {model_name}")
            print(f"      输出形状: {output_shape}")

        except Exception as e:
            failed_count += 1
            print(f"❌ 报错 | {model_name}")
            print(f"      错误信息: {str(e)}")
        
        print("-" * 70)

    print(f"📊 测试总结: {passed_count} 通过, {failed_count} 失败")
    if failed_count == 0:
        print("🎉 所有模型输入输出检查完成，未发现维度错误！")
    print("=" * 70)
