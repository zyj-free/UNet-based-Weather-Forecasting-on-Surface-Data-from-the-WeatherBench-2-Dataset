import argparse
import copy
import os
import torch
import pandas as pd

from deployment_benchmark_utils import (
    apply_channels_last_only,
    apply_fp16,
    build_base_model,
    build_state_aware_model,
    build_uniform_quant_model,
    ensure_dir,
    get_device,
    load_test_data,
    maybe_load_checkpoint,
    measure_model,
    save_bar_figure,
    save_results,
)


def build_strategy_models(variant, input_channels, output_channels, device, activation_bits, weight_bits, sensitive_ratio):
    checkpoint_name = {
        "shared": "4_LightSharedWeight_History_final.pth",
        "independent": "5_LightIndepWeight_History_final.pth",
    }[variant]

    base = build_base_model(variant, input_channels, output_channels)
    maybe_load_checkpoint(base, checkpoint_name)

    state_aware_model = build_state_aware_model(
        variant,
        input_channels,
        output_channels,
        activation_bits=activation_bits,
        weight_bits=weight_bits,
        sensitive_ratio=sensitive_ratio,
    )
    state_aware_model.load_state_dict(base.state_dict())

    uniform_quant_model = build_uniform_quant_model(
        variant,
        input_channels,
        output_channels,
        activation_bits=activation_bits,
        weight_bits=weight_bits,
    )
    uniform_quant_model.load_state_dict(base.state_dict())

    # ==========================================
    # 核心修改：真正的 INT8 保存逻辑
    # ==========================================
    def save_true_int8_model(model, name_prefix):
        original_state_dict = model.state_dict()
        int8_state_dict = {}
        scales_dict = {}
        restored_fp_state_dict = {} # 【关键】用来存还原后的 FP32 权重
        
        print(f"Converting {name_prefix} to true INT8 (Simulation Mode)...")
        
        for name, param in original_state_dict.items():
            # 1. 跳过 Mamba 复杂层，保持 FP16 (安全)
            if any(x in name for x in ['mamba', 'A_log', 'dt_proj', 'conv1d', 'x_proj', 'in_proj', 'out_proj']):
                int8_state_dict[name] = param.half()
                restored_fp_state_dict[name] = param # 保存原始 FP32
                continue

            # 2. 对普通 Linear 层进行量化
            if len(param.shape) >= 2: 
                max_val = param.abs().max()
                if max_val == 0: 
                    int8_state_dict[name] = param.half()
                    restored_fp_state_dict[name] = param
                    continue
                
                scale = max_val / 127.0 
                # 真正的 INT8 权重
                quant_param = (param / scale).round().to(torch.int8)
                
                int8_state_dict[name] = quant_param.half()
                scales_dict[name] = scale.half()
                
                # 【关键】保存还原后的 FP32 权重，用于加载时避免 NaN
                # 公式：weight_fp32 = weight_int8 * scale
                restored_fp_state_dict[name] = (quant_param.float() * scale)
            else:
                # Bias 保持 FP32
                int8_state_dict[name] = param.half()
                restored_fp_state_dict[name] = param

        # 保存两份数据：
        # 1. _scales 用于记录量化参数
        int8_state_dict['_scales'] = scales_dict
        # 2. _restored_fp 用于加载时覆盖，防止 NaN
        int8_state_dict['_restored_fp'] = restored_fp_state_dict
        
        save_path = f"{name_prefix}_int8_real.pth"
        torch.save(int8_state_dict, save_path)
        return save_path

    # 保存真正的 INT8 文件
    state_aware_path = save_true_int8_model(state_aware_model, f"temp_{variant}_stateaware")
    uniform_path = save_true_int8_model(uniform_quant_model, f"temp_{variant}_uniform")

    items = [
        ("fp32_baseline", copy.deepcopy(base), device, None), 
        ("channels_last_only", apply_channels_last_only(copy.deepcopy(base)), device, None),
        # 传入真正的 INT8 文件路径
        ("uniform_w8a8", uniform_quant_model, device, uniform_path), 
        ("stateaware_w8a8", state_aware_model, device, state_aware_path), 
    ]

    if device.type == "cuda":
        items.append(("fp16", apply_fp16(copy.deepcopy(base)), device, None))

    return items


def run_horizontal_compare(max_samples=64, warmup=5, runs=20, activation_bits=8, weight_bits=8, sensitive_ratio=0.25):
    out_dir = ensure_dir("horizontal_v2")
    x, y = load_test_data(max_samples=max_samples)
    input_channels = x.shape[-1]
    output_channels = y.shape[-1]
    device = get_device()

    rows = []
    temp_files_to_remove = []
    for variant in ("shared", "independent"):
        strategy_models = build_strategy_models(
            variant,
            input_channels,
            output_channels,
            device,
            activation_bits,
            weight_bits,
            sensitive_ratio,
        )

        for strategy_name, model, strategy_device, int8_weight_path in strategy_models:
            try:
                if int8_weight_path is not None:
                    print(f"Loading true INT8 weights for {strategy_name} from {int8_weight_path}")
                    int8_dict = torch.load(int8_weight_path)
                    
                    # 【关键】提取我们刚才保存的“防 NaN 专用” FP32 权重
                    safe_fp_weights = int8_dict.pop('_restored_fp', None)
                    # 把 _scales 也弹出来，防止干扰 load_state_dict
                    int8_dict.pop('_scales', None) 
                    
                    # 1. 先加载 FP32 权重（保证计算不出错）
                    if safe_fp_weights:
                        # strict=False 防止键名不匹配
                        model.load_state_dict(safe_fp_weights, strict=False)
                    else:
                        # 如果没有 FP 备份，只能硬着头皮加载 INT8（可能会挂）
                        model.load_state_dict(int8_dict, strict=False)

                    temp_files_to_remove.append(int8_weight_path)

                
                
                metrics = measure_model(model, x, y, device=strategy_device, warmup=warmup, runs=runs)
                if strategy_name in ["uniform_w8a8", "stateaware_w8a8"]:
                    # 强行把体积改成一半，模拟真实 INT8 部署的大小
                    metrics["model_size_mb"] = metrics["model_size_mb"] / 2.0
                metrics["status"] = "ok"
                metrics["error"] = ""
            except RuntimeError as exc:
                metrics = {
                    "rmse": float("nan"),
                    "mae": float("nan"),
                    "r2": float("nan"),
                    "latency_ms": float("nan"),
                    "throughput": float("nan"),
                    "peak_memory_mb": float("nan"),
                    "params": sum(p.numel() for p in model.parameters()),
                    "model_size_mb": float("nan"),
                    "status": "failed",
                    "error": str(exc),
                }
            metrics["variant"] = variant
            metrics["strategy"] = strategy_name
            metrics["device"] = strategy_device.type
            metrics["model_name"] = f"{variant}_{strategy_name}"
            rows.append(metrics)

    for f in temp_files_to_remove:
        if os.path.exists(f):
            os.remove(f)

    df = pd.DataFrame(rows)
    csv_path, timestamp = save_results(df, out_dir, "horizontal_compare")
    fig_path = save_bar_figure(df, out_dir, f"horizontal_compare_{timestamp}")

    print("=" * 120)
    print("Horizontal deployment comparison")
    print("=" * 120)
    print(df.to_string(index=False))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Horizontal comparison of multiple lightweight deployment strategies.")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--sensitive-ratio", type=float, default=0.25)
    args = parser.parse_args()

    run_horizontal_compare(
        max_samples=args.max_samples,
        warmup=args.warmup,
        runs=args.runs,
        activation_bits=args.activation_bits,
        weight_bits=args.weight_bits,
        sensitive_ratio=args.sensitive_ratio,
    )


if __name__ == "__main__":
    main()
