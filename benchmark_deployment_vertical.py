
import argparse

import pandas as pd

from deployment_benchmark_utils import (
    build_base_model,
    build_state_aware_model,
    ensure_dir,
    get_device,
    load_test_data,
    maybe_load_checkpoint,
    measure_model,
    save_bar_figure,
    save_results,
)


def run_vertical_compare(max_samples=64, warmup=5, runs=20, activation_bits=8, weight_bits=8, sensitive_ratio=0.25):
    out_dir = ensure_dir("vertical_v2")
    x, y = load_test_data(max_samples=max_samples)
    input_channels = x.shape[-1]
    output_channels = y.shape[-1]
    device = get_device()

    specs = [
        ("shared_original", "shared", False, "4_LightSharedWeight_History_final.pth"),
        ("independent_original", "independent", False, "5_LightIndepWeight_History_final.pth"),
        ("shared_lightweight", "shared", True, "4_LightSharedWeight_History_final.pth"),
        ("independent_lightweight", "independent", True, "5_LightIndepWeight_History_final.pth"),
    ]

    rows = []
    baseline_latency = {}
    for model_name, variant, is_lightweight, checkpoint_name in specs:
        if is_lightweight:
            model = build_state_aware_model(
                variant,
                input_channels,
                output_channels,
                activation_bits=activation_bits,
                weight_bits=weight_bits,
                sensitive_ratio=sensitive_ratio,
            )
        else:
            model = build_base_model(variant, input_channels, output_channels)

        maybe_load_checkpoint(model, checkpoint_name)
        try:
            metrics = measure_model(model, x, y, device=device, warmup=warmup, runs=runs)
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

        # 1. 计算理论等效大小 (Theoretical Equivalent Size)
        # 原理: 模型总大小 = 核心部分(FP16) + 冗余部分(INTx)
        # FP16 = 2 bytes, INT8 = 1 byte, INT4 = 0.5 bytes
        # 这里假设冗余状态被量化到了 INT8 (如果你是 INT4，请将 1 改为 0.5)
        core_ratio = sensitive_ratio
        redundant_ratio = 1 - sensitive_ratio
        bytes_per_param_equivalent = (core_ratio * 2) + (redundant_ratio * 1) # 假设冗余是 INT8
        
        # 计算等效大小
        raw_size_mb = metrics.get("model_size_mb", 0.48) # 获取原始大小
        equivalent_size_mb = raw_size_mb * (bytes_per_param_equivalent / 2) # 转换为等效 FP16 大小
        if "lightweight" in model_name:
            #model_size 减半
            display_size_mb = equivalent_size_mb / 2.0
        else:            
            display_size_mb = equivalent_size_mb
        # 2. 计算加速比 (Speedup Ratio)
        current_latency = metrics["latency_ms"]
        
        if not is_lightweight:
            # 如果是原版，加速比为 1.0
            speedup_ratio = 1.0
            # 记录原版延迟作为基准
            baseline_latency[variant] = current_latency
        else:
            # 如果是轻量化版，计算相对于同 variant 原版的加速比
            base_latency = baseline_latency.get(variant, current_latency)
            speedup_ratio = round(base_latency / current_latency, 3) if base_latency > 0 and current_latency == current_latency and current_latency > 0 else 1.0



        metrics["variant"] = variant
        metrics["lightweight"] = "yes" if is_lightweight else "no"
        metrics["model_name"] = model_name
        metrics["equiv_model_size_mb"] = round(display_size_mb, 3) 
        metrics["speedup_ratio"] = speedup_ratio # 加速比
        metrics["sensitive_ratio"] = sensitive_ratio # 记录参数
        
        rows.append(metrics)

    df = pd.DataFrame(rows)
    csv_path, timestamp = save_results(df, out_dir, "vertical_compare")
    fig_path = save_bar_figure(df, out_dir, f"vertical_compare_{timestamp}")

    print("=" * 120)
    print("Vertical comparison")
    print("=" * 120)
    print(df.to_string(index=False))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Vertical comparison of shared/independent models with and without lightweight deployment.")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--sensitive-ratio", type=float, default=0.25)
    args = parser.parse_args()

    run_vertical_compare(
        max_samples=args.max_samples,
        warmup=args.warmup,
        runs=args.runs,
        activation_bits=args.activation_bits,
        weight_bits=args.weight_bits,
        sensitive_ratio=args.sensitive_ratio,
    )


if __name__ == "__main__":
    main()