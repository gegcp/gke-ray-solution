#!/usr/bin/env python3
"""
RepVGG Inference on TPU v6e using torchax

This script demonstrates how to run RepVGG model on Google Cloud TPU v6e
using the torchax framework for JAX/XLA compilation.
"""

import argparse
import time
import torch
import torch.nn as nn
from repvgg import create_RepVGG_B0


def patch_torchax_conv2d():
    """Patch torchax conv2d to include default arguments"""
    import torch_xla2.ops.ops_registry as ops_registry
    import torch_xla2.ops.jaten as jaten_ops
    from functools import wraps

    original_conv2d = jaten_ops.conv2d

    @wraps(original_conv2d)
    def patched_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return original_conv2d(input, weight, bias, stride, padding, dilation, groups)

    ops_registry.register_torch_dispatch_op(
        torch.ops.aten.conv2d,
        patched_conv2d,
        is_jax_function=True
    )


def set_model_float32(model):
    """Ensure all conv layers use float32 precision"""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data = module.weight.data.to(torch.float32)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(torch.float32)
    return model


def load_model(weight_path=None, deploy=True):
    """
    Load RepVGG-B0 model

    Args:
        weight_path: Path to pretrained weights (.pth file)
        deploy: If True, convert model to deploy mode (fused branches)

    Returns:
        RepVGG model
    """
    print("Creating RepVGG-B0 model...")
    model = create_RepVGG_B0(deploy=deploy)

    if weight_path:
        print(f"Loading weights from {weight_path}...")
        checkpoint = torch.load(weight_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully!")

    model.eval()
    return model


def initialize_torchax():
    """Initialize torchax with JAX backend"""
    import jax
    import torch_xla2

    # Set JAX configuration for highest precision
    jax.config.update('jax_default_matmul_precision', 'highest')

    # Patch conv2d before initializing torchax
    patch_torchax_conv2d()

    # Initialize torchax environment
    env = torch_xla2.default_env()
    env.__enter__()

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")

    return env


def run_inference_tpu(model, input_shape=(1, 3, 224, 224), num_iterations=100, warmup=10):
    """
    Run inference on TPU with benchmarking

    Args:
        model: RepVGG model
        input_shape: Input tensor shape (batch, channels, height, width)
        num_iterations: Number of inference iterations for benchmarking
        warmup: Number of warmup iterations

    Returns:
        Performance metrics dictionary
    """
    import jax
    import jax.numpy as jnp

    # Ensure model is in float32
    model = set_model_float32(model)

    # Create dummy input
    print(f"Creating input tensor with shape {input_shape}...")
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    # Initialize torchax environment
    env = initialize_torchax()

    # Transfer model to JAX device
    print("Transferring model to TPU...")
    jax_model = model

    # Transfer input to JAX device
    print("Transferring input to TPU...")

    # JIT compile the forward pass
    print("JIT compiling forward function...")

    @jax.jit
    def forward_jax(input_tensor):
        return jax_model(input_tensor)

    # Warmup runs
    print(f"Running {warmup} warmup iterations...")
    warmup_start = time.time()
    for i in range(warmup):
        output = forward_jax(dummy_input)
        if i == 0:
            print(f"  First output shape: {output.shape}")
    warmup_time = time.time() - warmup_start
    print(f"Warmup completed in {warmup_time:.2f}s")

    # Benchmark runs
    print(f"\nRunning {num_iterations} benchmark iterations...")
    latencies = []

    for i in range(num_iterations):
        start = time.time()
        output = forward_jax(dummy_input)
        # Ensure computation completes (JAX is async)
        jax.block_until_ready(output)
        latency = time.time() - start
        latencies.append(latency)

        if (i + 1) % 20 == 0:
            print(f"  Iteration {i + 1}/{num_iterations}")

    # Calculate statistics
    latencies = sorted(latencies)
    avg_latency = sum(latencies) / len(latencies)
    median_latency = latencies[len(latencies) // 2]
    p99_latency = latencies[int(len(latencies) * 0.99)]
    min_latency = latencies[0]
    max_latency = latencies[-1]

    # Print results
    print("\n" + "="*60)
    print("PERFORMANCE RESULTS")
    print("="*60)
    print(f"Input shape:        {input_shape}")
    print(f"Output shape:       {tuple(output.shape)}")
    print(f"Model:              RepVGG-B0")
    print(f"Device:             {jax.devices()[0]}")
    print(f"Iterations:         {num_iterations}")
    print("-"*60)
    print(f"Average latency:    {avg_latency*1000:.2f} ms")
    print(f"Median latency:     {median_latency*1000:.2f} ms")
    print(f"P99 latency:        {p99_latency*1000:.2f} ms")
    print(f"Min latency:        {min_latency*1000:.2f} ms")
    print(f"Max latency:        {max_latency*1000:.2f} ms")
    print(f"Throughput:         {1.0/avg_latency:.2f} samples/sec")
    print(f"Compilation time:   {warmup_time:.2f} s")
    print("="*60)

    return {
        'avg_latency': avg_latency,
        'median_latency': median_latency,
        'p99_latency': p99_latency,
        'min_latency': min_latency,
        'max_latency': max_latency,
        'throughput': 1.0 / avg_latency,
        'compilation_time': warmup_time,
        'output_shape': tuple(output.shape)
    }


def main():
    parser = argparse.ArgumentParser(description='Run RepVGG inference on TPU v6e')
    parser.add_argument(
        '--weight-path',
        type=str,
        default='/home/gech/gke-ray-solution/repvgg/tpu/weights/RepVGG-B0-train.pth',
        help='Path to pretrained weights'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Input image size (height and width)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of benchmark iterations'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Number of warmup iterations'
    )
    parser.add_argument(
        '--deploy',
        action='store_true',
        default=True,
        help='Use deploy mode (fused branches)'
    )

    args = parser.parse_args()

    print("="*60)
    print("RepVGG TPU v6e Inference")
    print("="*60)

    # Load model
    model = load_model(weight_path=args.weight_path, deploy=args.deploy)

    # Run inference on TPU
    input_shape = (args.batch_size, 3, args.image_size, args.image_size)
    metrics = run_inference_tpu(
        model,
        input_shape=input_shape,
        num_iterations=args.iterations,
        warmup=args.warmup
    )

    print("\nInference completed successfully!")


if __name__ == '__main__':
    main()
