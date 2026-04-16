#!/usr/bin/env python3
"""
RepVGG Inference on TPU v6e using torch_xla2

Simplified version using torch_xla2 interop for JAX/XLA compilation.
"""

import argparse
import time
import torch
import torch.nn as nn
from repvgg import create_RepVGG_B0


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
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)

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
    import torch_xla2

    # Set JAX configuration for highest precision
    jax.config.update('jax_default_matmul_precision', 'highest')

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")

    # Create dummy input
    print(f"\nCreating input tensor with shape {input_shape}...")
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    # Convert PyTorch model to JAX using torch_xla2
    print("Converting model to JAX...")

    # Extract JAX function from PyTorch model
    jax_weights, jax_func = torch_xla2.extract_jax(model)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # JIT compile the function
    print("JIT compiling forward function...")

    @jax.jit
    def forward_jax(weights, x):
        return jax_func(weights, x)

    # Convert input to JAX
    import torch_xla2.tensor
    with torch_xla2.default_env():
        jax_input = torch_xla2.tensor.t2j(dummy_input)

    # Warmup runs
    print(f"\nRunning {warmup} warmup iterations...")
    warmup_start = time.time()
    for i in range(warmup):
        output = forward_jax(jax_weights, jax_input)
        jax.block_until_ready(output)
        if i == 0:
            print(f"  First output shape: {output.shape}")
    warmup_time = time.time() - warmup_start
    print(f"Warmup completed in {warmup_time:.2f}s")

    # Benchmark runs
    print(f"\nRunning {num_iterations} benchmark iterations...")
    latencies = []

    for i in range(num_iterations):
        start = time.time()
        output = forward_jax(jax_weights, jax_input)
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
