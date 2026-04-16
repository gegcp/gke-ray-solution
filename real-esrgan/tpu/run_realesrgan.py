#!/usr/bin/env python3
"""
Real-ESRGAN Inference on TPU v6e

Simple script to upscale images using Real-ESRGAN on TPU.
"""

import argparse
import time
import numpy as np
import torch
from PIL import Image
from rrdbnet import RRDBNet


def load_model(weight_path, scale=4):
    """
    Load Real-ESRGAN model

    Args:
        weight_path: Path to model weights (.pth file)
        scale: Upscaling factor (2 or 4)

    Returns:
        RRDBNet model
    """
    print(f"Creating Real-ESRGAN model (scale={scale}x)...")

    # RealESRGAN_x4plus uses:
    # - num_in_ch=3, num_out_ch=3
    # - num_feat=64
    # - num_block=23
    # - num_grow_ch=32
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale
    )

    if weight_path:
        print(f"Loading weights from {weight_path}...")
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)

        # Handle different checkpoint formats
        if 'params_ema' in checkpoint:
            state_dict = checkpoint['params_ema']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Load state dict
        model.load_state_dict(state_dict, strict=True)
        print("Weights loaded successfully!")

    model.eval()
    return model


def load_image(image_path):
    """Load and preprocess image for Real-ESRGAN"""
    print(f"Loading image from {image_path}...")
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0

    # Convert to tensor (C, H, W)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    print(f"Input image shape: {img.size} (W x H)")
    print(f"Input tensor shape: {img_tensor.shape}")

    return img_tensor, img


def save_image(output_tensor, save_path):
    """Save output tensor as image"""
    # Convert from tensor to numpy
    output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)

    # Save as image
    output_img = Image.fromarray(output_np)
    output_img.save(save_path)
    print(f"Output saved to {save_path}")
    print(f"Output image size: {output_img.size} (W x H)")

    return output_img


def run_inference_tpu(model, input_tensor, tile_size=None):
    """
    Run inference on TPU

    Args:
        model: RRDBNet model
        input_tensor: Input image tensor
        tile_size: Optional tile size for large images (e.g., 512)

    Returns:
        Output tensor and inference time
    """
    import jax
    import torch_xla2

    # Set JAX configuration
    jax.config.update('jax_default_matmul_precision', 'highest')

    print(f"\nJAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")

    # Convert PyTorch model to JAX
    print("\nConverting model to JAX...")
    jax_weights, jax_func = torch_xla2.extract_jax(model)

    # JIT compile
    print("JIT compiling...")

    @jax.jit
    def forward_jax(weights, x):
        return jax_func(weights, x)

    # Convert input to JAX
    with torch_xla2.default_env():
        jax_input = torch_xla2.tensor.t2j(input_tensor)

    # Warmup
    print("Running warmup...")
    warmup_start = time.time()
    _ = forward_jax(jax_weights, jax_input)
    jax.block_until_ready(_)
    warmup_time = time.time() - warmup_start
    print(f"Warmup/compilation time: {warmup_time:.2f}s")

    # Run inference
    print("\nRunning inference...")
    start_time = time.time()
    output = forward_jax(jax_weights, jax_input)
    jax.block_until_ready(output)
    inference_time = time.time() - start_time

    print(f"Inference time: {inference_time*1000:.2f} ms")
    print(f"Output shape: {output.shape}")

    # Convert back to PyTorch tensor
    with torch_xla2.default_env():
        output_tensor = torch_xla2.tensor.j2t(output)

    return output_tensor, inference_time


def run_inference_cpu(model, input_tensor):
    """Run inference on CPU (for comparison/testing)"""
    print("\nRunning inference on CPU...")

    with torch.no_grad():
        start_time = time.time()
        output = model(input_tensor)
        inference_time = time.time() - start_time

    print(f"CPU Inference time: {inference_time*1000:.2f} ms")
    print(f"Output shape: {output.shape}")

    return output, inference_time


def main():
    parser = argparse.ArgumentParser(description='Real-ESRGAN inference on TPU v6e')
    parser.add_argument(
        '--input',
        type=str,
        default='/home/gech/gke-ray-solution/real-esrgan/gpu/eva.jpg',
        help='Path to input image'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_upscaled.png',
        help='Path to save output image'
    )
    parser.add_argument(
        '--weight-path',
        type=str,
        default='/home/gech/gke-ray-solution/real-esrgan/tpu/weights/RealESRGAN_x4plus.pth',
        help='Path to model weights'
    )
    parser.add_argument(
        '--scale',
        type=int,
        default=4,
        choices=[2, 4],
        help='Upscaling factor'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Run on CPU instead of TPU (for testing)'
    )
    parser.add_argument(
        '--tile-size',
        type=int,
        default=None,
        help='Tile size for large images (optional)'
    )

    args = parser.parse_args()

    print("="*60)
    print("Real-ESRGAN Image Super-Resolution on TPU v6e")
    print("="*60)

    # Load model
    model = load_model(args.weight_path, scale=args.scale)

    # Load input image
    input_tensor, original_img = load_image(args.input)

    # Run inference
    if args.cpu:
        output_tensor, inference_time = run_inference_cpu(model, input_tensor)
    else:
        output_tensor, inference_time = run_inference_tpu(model, input_tensor, args.tile_size)

    # Save output
    output_img = save_image(output_tensor, args.output)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Input:          {args.input}")
    print(f"Input size:     {original_img.size[0]} x {original_img.size[1]}")
    print(f"Output:         {args.output}")
    print(f"Output size:    {output_img.size[0]} x {output_img.size[1]}")
    print(f"Scale factor:   {args.scale}x")
    print(f"Inference time: {inference_time*1000:.2f} ms")
    print(f"Device:         {'CPU' if args.cpu else 'TPU v6e'}")
    print("="*60)


if __name__ == '__main__':
    main()
