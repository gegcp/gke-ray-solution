#!/usr/bin/env python3
"""
RepVGG Image Inference on TPU v6e

Run inference on a real image using RepVGG on TPU.
"""

import argparse
import json
import time
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from repvgg import create_RepVGG_B0


def load_imagenet_classes(json_path):
    """Load ImageNet class names"""
    with open(json_path, 'r') as f:
        classes = json.load(f)
    return classes


def load_and_preprocess_image(image_path, image_size=224):
    """
    Load and preprocess image for inference

    Args:
        image_path: Path to input image
        image_size: Target size for image

    Returns:
        Preprocessed image tensor
    """
    # Define ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor, image


def load_model(weight_path=None):
    """Load RepVGG-B0 model"""
    print("Creating RepVGG-B0 model in training mode...")
    # IMPORTANT: Load in training mode first
    model = create_RepVGG_B0(deploy=False)

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

        result = model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded! Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")

    # Convert to deploy mode (fuse branches)
    print("Converting to deploy mode (fusing branches)...")
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()

    model.eval()
    return model


def run_inference_tpu(model, image_tensor, class_names, top_k=5):
    """
    Run inference on TPU

    Args:
        model: RepVGG model
        image_tensor: Preprocessed image tensor
        class_names: Dictionary of class names
        top_k: Number of top predictions to return

    Returns:
        List of (class_name, probability) tuples
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
        jax_input = torch_xla2.tensor.t2j(image_tensor)

    # Warmup
    print("Running warmup...")
    _ = forward_jax(jax_weights, jax_input)
    jax.block_until_ready(_)

    # Run inference
    print("Running inference...")
    start_time = time.time()
    output = forward_jax(jax_weights, jax_input)
    jax.block_until_ready(output)
    inference_time = time.time() - start_time

    print(f"Inference time: {inference_time*1000:.2f} ms")

    # Convert output to numpy and apply softmax
    import jax.numpy as jnp
    probabilities = jax.nn.softmax(output[0])
    probabilities = jnp.array(probabilities)

    # Get top-k predictions
    top_indices = jnp.argsort(probabilities)[::-1][:top_k]
    top_probs = probabilities[top_indices]

    # Format results
    results = []
    for idx, prob in zip(top_indices, top_probs):
        idx = int(idx)
        prob = float(prob)
        class_name = class_names.get(str(idx), f"Unknown (class {idx})")
        results.append((class_name, prob))

    return results, inference_time


def main():
    parser = argparse.ArgumentParser(description='Run RepVGG inference on image')
    parser.add_argument(
        '--image',
        type=str,
        default='/home/gech/gke-ray-solution/repvgg/gpu/burger.jpeg',
        help='Path to input image'
    )
    parser.add_argument(
        '--weight-path',
        type=str,
        default='/home/gech/gke-ray-solution/repvgg/tpu/weights/RepVGG-B0-train.pth',
        help='Path to pretrained weights'
    )
    parser.add_argument(
        '--classes',
        type=str,
        default='/home/gech/gke-ray-solution/repvgg/tpu/imagenet_classes.json',
        help='Path to ImageNet class names JSON'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top predictions to show'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Input image size'
    )

    args = parser.parse_args()

    print("="*60)
    print("RepVGG Image Inference on TPU v6e")
    print("="*60)

    # Load class names
    print(f"\nLoading ImageNet classes from {args.classes}...")
    class_names = load_imagenet_classes(args.classes)
    print(f"Loaded {len(class_names)} class names")

    # Load and preprocess image
    print(f"\nLoading image from {args.image}...")
    image_tensor, original_image = load_and_preprocess_image(args.image, args.image_size)
    print(f"Image shape: {image_tensor.shape}")
    print(f"Original image size: {original_image.size}")

    # Load model
    model = load_model(weight_path=args.weight_path)

    # Run inference
    predictions, inference_time = run_inference_tpu(
        model,
        image_tensor,
        class_names,
        top_k=args.top_k
    )

    # Display results
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    print(f"Image: {args.image}")
    print(f"Inference time: {inference_time*1000:.2f} ms")
    print(f"\nTop {args.top_k} predictions:")
    print("-"*60)
    for i, (class_name, prob) in enumerate(predictions, 1):
        print(f"{i}. {class_name:40s} {prob*100:6.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()
