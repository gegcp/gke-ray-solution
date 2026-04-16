#!/usr/bin/env python3
"""
Simple RepVGG inference example on TPU v6e

This is a minimal example to get started with RepVGG on TPU.
"""

import torch
from repvgg import create_RepVGG_B0


def main():
    # Load model with pretrained weights
    print("Loading RepVGG-B0 model...")
    model = create_RepVGG_B0(deploy=True)

    # Load weights
    weight_path = '/home/gech/gke-ray-solution/repvgg/tpu/weights/RepVGG-B0-train.pth'
    checkpoint = torch.load(weight_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded successfully!")

    # Create sample input (batch_size=1, channels=3, height=224, width=224)
    print("\nRunning inference on sample input...")
    sample_input = torch.randn(1, 3, 224, 224)

    # Run inference
    with torch.no_grad():
        output = model(sample_input)

    print(f"Input shape:  {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (first 10 values): {output[0, :10].tolist()}")
    print("\nInference completed successfully!")


if __name__ == '__main__':
    main()
