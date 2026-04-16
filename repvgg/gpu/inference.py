#!/usr/bin/env python3
"""
RepVGG Inference Script with GPU Support
Image classification using RepVGG models
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import cv2
import json


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    """Convolution + BatchNorm layer"""
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    """RepVGG Block with reparameterization"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups,
                                         bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                   stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)


class RepVGG(nn.Module):
    """RepVGG Model"""

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def create_RepVGG_B0(deploy=False, num_classes=1000):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B1(deploy=False, num_classes=1000):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)


def create_RepVGG_B2(deploy=False, num_classes=1000):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)


def create_RepVGG_A0(deploy=False, num_classes=1000):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)


def preprocess_image(image_path, img_size=224):
    """Preprocess image for RepVGG inference"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, (img_size, img_size))

    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std

    # Convert to tensor (CHW format)
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()

    return img.unsqueeze(0)  # Add batch dimension


def load_imagenet_labels():
    """Load ImageNet class labels"""
    # Try to load from local file first
    labels_file = Path(__file__).parent / 'imagenet_classes.json'

    if labels_file.exists():
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        return labels

    # Default: return index as label
    return {str(i): f"class_{i}" for i in range(1000)}


def main():
    parser = argparse.ArgumentParser(description='RepVGG Image Classification')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input image path')
    parser.add_argument('-w', '--weights', type=str, default='RepVGG-B0-train.pth',
                        help='Model weights path (default: RepVGG-B0-train.pth)')
    parser.add_argument('-m', '--model', type=str, default='B0',
                        choices=['A0', 'B0', 'B1', 'B2'],
                        help='RepVGG model variant (default: B0)')
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='Number of classes (default: 1000 for ImageNet)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Show top K predictions (default: 5)')
    parser.add_argument('--fp32', action='store_true',
                        help='Use FP32 precision (default: FP16)')

    args = parser.parse_args()

    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if device.type == 'cpu':
        print('WARNING: CUDA not available. Running on CPU will be slow.')
    else:
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'CUDA Version: {torch.version.cuda}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

    # Model weights path
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = Path(__file__).parent / weights_path

    if not weights_path.exists():
        raise FileNotFoundError(f'Model weights not found: {weights_path}')

    print(f'\nLoading model: RepVGG-{args.model}')
    print(f'Weights: {weights_path}')

    # Create model
    model_creators = {
        'A0': create_RepVGG_A0,
        'B0': create_RepVGG_B0,
        'B1': create_RepVGG_B1,
        'B2': create_RepVGG_B2,
    }

    model = model_creators[args.model](deploy=False, num_classes=args.num_classes)

    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)

    if not args.fp32 and device.type == 'cuda':
        model = model.half()
        print('Using FP16 (half precision)')
    else:
        print('Using FP32 (full precision)')

    # Load class labels
    class_labels = load_imagenet_labels()

    # Preprocess image
    print(f'\nProcessing image: {args.input}')
    img_tensor = preprocess_image(args.input, args.img_size)
    img_tensor = img_tensor.to(device)

    if not args.fp32 and device.type == 'cuda':
        img_tensor = img_tensor.half()

    # Run inference
    with torch.no_grad():
        output = model(img_tensor)

    # Get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top K predictions
    top_prob, top_indices = torch.topk(probabilities, args.top_k)

    print(f'\nTop {args.top_k} Predictions:')
    print('-' * 60)
    for i, (prob, idx) in enumerate(zip(top_prob, top_indices), 1):
        class_idx = idx.item()
        class_name = class_labels.get(str(class_idx), f'class_{class_idx}')
        print(f'{i}. {class_name:30s} {prob.item()*100:6.2f}%')

    print('\nInference complete!')


if __name__ == '__main__':
    main()
