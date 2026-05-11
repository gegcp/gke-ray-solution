# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RepVGG GPU Backend (no FastAPI ingress, for model composition)
"""

import os
import io
import base64
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict

from ray import serve


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


def create_RepVGG_A0(deploy=False, num_classes=1000):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)


@serve.deployment(
    name="RepVGGBackend",
    ray_actor_options={"num_cpus": 2, "num_gpus": 1},
    num_replicas=1,
)
class RepVGGBackend:
    """RepVGG backend for GPU (no FastAPI ingress)"""

    def __init__(self):
        model_variant = os.environ.get('MODEL_VARIANT', 'B0')
        num_classes = int(os.environ.get('NUM_CLASSES', '1000'))
        use_fp16 = os.environ.get('USE_FP16', 'true').lower() == 'true'

        print(f"Initializing RepVGG-{model_variant} backend")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if model_variant == 'A0':
            self.model = create_RepVGG_A0(deploy=True, num_classes=num_classes)
        else:
            self.model = create_RepVGG_B0(deploy=True, num_classes=num_classes)

        self.model.eval()
        self.model = self.model.to(self.device)

        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        if self.use_fp16:
            self.model = self.model.half()

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        print(f"✓ RepVGG backend ready on {self.device}")

    def classify(self, request: Dict) -> Dict:
        """Classify image (called by composition deployment)"""
        image_data = request.get("image")
        if not image_data:
            return {"error": "No image provided"}

        top_k = request.get("top_k", 5)
        img_size = request.get("img_size", 224)

        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            if image.mode != 'RGB':
                image = image.convert('RGB')

            image = image.resize((img_size, img_size), Image.BILINEAR)
            img_array = np.array(image).astype(np.float32) / 255.0
            img_array = (img_array - self.mean) / self.std
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float().unsqueeze(0)
            img_tensor = img_tensor.to(self.device)

            if self.use_fp16:
                img_tensor = img_tensor.half()

            with torch.no_grad():
                output = self.model(img_tensor)

            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_indices = torch.topk(probabilities, min(top_k, len(probabilities)))

            predictions = []
            for prob, idx in zip(top_prob, top_indices):
                predictions.append({
                    "class_id": int(idx.item()),
                    "probability": float(prob.item())
                })

            return {
                "predictions": predictions,
                "model": "RepVGG",
                "device": str(self.device)
            }
        except Exception as e:
            import traceback
            return {"error": f"Classification failed: {str(e)}", "traceback": traceback.format_exc()}
