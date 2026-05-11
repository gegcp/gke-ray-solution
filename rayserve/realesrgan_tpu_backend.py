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
Real-ESRGAN TPU Backend (no FastAPI ingress, for model composition)
"""

import os
import io
import base64
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict

from ray import serve


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights."""
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks."""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def pixel_unshuffle(x, scale):
    """Pixel unshuffle."""
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


def nearest_upsample_2x(x):
    """2x nearest-neighbor upsample (XLA compatible)."""
    return x.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet for super-resolution."""

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(nearest_upsample_2x(feat)))
        feat = self.lrelu(self.conv_up2(nearest_upsample_2x(feat)))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def set_model_float32(model):
    """Ensure all conv layers use float32 precision"""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data = module.weight.data.to(torch.float32)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(torch.float32)
    return model


@serve.deployment(
    name="RealESRGANBackend",
    ray_actor_options={"num_cpus": 1, "resources": {"TPU": 1}},
    num_replicas=1,
)
class RealESRGANBackend:
    """Real-ESRGAN backend for TPU (no FastAPI ingress)"""

    def __init__(self):
        model_name = os.environ.get('MODEL_NAME', 'RealESRGAN_x4plus')
        scale = int(os.environ.get('SCALE', '4'))
        enable_tpu = os.environ.get('ENABLE_TPU', 'false').lower() == 'true'

        print(f"Initializing Real-ESRGAN backend ({model_name}, scale={scale}x)")

        self.use_tpu = False
        self.env = None

        if enable_tpu:
            try:
                import jax
                import torch_xla2
                jax.config.update('jax_default_matmul_precision', 'highest')
                self.env = torch_xla2.default_env()
                self.env.__enter__()
                self.use_tpu = True
                self.device_name = "TPU (JAX/XLA)"
                print(f"✓ TPU initialized: {jax.devices()}")
            except Exception as e:
                print(f"⚠ TPU init failed: {e}, using CPU")
                self.use_tpu = False
                self.device_name = "CPU"
        else:
            self.device_name = "CPU"

        num_block = 6 if 'anime_6B' in model_name else 23
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=num_block, num_grow_ch=32, scale=scale)
        self.model.eval()

        if self.use_tpu:
            self.model = set_model_float32(self.model)

        self.scale = scale
        print(f"✓ Real-ESRGAN backend ready on {self.device_name}")

    def upscale(self, request: Dict) -> Dict:
        """Upscale image (called by composition deployment)"""
        image_data = request.get("image")
        if not image_data:
            return {"error": "No image provided"}

        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            if image.mode != 'RGB':
                image = image.convert('RGB')

            original_size = image.size
            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array[:, :, [2, 1, 0]].transpose(2, 0, 1)).float().unsqueeze(0)

            with torch.no_grad():
                output = self.model(img_tensor)

            if self.use_tpu:
                output_np = np.array(output)
                output = torch.from_numpy(output_np)

            output_np = output.squeeze(0).float().cpu().clamp(0, 1).numpy()
            output_np = output_np[[2, 1, 0], :, :].transpose(1, 2, 0)
            output_np = (output_np * 255.0).round().astype(np.uint8)
            output_image = Image.fromarray(output_np)
            output_size = output_image.size

            buffered = io.BytesIO()
            output_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return {
                "image": f"data:image/png;base64,{img_str}",
                "model": "Real-ESRGAN-TPU",
                "device": self.device_name,
                "input_size": {"width": original_size[0], "height": original_size[1]},
                "output_size": {"width": output_size[0], "height": output_size[1]},
                "scale": self.scale
            }
        except Exception as e:
            import traceback
            return {"error": f"Upscaling failed: {str(e)}", "traceback": traceback.format_exc()}
