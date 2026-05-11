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
Real-ESRGAN TPU Inference with Ray Serve
Image super-resolution using Real-ESRGAN models on TPU using torch-xla2 (torchax)
"""

import os
import sys
import io
import base64
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request

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
    """Pixel unshuffle: (b, c, hh, hw) -> (b, c*scale^2, h, w)."""
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


def nearest_upsample_2x(x):
    """2x nearest-neighbor upsample using repeat_interleave.

    Replaces F.interpolate(scale_factor=2, mode='nearest') which has a bug
    in torchax (raises OperatorNotFound for nearest mode).

    Pure tensor ops — 100% XLA compatible.
    """
    return x.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block with 5 convolutions.

    Each conv takes the concatenation of all previous outputs as input.
    Uses LeakyReLU(0.2) activation (non-inplace for XLA compatibility).
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        # Use non-inplace LeakyReLU for XLA/torchax compatibility
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
    """Residual in Residual Dense Block (3x ResidualDenseBlock)."""

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
    """RRDBNet: Residual in Residual Dense Block Network for super-resolution.

    For scale=2: uses pixel_unshuffle to reduce spatial size before processing.
    For scale=4: direct input, two 2x upsamples via nearest_upsample_2x (XLA compatible).

    Args:
        num_in_ch (int): Input channel number. Default: 3.
        num_out_ch (int): Output channel number. Default: 3.
        scale (int): Upsampling factor. Default: 4.
        num_feat (int): Intermediate feature channels. Default: 64.
        num_block (int): Number of RRDB blocks. Default: 23.
        num_grow_ch (int): Growth channels per dense layer. Default: 32.
    """

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
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # Use non-inplace for XLA/torchax compatibility
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
        # upsample (use repeat_interleave instead of F.interpolate for XLA compat)
        feat = self.lrelu(self.conv_up1(nearest_upsample_2x(feat)))
        feat = self.lrelu(self.conv_up2(nearest_upsample_2x(feat)))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def patch_torchax_conv2d():
    """Patch torchax conv2d to include default arguments (optional, for older torch-xla2 versions)"""
    try:
        import torch_xla2.ops.ops_registry as ops_registry
        import torch_xla2.ops.jaten as jaten_ops
        from functools import wraps

        # Check if conv2d exists in jaten_ops (may not exist in newer versions)
        if not hasattr(jaten_ops, 'conv2d'):
            print("ℹ Skipping conv2d patch (not needed in this torch-xla2 version)")
            return

        original_conv2d = jaten_ops.conv2d

        @wraps(original_conv2d)
        def patched_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
            return original_conv2d(input, weight, bias, stride, padding, dilation, groups)

        ops_registry.register_torch_dispatch_op(
            torch.ops.aten.conv2d,
            patched_conv2d,
            is_jax_function=True
        )
        print("✓ Applied conv2d patch")
    except Exception as e:
        print(f"ℹ Skipping conv2d patch: {e}")


def set_model_float32(model):
    """Ensure all conv layers use float32 precision"""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data = module.weight.data.to(torch.float32)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(torch.float32)
    return model


app = FastAPI()


@serve.deployment(
    name="RealESRGANDeployment",
    num_replicas=1,
)
@serve.ingress(app)
class RealESRGANTPUDeployment:
    """Real-ESRGAN deployment for TPU inference"""

    def __init__(self):
        """Initialize the Real-ESRGAN model on TPU with torch-xla2"""
        # Get model configuration from environment
        model_name = os.environ.get('MODEL_NAME', 'RealESRGAN_x4plus')
        scale = int(os.environ.get('SCALE', '4'))
        enable_tpu = os.environ.get('ENABLE_TPU', 'false').lower() == 'true'
        weight_path = os.environ.get('WEIGHT_PATH', '')

        print(f"Initializing Real-ESRGAN ({model_name})")
        print(f"Scale factor: {scale}x")
        print(f"TPU acceleration enabled: {enable_tpu}")

        # Try to initialize TPU with torch-xla2 if enabled
        self.use_tpu = False
        self.env = None

        if enable_tpu:
            try:
                import jax
                import torch_xla2

                print("Initializing torch-xla2 (torchax) for TPU...")

                # Set JAX configuration for highest precision
                jax.config.update('jax_default_matmul_precision', 'highest')

                # Patch conv2d before initializing torchax
                patch_torchax_conv2d()

                # Initialize torchax environment
                self.env = torch_xla2.default_env()
                self.env.__enter__()

                self.use_tpu = True
                self.device_name = "TPU (JAX/XLA)"

                print(f"✓ JAX devices: {jax.devices()}")
                print(f"✓ JAX backend: {jax.default_backend()}")
                print(f"✓ Using TPU with torch-xla2 (torchax)")

            except ImportError as e:
                print(f"⚠ torch-xla2 not available: {e}")
                print("⚠ Falling back to CPU mode")
                print("  To enable TPU: Set ENABLE_TPU=true and add jax[tpu], torch-xla2 to pip dependencies")
                self.use_tpu = False
                self.device_name = "CPU (TPU node)"

            except Exception as e:
                print(f"⚠ Could not initialize TPU: {e}")
                print("⚠ Falling back to CPU mode")
                self.use_tpu = False
                self.device_name = "CPU (TPU node)"
        else:
            print("ℹ TPU acceleration disabled (ENABLE_TPU=false)")
            print("  Running on TPU node in CPU mode")
            print("  To enable: Set ENABLE_TPU=true and add torch-xla2 dependencies")
            self.use_tpu = False
            self.device_name = "CPU (TPU node)"

        # Create model based on configuration
        if 'anime_6B' in model_name:
            num_block = 6
        else:
            num_block = 23

        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=num_block,
            num_grow_ch=32,
            scale=scale
        )

        # Load weights if path provided
        if weight_path and Path(weight_path).exists():
            print(f"Loading weights from {weight_path}")
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

            self.model.load_state_dict(state_dict, strict=True)
            print("Weights loaded successfully!")
        else:
            print("Warning: No weights loaded, using random initialization")

        self.model.eval()

        # For TPU, ensure float32 precision
        if self.use_tpu:
            self.model = set_model_float32(self.model)

        self.scale = scale
        print(f"✓ Real-ESRGAN model initialized successfully on {self.device_name}!")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for Real-ESRGAN inference"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(image).astype(np.float32) / 255.0

        # Convert to tensor (CHW format) - note BGR to RGB conversion
        img_tensor = torch.from_numpy(img_array[:, :, [2, 1, 0]].transpose(2, 0, 1)).float()

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor

    def postprocess_image(self, output_tensor: torch.Tensor) -> Image.Image:
        """Convert output tensor to PIL Image"""
        # Remove batch dimension and convert to numpy
        output_np = output_tensor.squeeze(0).float().cpu().clamp(0, 1).numpy()

        # Convert from CHW to HWC and BGR to RGB
        output_np = output_np[[2, 1, 0], :, :].transpose(1, 2, 0)

        # Convert to uint8
        output_np = (output_np * 255.0).round().astype(np.uint8)

        # Create PIL Image
        return Image.fromarray(output_np)

    @app.post("/upscale")
    async def upscale(self, request: Request) -> JSONResponse:
        """
        Upscale image using Real-ESRGAN model

        Request body should contain:
        - image: base64 encoded image string
        - return_image: (optional) return base64 image if true, else just metadata (default: true)
        """
        request_dict = await request.json()

        # Get image from request
        image_data = request_dict.get("image")
        if not image_data:
            return JSONResponse(
                status_code=400,
                content={"error": "No image provided"}
            )

        return_image = request_dict.get("return_image", True)

        try:
            # Decode base64 image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            original_size = image.size
            print(f"Processing image: {original_size[0]}x{original_size[1]}")

            # Preprocess image
            img_tensor = self.preprocess_image(image)

            # Run inference
            if self.use_tpu:
                # TPU inference with JAX
                import jax

                with torch.no_grad():
                    output = self.model(img_tensor)

                # Convert JAX array to numpy then back to torch for consistency
                # This ensures computation is complete
                output_np = np.array(output)
                output = torch.from_numpy(output_np)
            else:
                # CPU inference
                with torch.no_grad():
                    output = self.model(img_tensor)

            # Postprocess
            output_image = self.postprocess_image(output)
            output_size = output_image.size

            print(f"Output image: {output_size[0]}x{output_size[1]}")

            # Prepare response
            response_data = {
                "model": "Real-ESRGAN-TPU",
                "device": self.device_name,
                "input_size": {"width": original_size[0], "height": original_size[1]},
                "output_size": {"width": output_size[0], "height": output_size[1]},
                "scale": self.scale
            }

            if return_image:
                # Convert to base64
                buffered = io.BytesIO()
                output_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                response_data["image"] = f"data:image/png;base64,{img_str}"

            return JSONResponse(content=response_data)

        except Exception as e:
            import traceback
            return JSONResponse(
                status_code=500,
                content={"error": f"Upscaling failed: {str(e)}", "traceback": traceback.format_exc()}
            )

    @app.get("/health")
    async def health(self):
        """Health check endpoint"""
        health_info = {
            "status": "healthy",
            "model": "Real-ESRGAN-TPU",
            "device": self.device_name,
            "tpu_enabled": self.use_tpu,
            "scale": str(self.scale)
        }

        if self.use_tpu:
            try:
                import jax
                health_info["jax_backend"] = jax.default_backend()
                health_info["jax_device_count"] = len(jax.devices())
            except:
                pass
        else:
            health_info["note"] = "Install torch-xla2 for TPU acceleration"

        return health_info


model = RealESRGANTPUDeployment.bind()
