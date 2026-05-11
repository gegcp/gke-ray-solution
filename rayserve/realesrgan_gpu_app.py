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
Real-ESRGAN GPU Inference with Ray Serve
Image super-resolution using Real-ESRGAN models on GPU
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

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request

from ray import serve


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
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
    """RRDB Network for Real-ESRGAN"""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = self._make_layer(RRDB, num_block, num_feat, num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 4:
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def _make_layer(self, block, num_blocks, *args):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(*args))
        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # Upsample
        feat = self.lrelu(self.conv_up1(nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.scale == 4:
            feat = self.lrelu(self.conv_up2(nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
            feat = self.lrelu(self.conv_hr(feat))
        feat = self.conv_last(feat)
        return feat


def tile_process(img, model, tile_size=512, tile_pad=10, scale=4, device='cuda'):
    """Process image in tiles to avoid OOM"""
    batch, channel, height, width = img.shape
    output_height = height * scale
    output_width = width * scale
    output_shape = (batch, channel, output_height, output_width)

    # Start with black image
    output = img.new_zeros(output_shape)
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size

    for y in range(tiles_y):
        for x in range(tiles_x):
            # Extract tile
            ofs_x = x * tile_size
            ofs_y = y * tile_size

            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            # Tile with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # Tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # Process tile
            with torch.no_grad():
                output_tile = model(input_tile)

            # Output tile area
            output_start_x = input_start_x * scale
            output_end_x = output_start_x + input_tile_width * scale
            output_start_y = input_start_y * scale
            output_end_y = output_start_y + input_tile_height * scale

            # Adjustment for padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            # Put tile into output
            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]

    return output


app = FastAPI()


@serve.deployment(
    name="RealESRGANDeployment",
    ray_actor_options={"num_cpus": 8, "num_gpus": 1},
    num_replicas=1,
)
@serve.ingress(app)
class RealESRGANGPUDeployment:
    """Real-ESRGAN deployment for GPU inference"""

    def __init__(self):
        """Initialize the Real-ESRGAN model on GPU"""
        # Get model configuration from environment
        model_name = os.environ.get('MODEL_NAME', 'RealESRGAN_x4plus')
        scale = int(os.environ.get('SCALE', '4'))
        use_fp16 = os.environ.get('USE_FP16', 'true').lower() == 'true'
        weight_path = os.environ.get('WEIGHT_PATH', '')
        self.tile_size = int(os.environ.get('TILE_SIZE', '0'))
        self.tile_pad = int(os.environ.get('TILE_PAD', '10'))

        print(f"Initializing Real-ESRGAN ({model_name}) on GPU")
        print(f"Scale factor: {scale}x")
        print(f"Use FP16: {use_fp16}")
        print(f"Tile size: {self.tile_size if self.tile_size > 0 else 'disabled'}")

        # Check CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")

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
        self.model = self.model.to(self.device)

        # Use FP16 if enabled and GPU available
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        if self.use_fp16:
            self.model = self.model.half()
            print("Using FP16 precision")

        self.scale = scale
        print("Real-ESRGAN model initialized successfully!")

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
        - scale: (optional) upscaling factor (default: model's scale)
        - tile_size: (optional) tile size for processing (default: deployment config)
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

        tile_size = request_dict.get("tile_size", self.tile_size)
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
            img_tensor = img_tensor.to(self.device)

            if self.use_fp16:
                img_tensor = img_tensor.half()

            # Run inference
            with torch.no_grad():
                if tile_size > 0:
                    output = tile_process(
                        img_tensor,
                        self.model,
                        tile_size=tile_size,
                        tile_pad=self.tile_pad,
                        scale=self.scale,
                        device=str(self.device)
                    )
                else:
                    output = self.model(img_tensor)

            # Postprocess
            output_image = self.postprocess_image(output)
            output_size = output_image.size

            print(f"Output image: {output_size[0]}x{output_size[1]}")

            # Prepare response
            response_data = {
                "model": "Real-ESRGAN",
                "device": str(self.device),
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
    async def health(self) -> Dict[str, str]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model": "Real-ESRGAN",
            "device": str(self.device),
            "precision": "FP16" if self.use_fp16 else "FP32",
            "scale": str(self.scale),
            "tile_size": str(self.tile_size if self.tile_size > 0 else "disabled")
        }


model = RealESRGANGPUDeployment.bind()
