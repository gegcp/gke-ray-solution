#!/usr/bin/env python3
"""
Real-ESRGAN Inference Script with GPU Support
Upscale images using Real-ESRGAN models
"""

import argparse
import cv2
import glob
import os
from pathlib import Path
import torch
import numpy as np


class RRDBNet(torch.nn.Module):
    """RRDB Network for Real-ESRGAN"""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale

        self.conv_first = torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = self._make_layer(RRDB, num_block, num_feat, num_grow_ch)
        self.conv_body = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling
        self.conv_up1 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 4:
            self.conv_hr = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = torch.nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def _make_layer(self, block, num_blocks, *args):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(*args))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # Upsample
        feat = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.scale == 4:
            feat = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
            feat = self.lrelu(self.conv_hr(feat))
        feat = self.conv_last(feat)
        return feat


class ResidualDenseBlock(torch.nn.Module):
    """Residual Dense Block"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(torch.nn.Module):
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

            tile_idx = y * tiles_x + x + 1
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


def enhance_image(img, model, scale=4, tile_size=0, tile_pad=10, device='cuda', half_precision=True):
    """Enhance image with Real-ESRGAN model"""
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0).to(device)

    if half_precision:
        img = img.half()

    try:
        with torch.no_grad():
            if tile_size > 0:
                output = tile_process(img, model, tile_size, tile_pad, scale, device)
            else:
                output = model(img)
    except RuntimeError as error:
        if 'out of memory' in str(error):
            print('  WARNING: Out of memory, retrying with CPU...')
            torch.cuda.empty_cache()
            img = img.cpu()
            model = model.cpu()
            with torch.no_grad():
                if tile_size > 0:
                    output = tile_process(img, model, tile_size, tile_pad, scale, 'cpu')
                else:
                    output = model(img)
            model = model.to(device)
        else:
            raise error

    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    return output


def main():
    parser = argparse.ArgumentParser(description='Real-ESRGAN Image Upscaling')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input image or folder path')
    parser.add_argument('-o', '--output', type=str, default='results',
                        help='Output folder path (default: results)')
    parser.add_argument('-m', '--model', type=str, default='RealESRGAN_x4plus',
                        help='Model name (default: RealESRGAN_x4plus)')
    parser.add_argument('-s', '--scale', type=int, default=4,
                        help='Output scale factor (default: 4)')
    parser.add_argument('--fp32', action='store_true',
                        help='Use FP32 precision (default: FP16)')
    parser.add_argument('--tile', type=int, default=0,
                        help='Tile size for processing large images (0 = no tiling, try 512 for large images)')
    parser.add_argument('--tile_pad', type=int, default=10,
                        help='Tile padding')
    parser.add_argument('--suffix', type=str, default='out',
                        help='Suffix for output filename')

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
    weights_dir = Path(__file__).parent / 'weights'
    model_path = weights_dir / f'{args.model}.pth'

    if not model_path.exists():
        print(f'ERROR: Model weight not found: {model_path}')
        print(f'Available weights in {weights_dir}:')
        for weight_file in weights_dir.glob('*.pth'):
            print(f'  - {weight_file.name}')
        return

    print(f'Loading model: {args.model} from {model_path}')

    # Create model
    if 'anime_6B' in args.model:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=args.scale)
    elif 'x2' in args.model:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=args.scale)

    # Load weights
    loadnet = torch.load(model_path, map_location=device)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    elif 'params' in loadnet:
        keyname = 'params'
    else:
        keyname = 'state_dict'

    model.load_state_dict(loadnet[keyname], strict=True)
    model.eval()
    model = model.to(device)

    if not args.fp32 and device.type == 'cuda':
        model = model.half()
        print('Using FP16 (half precision)')
    else:
        print('Using FP32 (full precision)')

    # Prepare output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get input paths
    input_path = Path(args.input)
    if input_path.is_file():
        input_paths = [input_path]
    elif input_path.is_dir():
        input_paths = sorted(glob.glob(os.path.join(args.input, '*')))
        input_paths = [Path(p) for p in input_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
    else:
        raise ValueError(f'Input path does not exist: {input_path}')

    if not input_paths:
        raise ValueError(f'No valid images found in {input_path}')

    print(f'Processing {len(input_paths)} image(s)...\n')

    # Process each image
    for idx, img_path in enumerate(input_paths, 1):
        print(f'[{idx}/{len(input_paths)}] Processing: {img_path.name}')

        # Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f'  ERROR: Could not read image: {img_path}')
            continue

        try:
            # Enhance image
            output = enhance_image(
                img,
                model,
                scale=args.scale,
                tile_size=args.tile,
                tile_pad=args.tile_pad,
                device=device,
                half_precision=not args.fp32
            )

            # Save output
            output_path = output_dir / f'{img_path.stem}_{args.suffix}{img_path.suffix}'
            cv2.imwrite(str(output_path), output)

            print(f'  Saved to: {output_path}')
            print(f'  Input: {img.shape[1]}x{img.shape[0]} -> Output: {output.shape[1]}x{output.shape[0]}')

        except Exception as e:
            print(f'  ERROR: {str(e)}')
            import traceback
            traceback.print_exc()
            continue

    print(f'\nProcessing complete! Results saved to: {output_dir}')


if __name__ == '__main__':
    main()
