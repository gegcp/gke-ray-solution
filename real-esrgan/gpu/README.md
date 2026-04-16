# Real-ESRGAN GPU Inference

Python script for running Real-ESRGAN image upscaling with GPU acceleration.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure model weights are in the `weights/` folder. Common models:
   - `RealESRGAN_x4plus.pth` - General purpose 4x upscaling
   - `RealESRNet_x4plus.pth` - 4x upscaling (training model)
   - `RealESRGAN_x4plus_anime_6B.pth` - Anime images 4x
   - `RealESRGAN_x2plus.pth` - 2x upscaling
   - `realesr-animevideov3.pth` - Anime video frames
   - `realesr-general-x4v3.pth` - General purpose v3

## Usage

Basic usage:
```bash
python inference.py -i input.jpg -o output_folder
```

### Common Examples

Upscale a single image:
```bash
python inference.py -i image.jpg -o results
```

Upscale all images in a folder:
```bash
python inference.py -i input_folder/ -o results
```

Use anime model:
```bash
python inference.py -i anime.jpg -o results -m RealESRGAN_x4plus_anime_6B
```

2x upscaling instead of 4x:
```bash
python inference.py -i image.jpg -o results -m RealESRGAN_x2plus -s 2
```

For large images (use tiling to avoid OOM):
```bash
python inference.py -i large_image.jpg -o results --tile 512
```

With face enhancement:
```bash
python inference.py -i portrait.jpg -o results --face_enhance
```

Use FP32 precision (higher quality, slower):
```bash
python inference.py -i image.jpg -o results --fp32
```

### Arguments

- `-i, --input`: Input image or folder path (required)
- `-o, --output`: Output folder path (default: results)
- `-m, --model`: Model to use (default: RealESRGAN_x4plus)
- `-s, --outscale`: Output scale factor (default: 4)
- `--fp32`: Use FP32 precision instead of FP16
- `--tile`: Tile size for processing large images (0 = no tiling)
- `--tile_pad`: Tile padding (default: 10)
- `--face_enhance`: Use GFPGAN for face enhancement
- `--suffix`: Suffix for output filename (default: out)
- `--ext`: Output image extension (auto, jpg, png)

## GPU Requirements

- NVIDIA GPU with CUDA support
- Sufficient VRAM (4GB+ recommended for 4K images)
- For very large images, use `--tile 512` or `--tile 256` to reduce memory usage

## Troubleshooting

**Out of memory error**: Use smaller tile size `--tile 256` or `--tile 128`

**CUDA not available**: Check PyTorch CUDA installation with:
```python
import torch
print(torch.cuda.is_available())
```

**Slow performance**: Make sure you're using GPU, check with:
```bash
nvidia-smi
```
