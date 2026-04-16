# Real-ESRGAN on TPU v6e

Image super-resolution using Real-ESRGAN on Google Cloud TPU v6e.

## Files

- `rrdbnet.py` - RRDBNet model architecture (standalone, no basicsr dependency)
- `run_realesrgan.py` - Simple inference script for image upscaling
- `inference_tpu.py` - Advanced tiled inference (for very large images)
- `weights/RealESRGAN_x4plus.pth` - Pretrained model weights (64MB)

## Quick Start

### 1. Basic Usage

Upscale any image 4x on TPU:

```bash
python run_realesrgan.py --input image.jpg --output upscaled.png
```

### 2. Custom Scale Factor

```bash
# 2x upscaling
python run_realesrgan.py --input image.jpg --output upscaled.png --scale 2

# 4x upscaling (default)
python run_realesrgan.py --input image.jpg --output upscaled.png --scale 4
```

### 3. CPU Testing

Test on CPU (useful for comparison):

```bash
python run_realesrgan.py --input image.jpg --output upscaled.png --cpu
```

## Example Results

### Test 1: Eva Image (350x197 → 1400x788)
```
Input size:     350 x 197
Output size:    1400 x 788
Scale factor:   4x
Inference time: 3737.04 ms (TPU v6e)
CPU time:       2176.27 ms
```

### Test 2: Burger Image (225x225 → 900x900)
```
Input size:     225 x 225
Output size:    900 x 900
Scale factor:   4x
Inference time: 2907.82 ms (TPU v6e)
```

## Command Line Arguments

### run_realesrgan.py

- `--input`: Path to input image (default: eva.jpg)
- `--output`: Path to save output image (default: output_upscaled.png)
- `--weight-path`: Path to model weights (default: weights/RealESRGAN_x4plus.pth)
- `--scale`: Upscaling factor, 2 or 4 (default: 4)
- `--cpu`: Run on CPU instead of TPU
- `--tile-size`: Tile size for large images (optional, for memory efficiency)

## Performance Notes

### Small Images (<500px)
- TPU may be slower than CPU due to compilation overhead
- First run includes ~5-6s JIT compilation time
- Subsequent runs would be faster if model stays in memory

### Medium/Large Images (>500px)
- TPU becomes more efficient with larger images
- Consider using tiled inference for very large images (>2000px)

### Tiled Inference

For very large images, use the advanced tiled script:

```bash
python inference_tpu.py --input-h 2048 --input-w 1536 --tile 512 --halo 16
```

This splits the image into overlapping tiles, processes them in batches, and stitches the result.

## Model Architecture

### RealESRGAN_x4plus Specifications

- **Architecture**: RRDB (Residual in Residual Dense Block)
- **Input channels**: 3 (RGB)
- **Output channels**: 3 (RGB)
- **Feature channels**: 64
- **Number of blocks**: 23
- **Growth channels**: 32
- **Upscaling**: 4x (two 2x nearest-neighbor upsamplers)

### Key Modifications for TPU/XLA

1. **Non-inplace operations**: All `LeakyReLU(inplace=True)` changed to `inplace=False`
2. **Custom upsampling**: Replaced `F.interpolate` with `repeat_interleave` for XLA compatibility
3. **No basicsr dependency**: Standalone implementation

## Requirements

```bash
pip install torch torchvision pillow numpy
pip install torch-xla2 jax[tpu]
```

## Technical Details

### Inference Pipeline

1. **Load model** - RRDBNet with 23 RRDB blocks
2. **Preprocess image** - Convert to float32 tensor, normalize to [0, 1]
3. **Convert to JAX** - Extract JAX weights and function from PyTorch model
4. **JIT compile** - Compile forward pass with JAX for TPU
5. **Run inference** - Execute on TPU with automatic XLA optimization
6. **Postprocess** - Convert back to image, clip to [0, 255], save as PNG

### Memory Considerations

- RealESRGAN_x4plus: ~64M parameters (~256MB memory)
- Output is 16x larger in pixels (4x in each dimension)
- For 1000x1000 input → 4000x4000 output (~64MB in memory)

## Troubleshooting

### Out of Memory

Use tiled inference for very large images:
```bash
python run_realesrgan.py --input large.jpg --output out.png --tile-size 512
```

### Slow First Run

The first inference includes JIT compilation (~5-6s). This is normal and only happens once per input shape.

### Import Errors

Make sure all dependencies are installed:
```bash
pip install torch torchvision pillow numpy torch-xla2
```

## Reference

Based on the implementation from:
- https://github.com/yangwhale/gpu-tpu-pedia/tree/main/tpu/Real-ESRGAN
- https://github.com/xinntao/Real-ESRGAN

## License

This implementation follows the BSD 3-Clause License from the original Real-ESRGAN repository.
