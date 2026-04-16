# RepVGG on TPU v6e

This directory contains scripts to run RepVGG models on Google Cloud TPU v6e using PyTorch and torchax.

## Files

- `repvgg.py` - RepVGG model implementation with all variants (A0-A2, B0-B3, D2se)
- `se_block.py` - Squeeze-and-Excitation block implementation
- `run_repvgg_tpu.py` - Main TPU inference script with benchmarking
- `simple_inference.py` - Simple CPU inference example
- `weights/` - Directory containing pretrained model weights

## Requirements

### For CPU/GPU inference:
```bash
pip install torch torchvision
```

### For TPU inference:
```bash
pip install torch torchvision
pip install torch-xla2  # torchax for TPU support
pip install jax[tpu]
```

## Quick Start

### 1. Simple CPU Inference

Run a quick test on CPU to verify the model loads correctly:

```bash
python simple_inference.py
```

### 2. TPU Inference with Benchmarking

Run inference on TPU v6e with performance metrics:

```bash
python run_repvgg_tpu.py
```

### 3. Custom Configuration

```bash
# Custom batch size and image size
python run_repvgg_tpu.py --batch-size 8 --image-size 224

# More benchmark iterations
python run_repvgg_tpu.py --iterations 200 --warmup 20

# Custom weight path
python run_repvgg_tpu.py --weight-path /path/to/weights.pth
```

## Command Line Arguments

### run_repvgg_tpu.py

- `--weight-path`: Path to pretrained weights (default: `weights/RepVGG-B0-train.pth`)
- `--batch-size`: Batch size for inference (default: 1)
- `--image-size`: Input image size in pixels (default: 224)
- `--iterations`: Number of benchmark iterations (default: 100)
- `--warmup`: Number of warmup iterations (default: 10)
- `--deploy`: Use deploy mode with fused branches (default: True)

## Model Weights

The pretrained weights are located at:
```
/home/gech/gke-ray-solution/repvgg/tpu/weights/RepVGG-B0-train.pth
```

These weights are in "train mode" format and will be automatically converted to "deploy mode" for efficient inference.

## RepVGG Variants

The `repvgg.py` module supports multiple model variants:

### A-Series (Lightweight)
- `create_RepVGG_A0()` - Smallest model
- `create_RepVGG_A1()`
- `create_RepVGG_A2()`

### B-Series (Standard)
- `create_RepVGG_B0()` - Used in this example
- `create_RepVGG_B1()`
- `create_RepVGG_B2()`
- `create_RepVGG_B3()` - Largest standard model

### B-Series with Grouped Convolutions
- `create_RepVGG_B1g2()`
- `create_RepVGG_B1g4()`
- `create_RepVGG_B2g2()`
- `create_RepVGG_B2g4()`
- `create_RepVGG_B3g2()`
- `create_RepVGG_B3g4()`

### D-Series (with SE blocks)
- `create_RepVGG_D2se()` - With Squeeze-and-Excitation blocks

## Training vs Deploy Mode

RepVGG uses **structural reparameterization**:

- **Training mode** (`deploy=False`): Uses multi-branch architecture (3x3 conv + 1x1 conv + identity) for better optimization
- **Deploy mode** (`deploy=True`): Fuses all branches into a single 3x3 convolution for faster inference

The scripts automatically handle this conversion.

## Performance

Expected performance on TPU v6e:
- Single image inference: ~1-5ms latency
- Batch processing: Scales with batch size
- Significant speedup compared to CPU/GPU

Actual performance will be reported by the benchmark script.

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'torch_xla2'`:
```bash
pip install torch-xla2
```

### JAX/TPU Issues

Ensure you're running on a TPU instance:
```bash
python -c "import jax; print(jax.devices())"
```

Should show TPU devices, not CPU.

### Weight Loading Issues

If weights fail to load, check:
1. File exists: `ls -lh weights/RepVGG-B0-train.pth`
2. File is not corrupted
3. PyTorch version compatibility

## Reference

Based on the implementation from:
https://github.com/yangwhale/gpu-tpu-pedia/tree/main/tpu/RepVGG

## License

This implementation follows the MIT License from the original RepVGG repository.
