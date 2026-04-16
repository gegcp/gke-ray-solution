# GKE Ray Solution

GPU-accelerated deep learning inference scripts for image processing and classification.

## Projects

This repository contains two GPU-optimized inference solutions:

### 1. Real-ESRGAN - Image Super-Resolution
Location: `real-esrgan/gpu/`

High-quality image upscaling using Real-ESRGAN models with GPU acceleration.

**Features:**
- 2x to 4x image upscaling
- Support for multiple model variants (general, anime, video)
- GPU acceleration with FP16/FP32 precision
- Memory-efficient tiling for large images
- Batch processing support

[📖 Real-ESRGAN Documentation](./real-esrgan/gpu/README.md)

**Quick Start:**
```bash
cd real-esrgan/gpu
pip install -r requirements.txt
# Download weights (see README)
python inference.py -i image.jpg -o results
```

---

### 2. RepVGG - Image Classification
Location: `repvgg/gpu/`

Fast and accurate ImageNet classification using RepVGG models with GPU acceleration.

**Features:**
- ImageNet 1000-class classification
- Multiple model variants (A0, B0, B1, B2)
- GPU acceleration with FP16/FP32 precision
- Top-K predictions with confidence scores
- Efficient VGG-style architecture

[📖 RepVGG Documentation](./repvgg/gpu/README.md)

**Quick Start:**
```bash
cd repvgg/gpu
pip install -r requirements.txt
python download_labels.py
# Download weights (see README)
python inference.py -i image.jpg
```

---

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
- PyTorch 2.0+ with CUDA
- 4GB+ GPU VRAM (recommended)

## GPU Setup

Both projects require PyTorch with CUDA support:

```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA is available:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## Model Weights

⚠️ **Model weights are not included in this repository due to file size.**

Each project includes detailed instructions for downloading the required weights:
- [Real-ESRGAN weights instructions](./real-esrgan/gpu/README.md#setup)
- [RepVGG weights instructions](./repvgg/gpu/README.md#setup)

## Project Structure

```
gke-ray-solution/
├── README.md                   # This file
├── .gitignore                 # Git ignore rules
├── real-esrgan/gpu/           # Real-ESRGAN project
│   ├── inference.py           # Inference script
│   ├── requirements.txt       # Dependencies
│   ├── README.md             # Documentation
│   └── weights/              # Model weights (download separately)
└── repvgg/gpu/               # RepVGG project
    ├── inference.py          # Inference script
    ├── requirements.txt      # Dependencies
    ├── README.md            # Documentation
    ├── download_labels.py   # Download ImageNet labels
    └── RepVGG-*.pth        # Model weights (download separately)
```

## License

See individual project documentation for licensing information.

## References

- [Real-ESRGAN GitHub](https://github.com/xinntao/Real-ESRGAN)
- [RepVGG GitHub](https://github.com/DingXiaoH/RepVGG)
