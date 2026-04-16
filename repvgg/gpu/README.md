# RepVGG GPU Inference

Python script for running RepVGG image classification with GPU acceleration.

## About RepVGG

RepVGG is an efficient CNN architecture for image classification that achieves excellent speed-accuracy trade-off. It uses a simple VGG-style architecture during inference while maintaining high accuracy.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. **Download Model Weights**

Download the RepVGG model weights from the [official repository](https://github.com/DingXiaoH/RepVGG):

**Option A: Using wget (recommended)**

```bash
# RepVGG-A0 (72.41% ImageNet top-1 accuracy)
wget https://drive.google.com/uc?export=download&id=1HN_TNXAyJKJWCcbITxLGRVzW3T7zgVLa -O RepVGG-A0-train.pth

# RepVGG-B0 (75.14% ImageNet top-1 accuracy) - Default
wget https://drive.google.com/uc?export=download&id=1nCxqOHN2kX8Mb2XhzHEz0XYMtmPK3KbI -O RepVGG-B0-train.pth

# RepVGG-B1 (78.37% ImageNet top-1 accuracy)
wget https://drive.google.com/uc?export=download&id=1D_FoScT3ckVVhLJEbdAlO8GdzwIq4Aqr -O RepVGG-B1-train.pth

# RepVGG-B2 (78.78% ImageNet top-1 accuracy)
wget https://drive.google.com/uc?export=download&id=1U0JRJiPXyX9SHuXVqSTd9ESDPVPcWsQz -O RepVGG-B2-train.pth
```

**Option B: Manual Download**

Visit the [RepVGG GitHub repository](https://github.com/DingXiaoH/RepVGG#use-our-pretrained-models) and download the weights manually.

Available models:
- **RepVGG-A0-train.pth** - Smallest, fastest (72.41% top-1)
- **RepVGG-B0-train.pth** - Balanced (75.14% top-1) ⭐ Recommended
- **RepVGG-B1-train.pth** - Larger, more accurate (78.37% top-1)
- **RepVGG-B2-train.pth** - Largest, most accurate (78.78% top-1)

After downloading, place the .pth file in the same directory as `inference.py`:
```
repvgg/gpu/
├── inference.py
├── requirements.txt
├── RepVGG-B0-train.pth
└── (other files...)
```

3. (Optional) Download ImageNet class labels:
```bash
python download_labels.py
```

## Model Variants

- **RepVGG-A0**: Smallest, fastest model
- **RepVGG-B0**: Balanced model (default)
- **RepVGG-B1**: Larger, more accurate
- **RepVGG-B2**: Largest, most accurate

## Usage

Basic usage:
```bash
python inference.py -i image.jpg
```

### Examples

Classify an image with default settings:
```bash
python inference.py -i cat.jpg
```

Use specific model weights:
```bash
python inference.py -i image.jpg -w RepVGG-B0-train.pth
```

Specify model variant:
```bash
python inference.py -i image.jpg -m B1 -w RepVGG-B1-train.pth
```

Show top 10 predictions:
```bash
python inference.py -i image.jpg --top_k 10
```

Use FP32 precision (higher accuracy, slower):
```bash
python inference.py -i image.jpg --fp32
```

Custom image size:
```bash
python inference.py -i image.jpg --img_size 256
```

### Arguments

- `-i, --input`: Input image path (required)
- `-w, --weights`: Model weights path (default: RepVGG-B0-train.pth)
- `-m, --model`: Model variant - A0, B0, B1, B2 (default: B0)
- `--num_classes`: Number of output classes (default: 1000 for ImageNet)
- `--img_size`: Input image size (default: 224)
- `--top_k`: Show top K predictions (default: 5)
- `--fp32`: Use FP32 precision instead of FP16

## Expected Input

- Images in common formats: JPG, PNG, BMP, etc.
- Will be resized to 224x224 (or specified size)
- Automatically normalized using ImageNet statistics

## Output

The script will display:
- Device information (GPU/CPU)
- Model loading status
- Top K class predictions with probabilities

Example output:
```
Using device: cuda
GPU: NVIDIA L4
CUDA Version: 13.0

Loading model: RepVGG-B0
Weights: RepVGG-B0-train.pth
Using FP16 (half precision)

Processing image: cat.jpg

Top 5 Predictions:
------------------------------------------------------------
1. tabby_cat                   85.32%
2. tiger_cat                   12.15%
3. Egyptian_cat                 1.23%
4. lynx                         0.45%
5. Persian_cat                  0.31%

Inference complete!
```

## GPU Requirements

- NVIDIA GPU with CUDA support
- Sufficient VRAM (2GB+ recommended)
- For FP16 inference: GPU with Tensor Cores (recommended)

## Troubleshooting

**CUDA not available**: Check PyTorch CUDA installation:
```python
import torch
print(torch.cuda.is_available())
```

**Missing labels file**: Run `python download_labels.py` or use class indices

**Wrong predictions**: Make sure you're using the correct model variant that matches your weights

## Performance

RepVGG models are designed for efficiency:
- **RepVGG-B0**: ~15ms inference on GPU (224x224)
- Uses FP16 by default for faster inference
- Minimal memory footprint

## Notes

- The script supports both training and deploy model weights
- FP16 precision is used by default on GPU for faster inference
- For best accuracy, use FP32 with `--fp32` flag
