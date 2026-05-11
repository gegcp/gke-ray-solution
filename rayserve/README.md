# Ray Serve on GKE with GPU and TPU

Deploy computer vision models on GPU (NVIDIA L4) and TPU (v6e) accelerators using Ray Serve on GKE.

## 🚀 Quick Start

### Available Deployments

| Deployment | Models | Accelerators | Documentation |
|------------|--------|--------------|---------------|
| **RepVGG** | Image classification | GPU + TPU | [Quick Reference](docs/quick-reference.md) |
| **Real-ESRGAN** | Image super-resolution | GPU + TPU | [Real-ESRGAN Guide](docs/realesrgan.md) |
| **Model Composition** | Upscale → Classify | TPU → GPU | [Composition Guide](docs/model-composition.md) |

### Deploy RepVGG (2 minutes)

```bash
cd rayserve

# Create ConfigMap
kubectl create configmap repvgg-apps \
  --from-file=repvgg_gpu_app.py \
  --from-file=repvgg_tpu_app.py

# Deploy RayService
kubectl apply -f repvgg-gpu-tpu-app.rayservice.yaml

# Check status
kubectl get pods -w
```

### Deploy Real-ESRGAN

```bash
bash deploy-realesrgan.sh
```

### Deploy Model Composition (Real-ESRGAN → RepVGG)

```bash
bash deploy-composition.sh
```

## 📚 Documentation

### Getting Started
- **[Quick Reference](docs/quick-reference.md)** - Fast deployment and testing
- **[Scripts Reference](docs/scripts.md)** - Available scripts and usage

### Deployment Guides
- **[Real-ESRGAN Deployment](docs/realesrgan.md)** - Super-resolution on GPU/TPU
- **[Model Composition](docs/model-composition.md)** - Chain models across accelerators
- **[TPU Setup Guide](docs/tpu-setup.md)** - TPU configuration and troubleshooting

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        RayService                           │
├─────────────────────────────────────────────────────────────┤
│  Head Node (2 CPU, 8GB)                                     │
│  ├─ Ray Dashboard (8265)                                    │
│  └─ Ray Serve (8000)                                        │
├─────────────────────────────────────────────────────────────┤
│  GPU Worker (NVIDIA L4)          │  TPU Worker (v6e)        │
│  ├─ 2 CPU, 10GB RAM              │  ├─ 1 CPU, 20GB RAM     │
│  ├─ 1 GPU                         │  ├─ 1 TPU               │
│  └─ GPU Deployment                │  └─ TPU Deployment      │
│     • RepVGG: /gpu/classify       │     • RepVGG: /tpu/...  │
│     • Real-ESRGAN: /gpu/upscale   │     • Real-ESRGAN: ...  │
└─────────────────────────────────────────────────────────────┘
```

## 🔑 Key Features

- **Multi-Accelerator**: Deploy on GPU and TPU in single RayService
- **Model Composition**: Chain models across different accelerators
- **Auto-scaling**: Scale replicas based on load
- **Unified API**: Single endpoint for all models
- **Production-Ready**: Health checks, monitoring, graceful shutdown

## 📊 Model Capabilities

### RepVGG (Image Classification)
- **Models**: RepVGG-A0, RepVGG-B0
- **Classes**: 1000 ImageNet categories
- **Input**: 224×224 RGB images
- **Output**: Top-K predictions with probabilities

### Real-ESRGAN (Super-Resolution)
- **Models**: RealESRGAN_x4plus, RealESRGAN_x4plus_anime_6B
- **Scale**: 2x, 4x upscaling
- **Features**: Tile processing, FP16 precision
- **Input**: Any resolution RGB image
- **Output**: Upscaled high-resolution image

### Model Composition
- **Pipeline**: Image → Real-ESRGAN (TPU) → RepVGG (GPU) → Classification
- **Use Case**: Classify low-resolution images with enhanced accuracy
- **Endpoint**: `POST /classify_upscaled`

## 🧪 Testing

### Quick Health Check

```bash
# Get head pod
HEAD_POD=$(kubectl get pod -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

# Check GPU endpoint
kubectl exec $HEAD_POD -- curl -s http://localhost:8000/gpu/health

# Check TPU endpoint
kubectl exec $HEAD_POD -- curl -s http://localhost:8000/tpu/health
```

### Run Inference Tests

```bash
# Test RepVGG
python test_repvgg_apps.py <image.jpg>

# Test Real-ESRGAN
python test_realesrgan_apps.py <image.jpg>

# Test Model Composition
python test_composition.py <image.jpg>
```

## 📦 Files Structure

```
rayserve/
├── README.md                                    # This file
├── docs/                                        # Documentation
│   ├── quick-reference.md                       # Quick start guide
│   ├── realesrgan.md                           # Real-ESRGAN deployment
│   ├── model-composition.md                     # Model composition guide
│   ├── tpu-setup.md                            # TPU setup and troubleshooting
│   └── scripts.md                              # Scripts reference
├── Deployments
│   ├── repvgg-gpu-tpu-app.rayservice.yaml      # RepVGG deployment
│   ├── realesrgan-gpu-tpu-app.rayservice.yaml  # Real-ESRGAN deployment
│   └── composition-esrgan-repvgg.rayservice.yaml # Composition deployment
├── Applications
│   ├── repvgg_gpu_app.py                       # RepVGG GPU inference
│   ├── repvgg_tpu_app.py                       # RepVGG TPU inference
│   ├── realesrgan_gpu_app.py                   # Real-ESRGAN GPU
│   ├── realesrgan_tpu_app.py                   # Real-ESRGAN TPU
│   └── composition_esrgan_repvgg_app.py        # Model composition
├── Backends (for composition)
│   ├── repvgg_gpu_backend.py                   # RepVGG backend
│   └── realesrgan_tpu_backend.py               # Real-ESRGAN backend
├── Scripts
│   ├── deploy-realesrgan.sh                    # Deploy Real-ESRGAN
│   └── deploy-composition.sh                   # Deploy composition
└── Tests
    ├── test_repvgg_apps.py                     # RepVGG tests
    ├── test_realesrgan_apps.py                 # Real-ESRGAN tests
    └── test_composition.py                     # Composition tests
```

## 🔧 Prerequisites

- GKE cluster with GPU and TPU node pools
- kubectl configured
- Ray Operator installed
- Container images built and pushed

See [TPU Setup Guide](docs/tpu-setup.md) for detailed setup instructions.

## 📖 API Reference

### RepVGG Classification

**Endpoint**: `POST /gpu/classify` or `POST /tpu/classify`

```json
{
  "image": "<base64_encoded_image>",
  "top_k": 5,
  "img_size": 224
}
```

### Real-ESRGAN Upscaling

**Endpoint**: `POST /gpu/upscale` or `POST /tpu/upscale`

```json
{
  "image": "<base64_encoded_image>",
  "return_image": true
}
```

### Model Composition

**Endpoint**: `POST /classify_upscaled`

```json
{
  "image": "<base64_encoded_image>",
  "top_k": 5,
  "return_upscaled_image": false
}
```

## 🐛 Troubleshooting

Common issues and solutions:

| Issue | Solution |
|-------|----------|
| Pods not starting | Check node selectors and available resources |
| TPU not detected | See [TPU Setup Guide](docs/tpu-setup.md) |
| Out of memory | Reduce batch size or enable tiling |
| Slow inference | Check GPU/TPU utilization, enable FP16 |

For detailed troubleshooting, see the [TPU Setup Guide](docs/tpu-setup.md).

## 📝 License

Copyright 2024 Google LLC. Licensed under Apache 2.0.
