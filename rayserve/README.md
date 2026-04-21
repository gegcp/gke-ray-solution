# RepVGG GPU and TPU Inference on Ray Serve

Deploy RepVGG image classification models on both GPU (NVIDIA L4) and TPU (v6e) accelerators in a single RayService on GKE.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Deployment Guide](#deployment-guide)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

---

## Quick Start

Get up and running in 5 minutes.

### 1. Deploy (2 minutes)

```bash
cd /home/gech/workspace/oppo/gke-ray-solution/rayserve

# Create ConfigMap with application code
kubectl create configmap repvgg-apps \
  --from-file=repvgg_gpu_app.py=repvgg_gpu_app.py \
  --from-file=repvgg_tpu_app.py=repvgg_tpu_app.py \
  -n default

# Deploy RayService
kubectl apply -f gpu-tpu-app.rayservice.yaml

# Wait for pods (takes ~2 minutes)
kubectl get pods -n default -w | grep gpu-tpu-mix
```

**Wait for all pods to show `1/1 Running`**

### 2. Verify (30 seconds)

```bash
# Get head pod
HEAD_POD=$(kubectl get pod -n default -l ray.io/node-type=head,ray.io/cluster | grep gpu-tpu-mix | awk '{print $1}')

# Check status (should show RUNNING/HEALTHY)
kubectl exec -n default $HEAD_POD -- ray serve status
```

Expected:
```yaml
applications:
  gpu-app: 
    status: RUNNING
    deployments:
      RepVGGDeployment:
        status: HEALTHY
  tpu-app:
    status: RUNNING  
    deployments:
      RepVGGDeployment:
        status: HEALTHY
```

### 3. Test (1 minute)

```bash
# Quick health check
kubectl exec -n default $HEAD_POD -- \
  python3 -c "import requests; \
  print('GPU:', requests.get('http://localhost:8000/gpu/health').json()); \
  print('TPU:', requests.get('http://localhost:8000/tpu/health').json())"
```

Expected output:
```
GPU: {'status': 'healthy', 'model': 'RepVGG', 'device': 'cuda', 'precision': 'FP16'}
TPU: {'status': 'healthy', 'model': 'RepVGG-TPU', 'device': 'TPU (JAX/XLA)', 'tpu_enabled': true}
```

### 4. Run Inference Test (Optional)

```bash
# Run automated test script
./test-inference.sh
```

**That's it! 🎉** Your GPU and TPU inference apps are running!

- **GPU endpoint**: `http://localhost:8000/gpu/classify`
- **TPU endpoint**: `http://localhost:8000/tpu/classify`

---

## Architecture

### Overview

```
┌─────────────────────────────────────────────────────────┐
│                    RayService                            │
│                   gpu-tpu-mix                            │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
    ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
    │  Head Pod │  │ GPU Worker│  │ TPU Worker│
    │   (CPU)   │  │ (L4 GPU)  │  │(TPU v6e)  │
    └───────────┘  └───────────┘  └───────────┘
          │               │               │
    ┌─────▼───────────────▼───────────────▼─────┐
    │         ConfigMap: repvgg-apps             │
    │  - repvgg_gpu_app.py (GPU inference)       │
    │  - repvgg_tpu_app.py (TPU inference)       │
    └────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
    ┌─────▼─────┐                   ┌─────▼─────┐
    │  /gpu/*   │                   │  /tpu/*   │
    │  GPU App  │                   │  TPU App  │
    └───────────┘                   └───────────┘
```

### Components

- **RayService**: `gpu-tpu-mix` - Single RayService with dual-accelerator support
- **GPU App**: RepVGG inference on NVIDIA L4 GPU with FP16 precision
- **TPU App**: RepVGG inference on TPU v6e with JAX/torch-xla2
- **Deployment**: ConfigMap-based code deployment (no GitHub dependency)

### Resource Allocation

| Component | CPU | Memory | Accelerator | Storage |
|-----------|-----|--------|-------------|---------|
| Head Node | 2 cores | 8Gi | - | 20Gi |
| GPU Worker | 4 cores | 20Gi | 1x NVIDIA L4 | 20Gi |
| TPU Worker | 1 core | 20Gi | 1x TPU v6e | 40Gi |

---

## Prerequisites

### Required

- **GKE Cluster** with:
  - GPU node pool (NVIDIA L4)
  - TPU node pool (TPU v6e)
  - GKE version 1.28+ (for TPU support without privileged mode)
- **kubectl** configured to access your cluster
- **Ray Operator** installed in the cluster
- **Custom Docker image** with JAX 0.4.35+ and torch-xla2 (for TPU acceleration)

### Optional

- HuggingFace secret for model weights: `hf-secret`

### Verify Prerequisites

```bash
# Check node pools
kubectl get nodes --show-labels | grep -E "nvidia-l4|tpu-v6e"

# Check Ray Operator
kubectl get deployment -n ray-system kuberay-operator

# Check HF secret (optional)
kubectl get secret hf-secret -n default

# Verify GKE version (should be 1.28+)
kubectl version --short
```

---

## Deployment Guide

### Method 1: Automated Deployment (Recommended)

Use the deployment automation script:

```bash
cd /home/gech/workspace/oppo/gke-ray-solution/rayserve

# Full deployment with rebuild
./deploy-tpu-rayservice.sh --rebuild --clean

# Or deploy with existing image
./deploy-tpu-rayservice.sh
```

See [SCRIPTS_README.md](./SCRIPTS_README.md) for detailed script documentation.

### Method 2: Manual Deployment

#### Step 1: Build Custom TPU Image (if needed)

```bash
# Clean Docker cache and build
./build-tpu-image.sh --clean

# Push to Artifact Registry
./push-tpu-image.sh
```

See [TPU_SETUP_GUIDE.md](./TPU_SETUP_GUIDE.md) for detailed TPU setup instructions.

#### Step 2: Create ConfigMap

The GPU and TPU inference apps are stored in a ConfigMap and mounted into Ray pods.

```bash
# Create ConfigMap from Python files
kubectl create configmap repvgg-apps \
  --from-file=repvgg_gpu_app.py=repvgg_gpu_app.py \
  --from-file=repvgg_tpu_app.py=repvgg_tpu_app.py \
  -n default \
  --dry-run=client -o yaml | kubectl apply -f -

# Verify ConfigMap
kubectl get configmap repvgg-apps -n default
kubectl describe configmap repvgg-apps -n default
```

#### Step 3: Deploy RayService

```bash
# Apply RayService configuration
kubectl apply -f gpu-tpu-app.rayservice.yaml

# Verify RayService creation
kubectl get rayservice gpu-tpu-mix -n default
```

#### Step 4: Monitor Deployment

```bash
# Watch RayService status
kubectl get rayservice gpu-tpu-mix -n default -w

# Check pods
kubectl get pods -n default -l ray.io/cluster

# Wait for all pods to be Running (1/1)
# - Head node: 1 pod
# - GPU worker: 1 pod  
# - TPU worker: 1 pod
```

Expected pod status:
```
NAME                                       READY   STATUS    RESTARTS   AGE
gpu-tpu-mix-xxxxx-head-xxxxx               1/1     Running   0          2m
gpu-tpu-mix-xxxxx-gpu-group-worker-xxxxx   1/1     Running   0          2m
gpu-tpu-mix-xxxxx-tpu-group-worker-xxxxx   1/1     Running   0          2m
```

#### Step 5: Verify Serve Applications

```bash
# Get head pod name
HEAD_POD=$(kubectl get pod -n default -l ray.io/node-type=head | grep gpu-tpu-mix | awk '{print $1}')

# Check Ray Serve status
kubectl exec -n default $HEAD_POD -- ray serve status
```

---

## Testing

### Quick Health Check

```bash
HEAD_POD=$(kubectl get pod -n default -l ray.io/node-type=head | grep gpu-tpu-mix | awk '{print $1}')

# Test GPU app
kubectl exec -n default $HEAD_POD -- \
  curl -s http://localhost:8000/gpu/health | python3 -m json.tool

# Test TPU app
kubectl exec -n default $HEAD_POD -- \
  curl -s http://localhost:8000/tpu/health | python3 -m json.tool
```

### Automated Testing

Run the comprehensive test script:

```bash
./test-inference.sh
```

This script will:
- Create a test image
- Test both GPU and TPU endpoints
- Show latency comparison
- Display performance metrics

### Manual Inference Test

Test with your own image:

```bash
# Encode test image
IMAGE_BASE64=$(base64 -w 0 /path/to/your/image.jpg)

# Test GPU inference
curl -X POST http://localhost:8000/gpu/classify \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"${IMAGE_BASE64}\", \"top_k\": 5}"

# Test TPU inference
curl -X POST http://localhost:8000/tpu/classify \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"${IMAGE_BASE64}\", \"top_k\": 5}"
```

### Direct Pod Testing (No Port-Forward)

Test from inside the cluster:

```bash
HEAD_POD=$(kubectl get pod -n default -l ray.io/node-type=head | grep gpu-tpu-mix | awk '{print $1}')

# Create and run test inside pod
kubectl exec -n default $HEAD_POD -- python3 << 'EOF'
import requests
import base64
import numpy as np
from PIL import Image
from io import BytesIO

# Generate random test image
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
buffer = BytesIO()
img.save(buffer, format='PNG')
img_b64 = base64.b64encode(buffer.getvalue()).decode()

# Test GPU endpoint
gpu_response = requests.post(
    'http://localhost:8000/gpu/classify',
    json={'image': img_b64, 'top_k': 3}
)
print("GPU:", gpu_response.json())

# Test TPU endpoint
tpu_response = requests.post(
    'http://localhost:8000/tpu/classify',
    json={'image': img_b64, 'top_k': 3}
)
print("TPU:", tpu_response.json())
EOF
```

---

## API Reference

### Health Endpoints

#### GPU Health Check
```
GET /gpu/health
```

**Response**:
```json
{
  "status": "healthy",
  "model": "RepVGG",
  "device": "cuda",
  "precision": "FP16"
}
```

#### TPU Health Check
```
GET /tpu/health
```

**Response**:
```json
{
  "status": "healthy",
  "model": "RepVGG-TPU",
  "device": "TPU (JAX/XLA)",
  "tpu_enabled": true,
  "jax_backend": "tpu",
  "jax_device_count": 1
}
```

### Inference Endpoints

#### GPU Inference
```
POST /gpu/classify
Content-Type: application/json
```

**Request**:
```json
{
  "image": "<base64-encoded-image>",
  "top_k": 5,
  "img_size": 224
}
```

**Response**:
```json
{
  "predictions": [
    {
      "class_id": 281,
      "probability": 0.15234
    },
    {
      "class_id": 385,
      "probability": 0.12456
    }
  ],
  "model": "RepVGG",
  "device": "cuda"
}
```

#### TPU Inference
```
POST /tpu/classify
Content-Type: application/json
```

Same request/response format as GPU endpoint, with `"device": "TPU (JAX/XLA)"`.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | Yes | - | Base64-encoded image (JPEG, PNG, etc.) |
| `top_k` | integer | No | 5 | Number of top predictions to return |
| `img_size` | integer | No | 224 | Input image size (resized to img_size × img_size) |

### Error Responses

**400 Bad Request** - Missing or invalid image:
```json
{
  "error": "No image provided"
}
```

**500 Internal Server Error** - Inference failure:
```json
{
  "error": "Inference failed: <error details>"
}
```

---

## Monitoring

### Ray Dashboard

```bash
# Port forward Ray dashboard
HEAD_POD=$(kubectl get pod -n default -l ray.io/node-type=head | grep gpu-tpu-mix | awk '{print $1}')
kubectl port-forward -n default $HEAD_POD 8265:8265

# Open in browser: http://localhost:8265
```

### Check Logs

```bash
# Get pod names
kubectl get pods -n default -l ray.io/cluster

# Head pod logs
kubectl logs -n default <head-pod-name> --tail=100

# GPU worker logs
kubectl logs -n default <gpu-worker-pod-name> --tail=100

# TPU worker logs
kubectl logs -n default <tpu-worker-pod-name> --tail=100

# Serve controller logs
kubectl exec -n default $HEAD_POD -- \
  cat /tmp/ray/session_latest/logs/serve/controller*.log | tail -100
```

### Ray Cluster Status

```bash
# Check Ray cluster resources
kubectl exec -n default $HEAD_POD -- ray status

# Check serve configuration
kubectl exec -n default $HEAD_POD -- ray serve status
```

### Monitor Resource Usage

```bash
# GPU utilization
GPU_POD=$(kubectl get pod -n default -l ray.io/group-name=gpu-group -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n default $GPU_POD -- nvidia-smi

# Pod resources
kubectl top pods -n default -l ray.io/cluster

# TPU verification
TPU_POD=$(kubectl get pod -n default -l ray.io/group-name=tpu-group -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n default $TPU_POD -- python3 -c "import jax; print('JAX devices:', jax.devices())"
```

---

## Troubleshooting

### Pods Not Starting

```bash
# Check events
kubectl describe rayservice gpu-tpu-mix -n default

# Check pod events
kubectl describe pod <pod-name> -n default

# Common causes:
# - Node pool not available
# - Resource constraints
# - Image pull errors
```

### Serve Applications Stuck in DEPLOYING

```bash
# Check serve status
kubectl exec -n default $HEAD_POD -- ray serve status

# Check if dependencies are installing
kubectl logs -n default <worker-pod> --tail=200

# Common causes:
# - pip install timeout
# - Python import errors  
# - Resource constraints
```

### TPU Not Detected

```bash
# Check JAX version (must be 0.4.35+ for TPU v6e)
kubectl exec -n default $TPU_POD -- python3 -c "import jax; print(jax.__version__)"

# Check TPU devices
kubectl exec -n default $TPU_POD -- python3 -c "import jax; print(jax.devices())"

# Expected: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]

# Check environment
kubectl exec -n default $TPU_POD -- env | grep -i tpu
```

See [TPU_SETUP_GUIDE.md](./TPU_SETUP_GUIDE.md) for comprehensive TPU troubleshooting.

### Health Endpoint Returns Error

```bash
# Check if deployment is healthy
kubectl exec -n default $HEAD_POD -- ray serve status

# Check application logs
kubectl logs -n default <worker-pod> | grep -A 10 "RepVGG"

# Test import manually
kubectl exec -n default <worker-pod> -- \
  python3 -c "import sys; sys.path.append('/rayserve/apps'); import repvgg_gpu_app"
```

### ConfigMap Not Mounted

```bash
# Check if ConfigMap exists
kubectl get configmap repvgg-apps -n default

# Check mount in pod
kubectl exec -n default <pod-name> -- ls -la /rayserve/apps/

# Should show:
# repvgg_gpu_app.py
# repvgg_tpu_app.py
```

---

## Advanced Topics

### Updating Deployment

#### Update Application Code

```bash
# Edit Python files
vim repvgg_gpu_app.py
vim repvgg_tpu_app.py

# Update ConfigMap
kubectl create configmap repvgg-apps \
  --from-file=repvgg_gpu_app.py \
  --from-file=repvgg_tpu_app.py \
  -n default \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart RayService to pick up changes
kubectl delete rayservice gpu-tpu-mix -n default
kubectl apply -f gpu-tpu-app.rayservice.yaml
```

#### Update RayService Configuration

```bash
# Edit YAML file
vim gpu-tpu-app.rayservice.yaml

# Apply changes
kubectl apply -f gpu-tpu-app.rayservice.yaml

# Monitor rollout
kubectl get rayservice gpu-tpu-mix -n default -w
```

### Environment Variables

Configure applications via environment variables in the RayService YAML:

**GPU App**:
- `MODEL_VARIANT`: RepVGG model variant (A0, B0) - default: B0
- `NUM_CLASSES`: Number of output classes - default: 1000
- `USE_FP16`: Enable FP16 precision - default: true

**TPU App**:
- `MODEL_VARIANT`: RepVGG model variant (A0, B0) - default: B0
- `NUM_CLASSES`: Number of output classes - default: 1000
- `ENABLE_TPU`: Enable TPU acceleration - default: true
- `TPU_LIBRARY_PATH`: Path to libtpu.so library

### Performance Tuning

**GPU App** (NVIDIA L4, FP16):
- Cold start: ~10-15s (first request)
- Warm latency: ~5-20ms per image
- Throughput: ~50-200 images/sec
- Memory: ~2-4 GB GPU memory

**TPU App** (TPU v6e with JAX):
- Cold start: ~10-15s (first request)
- Warm latency: ~5-20ms per image
- Throughput: ~50-200 images/sec
- **10-50x faster** than CPU mode

---

## Cleanup

```bash
# Delete RayService (deletes all associated resources)
kubectl delete rayservice gpu-tpu-mix -n default

# Delete ConfigMap
kubectl delete configmap repvgg-apps -n default

# Verify cleanup
kubectl get pods -n default -l ray.io/cluster
kubectl get svc -n default | grep gpu-tpu-mix
```

---

## Files Overview

```
gke-ray-solution/rayserve/
├── README.md                      # This file - comprehensive guide
├── QUICK_REFERENCE.md             # Quick reference card
├── TPU_SETUP_GUIDE.md             # Detailed TPU setup and troubleshooting
├── SCRIPTS_README.md              # Script documentation
├── gpu-tpu-app.rayservice.yaml    # RayService definition
├── Dockerfile.ray-tpu             # Custom Docker image for TPU
├── repvgg_gpu_app.py              # GPU inference application
├── repvgg_tpu_app.py              # TPU inference application
├── build-tpu-image.sh             # Build Docker image
├── push-tpu-image.sh              # Push image to registry
├── deploy-tpu-rayservice.sh       # Automated deployment
└── test-inference.sh              # Test GPU and TPU inference
```

---

## Additional Resources

- **TPU Setup Guide**: [TPU_SETUP_GUIDE.md](./TPU_SETUP_GUIDE.md)
- **Quick Reference**: [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
- **Script Documentation**: [SCRIPTS_README.md](./SCRIPTS_README.md)
- **Ray Serve Docs**: https://docs.ray.io/en/latest/serve/
- **KubeRay Docs**: https://docs.ray.io/en/latest/cluster/kubernetes/
- **GKE TPU Docs**: https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus
- **RepVGG Paper**: https://arxiv.org/abs/2101.03697

---

**Version**: 2.0  
**Last Updated**: April 21, 2026  
**Status**: Production-ready with TPU v6e acceleration ✅
