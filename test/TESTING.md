# Testing RepVGG GPU and TPU Apps

Comprehensive testing guide for the RepVGG inference applications.

## Quick Start

```bash
cd /home/gech/workspace/oppo/gke-ray-solution/test

# Ensure RayService is deployed and running
kubectl get rayservice gpu-tpu-mix -n default

# Run automated tests
python3 test_repvgg_apps.py
```

## Prerequisites

- RayService `gpu-tpu-mix` is deployed and running
- All pods are in `Running` state with `1/1` ready
- Serve applications are in `RUNNING` status

### Verify Prerequisites

```bash
# Check RayService status
kubectl get rayservice gpu-tpu-mix -n default

# Check pods
kubectl get pods -n default | grep gpu-tpu-mix

# All should show 1/1 Running:
# - gpu-tpu-mix-xxxxx-head-xxxxx
# - gpu-tpu-mix-xxxxx-gpu-group-worker-xxxxx
# - gpu-tpu-mix-xxxxx-tpu-group-worker-xxxxx

# Check Serve status
HEAD_POD=$(kubectl get pod -n default -l ray.io/node-type=head,ray.io/cluster | grep gpu-tpu-mix | awk '{print $1}')
kubectl exec -n default $HEAD_POD -- serve status
```

Expected Serve status:
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

## Testing Methods

### Method 1: Automated Python Test (Recommended)

The automated test script tests both GPU and TPU endpoints with a sample image.

```bash
cd /home/gech/workspace/oppo/gke-ray-solution/test

# Setup port-forward
kubectl port-forward -n default svc/$(kubectl get svc -n default | grep gpu-tpu-mix | grep head-svc | awk '{print $1}') 8000:8000 &

# Wait for port-forward to establish
sleep 2

# Run tests with default image
python3 test_repvgg_apps.py

# Or test with custom image
python3 test_repvgg_apps.py /path/to/your/image.jpg
```

**Expected Output:**
```
############################################################
# RepVGG Inference Test
############################################################

Base URL: http://localhost:8000
Test Image: /home/gech/workspace/oppo/gke-ray-solution/repvgg/gpu/burger.jpeg


************************************************************
* GPU INFERENCE TEST
************************************************************

============================================================
Testing: http://localhost:8000/gpu/classify
Image: burger.jpeg
============================================================

✓ Health check: {'status': 'healthy', 'model': 'RepVGG', 'device': 'cuda', 'precision': 'FP16'}

Encoding image...
Sending inference request...

✓ Inference successful!
Model: RepVGG
Device: cuda

Top 5 Predictions:
------------------------------------------------------------
1. Class ID:  304  Probability:   0.10%
2. Class ID:  690  Probability:   0.10%
...


************************************************************
* TPU INFERENCE TEST
************************************************************

✓ Health check: {'status': 'healthy', 'model': 'RepVGG-TPU', 'device': 'TPU-node (CPU mode)'}
...

============================================================
Testing complete!
============================================================
```

### Method 2: Bash/curl Test Script

Simple shell script using curl for testing.

```bash
cd /home/gech/workspace/oppo/gke-ray-solution/test

# Setup port-forward (if not already done)
kubectl port-forward -n default svc/$(kubectl get svc -n default | grep gpu-tpu-mix | grep head-svc | awk '{print $1}') 8000:8000 &

# Run curl tests
./test_curl.sh

# Or with custom image
./test_curl.sh /path/to/your/image.jpg
```

### Method 3: Direct Pod Testing (No Port-Forward)

Test directly from inside the cluster without port-forwarding.

```bash
# Get head pod name
HEAD_POD=$(kubectl get pod -n default -l ray.io/node-type=head,ray.io/cluster | grep gpu-tpu-mix | awk '{print $1}')

# Test GPU health
kubectl exec -n default $HEAD_POD -- \
  python3 -c "import requests; print('GPU Health:', requests.get('http://localhost:8000/gpu/health').json())"

# Test TPU health
kubectl exec -n default $HEAD_POD -- \
  python3 -c "import requests; print('TPU Health:', requests.get('http://localhost:8000/tpu/health').json())"

# Test GPU inference with sample data
kubectl exec -n default $HEAD_POD -- python3 << 'EOF'
import requests
import base64

# Create a small test image (3x224x224 RGB)
import numpy as np
from PIL import Image
import io

# Generate random test image
img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(img_array)

# Convert to base64
buffer = io.BytesIO()
img.save(buffer, format='JPEG')
img_b64 = base64.b64encode(buffer.getvalue()).decode()

# Test GPU endpoint
response = requests.post(
    'http://localhost:8000/gpu/classify',
    json={'image': img_b64, 'top_k': 3}
)
print("GPU Response:", response.json())

# Test TPU endpoint
response = requests.post(
    'http://localhost:8000/tpu/classify',
    json={'image': img_b64, 'top_k': 3}
)
print("TPU Response:", response.json())
EOF
```

### Method 4: Manual curl Commands

For fine-grained testing and debugging.

```bash
# Setup port-forward
kubectl port-forward -n default svc/$(kubectl get svc -n default | grep gpu-tpu-mix | grep head-svc | awk '{print $1}') 8000:8000 &

# Test health endpoints
echo "=== GPU Health ==="
curl -s http://localhost:8000/gpu/health | python3 -m json.tool

echo "=== TPU Health ==="
curl -s http://localhost:8000/tpu/health | python3 -m json.tool

# Prepare test image
IMAGE_PATH="/home/gech/workspace/oppo/gke-ray-solution/repvgg/gpu/burger.jpeg"
IMAGE_BASE64=$(base64 -w 0 "$IMAGE_PATH")

# Test GPU inference
echo "=== GPU Inference ==="
curl -s -X POST http://localhost:8000/gpu/classify \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_BASE64}\",
    \"top_k\": 5,
    \"img_size\": 224
  }" | python3 -m json.tool

# Test TPU inference
echo "=== TPU Inference ==="
curl -s -X POST http://localhost:8000/tpu/classify \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_BASE64}\",
    \"top_k\": 5,
    \"img_size\": 224
  }" | python3 -m json.tool
```

### Method 5: Python Script (Custom)

Create your own test script:

```python
#!/usr/bin/env python3
import requests
import base64
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
IMAGE_PATH = "/path/to/your/image.jpg"

# Read and encode image
with open(IMAGE_PATH, 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# Test GPU endpoint
print("Testing GPU endpoint...")
gpu_response = requests.post(
    f"{BASE_URL}/gpu/classify",
    json={
        "image": image_base64,
        "top_k": 5,
        "img_size": 224
    },
    timeout=30
)
print(f"GPU Status: {gpu_response.status_code}")
print(f"GPU Result: {gpu_response.json()}")

# Test TPU endpoint
print("\nTesting TPU endpoint...")
tpu_response = requests.post(
    f"{BASE_URL}/tpu/classify",
    json={
        "image": image_base64,
        "top_k": 5,
        "img_size": 224
    },
    timeout=30
)
print(f"TPU Status: {tpu_response.status_code}")
print(f"TPU Result: {tpu_response.json()}")
```

## API Reference

### Health Check Endpoints

#### GPU Health
```bash
GET /gpu/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "RepVGG",
  "device": "cuda",
  "precision": "FP16"
}
```

#### TPU Health
```bash
GET /tpu/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "RepVGG-TPU",
  "device": "TPU-node (CPU mode)",
  "note": "Running on TPU node, CPU inference (add torch-xla2 for TPU acceleration)"
}
```

### Inference Endpoints

#### GPU Inference
```bash
POST /gpu/classify
Content-Type: application/json
```

**Request Body:**
```json
{
  "image": "<base64-encoded-image-string>",
  "top_k": 5,
  "img_size": 224
}
```

**Response:**
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
```bash
POST /tpu/classify
Content-Type: application/json
```

Same request/response format as GPU endpoint, but with `"device": "TPU-node (CPU mode)"`.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | Yes | - | Base64-encoded image (JPEG, PNG, etc.) |
| `top_k` | integer | No | 5 | Number of top predictions to return |
| `img_size` | integer | No | 224 | Input image size (will be resized to img_size x img_size) |

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

## Performance Testing

### Measure Latency

```bash
# Test GPU latency (10 requests)
for i in {1..10}; do
  time curl -s -X POST http://localhost:8000/gpu/classify \
    -H "Content-Type: application/json" \
    -d "{\"image\": \"${IMAGE_BASE64}\", \"top_k\": 1}" \
    > /dev/null
done

# Test TPU latency (10 requests)
for i in {1..10}; do
  time curl -s -X POST http://localhost:8000/tpu/classify \
    -H "Content-Type: application/json" \
    -d "{\"image\": \"${IMAGE_BASE64}\", \"top_k\": 1}" \
    > /dev/null
done
```

### Concurrent Requests

```bash
# Install parallel if needed: sudo apt-get install parallel

# Create request file
echo "${IMAGE_BASE64}" > /tmp/image.b64

# Run 10 concurrent requests to GPU
seq 10 | parallel -j 10 \
  "curl -s -X POST http://localhost:8000/gpu/classify \
   -H 'Content-Type: application/json' \
   -d '{\"image\": \"$(cat /tmp/image.b64)\", \"top_k\": 1}'"

# Run 10 concurrent requests to TPU  
seq 10 | parallel -j 10 \
  "curl -s -X POST http://localhost:8000/tpu/classify \
   -H 'Content-Type: application/json' \
   -d '{\"image\": \"$(cat /tmp/image.b64)\", \"top_k\": 1}'"
```

## Monitoring During Tests

### Watch Ray Serve Status

```bash
# In a separate terminal
watch -n 2 "kubectl exec -n default $HEAD_POD -- serve status"
```

### Monitor Resource Usage

```bash
# GPU utilization
kubectl exec -n default <gpu-worker-pod> -- nvidia-smi

# Watch pod resources
kubectl top pods -n default | grep gpu-tpu-mix

# Ray cluster resources
kubectl exec -n default $HEAD_POD -- ray status
```

### View Real-time Logs

```bash
# GPU worker logs
kubectl logs -f -n default <gpu-worker-pod>

# TPU worker logs
kubectl logs -f -n default <tpu-worker-pod>

# Head/controller logs
kubectl logs -f -n default $HEAD_POD
```

## Troubleshooting

### Port-forward Issues

```bash
# Kill existing port-forwards
pkill -f "port-forward.*8000"

# Get correct service name
SVC_NAME=$(kubectl get svc -n default | grep gpu-tpu-mix | grep head-svc | awk '{print $1}')
echo "Service name: $SVC_NAME"

# Start new port-forward
kubectl port-forward -n default svc/$SVC_NAME 8000:8000 &

# Verify port is listening
netstat -tln | grep 8000
```

### Connection Refused

```bash
# Check if Serve is running
kubectl exec -n default $HEAD_POD -- serve status

# Check if port 8000 is exposed
kubectl get svc -n default | grep gpu-tpu-mix

# Verify pods are ready
kubectl get pods -n default | grep gpu-tpu-mix
```

### Timeout Errors

```bash
# Check if workers are healthy
kubectl exec -n default $HEAD_POD -- ray status

# Check deployment status
kubectl exec -n default $HEAD_POD -- serve status

# Increase timeout in test scripts
# Python: timeout=60
# curl: --max-time 60
```

### Wrong Results / Model Issues

```bash
# Verify model loaded correctly
kubectl logs -n default <gpu-worker-pod> | grep -i "repvgg\|initialized"
kubectl logs -n default <tpu-worker-pod> | grep -i "repvgg\|initialized"

# Check if ConfigMap is mounted
kubectl exec -n default <worker-pod> -- ls -la /rayserve/apps/

# Test import manually
kubectl exec -n default <worker-pod> -- \
  python3 -c "import sys; sys.path.append('/rayserve/apps'); import repvgg_gpu_app; print('OK')"
```

## Test Coverage Checklist

- [ ] GPU health endpoint responds
- [ ] TPU health endpoint responds
- [ ] GPU inference with valid image
- [ ] TPU inference with valid image
- [ ] Error handling for missing image
- [ ] Error handling for invalid base64
- [ ] Different image formats (JPEG, PNG)
- [ ] Different image sizes
- [ ] top_k parameter variations
- [ ] Concurrent requests
- [ ] Latency measurements
- [ ] Resource utilization check

## Expected Performance

### GPU App (NVIDIA L4, FP16)
- **Cold start**: ~10-15s (first request, pip install)
- **Warm latency**: ~5-20ms per image
- **Throughput**: ~50-200 images/sec (batch size 1)
- **Memory**: ~2-4 GB GPU memory

### TPU App (TPU v6e node, CPU mode)
- **Cold start**: ~10-15s (first request, pip install)
- **Warm latency**: ~50-200ms per image (CPU)
- **Throughput**: ~5-20 images/sec (CPU mode)
- **Note**: With torch-xla2, expect 10-50x faster

**Note**: Current predictions show uniform probabilities because pretrained weights are not loaded. This is expected - the infrastructure is working correctly.

## Cleanup

```bash
# Kill port-forward
pkill -f "port-forward.*8000"

# Clean up test files
rm -f /tmp/image.b64
```

## Next Steps

1. **Add Pretrained Weights**: Mount model weights to improve prediction accuracy
2. **Enable TPU Acceleration**: Build custom image with torch-xla2
3. **Batch Inference**: Test with batch requests
4. **Load Testing**: Use tools like `locust` or `ab` for stress testing
5. **Metrics**: Add Prometheus metrics for monitoring

## Additional Resources

- Main documentation: `README.md`
- Test scripts: `test_repvgg_apps.py`, `test_curl.sh`
- RayService config: `gpu-tpu-app.rayservice.yaml`
- Application code: `repvgg_gpu_app.py`, `repvgg_tpu_app.py`
