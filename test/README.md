# RepVGG GPU and TPU Inference on Ray Serve

This demo shows how to deploy RepVGG image classification models on both GPU and TPU nodes in a single RayService/RayCluster.

## Architecture

- **RayService**: `gpu-tpu-mix` - Single RayService with dual-accelerator support
- **GPU App**: RepVGG inference on NVIDIA GPU (L4) with FP16 precision
- **TPU App**: RepVGG inference on TPU node (TPU v6e)
- **Deployment**: ConfigMap-based code deployment (no GitHub dependency)

## Prerequisites

### Required
- GKE cluster with:
  - GPU node pool (NVIDIA L4)
  - TPU node pool (TPU v6e)
- `kubectl` configured to access your cluster
- Ray Operator installed in the cluster
- (Optional) HuggingFace secret for model weights: `hf-secret`

### Verify Prerequisites

```bash
# Check node pools
kubectl get nodes --show-labels | grep -E "nvidia-l4|tpu-v6e"

# Check Ray Operator
kubectl get deployment -n ray-system kuberay-operator

# Check HF secret (optional)
kubectl get secret hf-secret -n default
```

## Deployment Steps

### Step 1: Create ConfigMap with Application Code

The GPU and TPU inference apps are stored in a ConfigMap and mounted into the Ray pods.

```bash
cd /home/gech/workspace/oppo/gke-ray-solution/test

# Create ConfigMap from Python files
kubectl create configmap repvgg-apps \
  --from-file=repvgg_gpu_app.py=repvgg_gpu_app.py \
  --from-file=repvgg_tpu_app.py=repvgg_tpu_app.py \
  -n default

# Verify ConfigMap
kubectl get configmap repvgg-apps -n default
kubectl describe configmap repvgg-apps -n default
```

### Step 2: Deploy RayService

```bash
# Apply RayService configuration
kubectl apply -f gpu-tpu-app.rayservice.yaml

# Verify RayService creation
kubectl get rayservice gpu-tpu-mix -n default
```

### Step 3: Monitor Deployment

```bash
# Watch RayService status
kubectl get rayservice gpu-tpu-mix -n default -w

# Check pods
kubectl get pods -n default | grep gpu-tpu-mix

# Wait for all pods to be Running (1/1)
# - Head node: 1 pod
# - GPU worker: 1 pod  
# - TPU worker: 1 pod
```

Expected output:
```
NAME                                       READY   STATUS    RESTARTS   AGE
gpu-tpu-mix-xxxxx-head-xxxxx               1/1     Running   0          2m
gpu-tpu-mix-xxxxx-gpu-group-worker-xxxxx   1/1     Running   0          2m
gpu-tpu-mix-xxxxx-tpu-group-worker-xxxxx   1/1     Running   0          2m
```

### Step 4: Verify Serve Applications

```bash
# Get head pod name
HEAD_POD=$(kubectl get pod -n default -l ray.io/node-type=head,ray.io/cluster | grep gpu-tpu-mix | awk '{print $1}')

# Check Ray Serve status
kubectl exec -n default $HEAD_POD -- serve status

# Should show both apps as RUNNING/HEALTHY
```

Expected output:
```
applications:
  gpu-app:
    status: RUNNING
    deployments:
      RepVGGDeployment:
        status: HEALTHY
        replica_states:
          RUNNING: 1
  tpu-app:
    status: RUNNING
    deployments:
      RepVGGDeployment:
        status: HEALTHY
        replica_states:
          RUNNING: 1
```

## Testing

### Option 1: Quick Health Check

```bash
# Get head pod name
HEAD_POD=$(kubectl get pod -n default -l ray.io/node-type=head,ray.io/cluster | grep gpu-tpu-mix | awk '{print $1}')

# Test GPU app
kubectl exec -n default $HEAD_POD -- \
  python3 -c "import requests; print(requests.get('http://localhost:8000/gpu/health').json())"

# Test TPU app
kubectl exec -n default $HEAD_POD -- \
  python3 -c "import requests; print(requests.get('http://localhost:8000/tpu/health').json())"
```

Expected output:
```
GPU: {'status': 'healthy', 'model': 'RepVGG', 'device': 'cuda', 'precision': 'FP16'}
TPU: {'status': 'healthy', 'model': 'RepVGG-TPU', 'device': 'TPU-node (CPU mode)', 'note': '...'}
```

### Option 2: Port Forward and Test Locally

```bash
# Get service name
kubectl get svc -n default | grep gpu-tpu-mix

# Port forward (use the actual service name)
kubectl port-forward -n default svc/gpu-tpu-mix-xxxxx-head-svc 8000:8000 &

# Run automated test script
cd /home/gech/workspace/oppo/gke-ray-solution/test
python3 test_repvgg_apps.py

# Or use curl script
./test_curl.sh
```

### Option 3: Manual Inference Test

```bash
# Encode test image
IMAGE_BASE64=$(base64 -w 0 /home/gech/workspace/oppo/gke-ray-solution/repvgg/gpu/burger.jpeg)

# Test GPU inference
curl -X POST http://localhost:8000/gpu/classify \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"${IMAGE_BASE64}\", \"top_k\": 5}"

# Test TPU inference
curl -X POST http://localhost:8000/tpu/classify \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"${IMAGE_BASE64}\", \"top_k\": 5}"
```

## Accessing the Applications

### Endpoints

Once deployed, the applications are available at:

- **GPU App**: 
  - Health: `http://<service>:8000/gpu/health`
  - Inference: `http://<service>:8000/gpu/classify`
  
- **TPU App**:
  - Health: `http://<service>:8000/tpu/health`
  - Inference: `http://<service>:8000/tpu/classify`

### Request Format

```json
POST /gpu/classify
Content-Type: application/json

{
  "image": "<base64-encoded-image>",
  "top_k": 5,
  "img_size": 224
}
```

### Response Format

```json
{
  "predictions": [
    {
      "class_id": 281,
      "probability": 0.15
    },
    {
      "class_id": 282,
      "probability": 0.12
    }
  ],
  "model": "RepVGG",
  "device": "cuda"
}
```

## Monitoring

### Ray Dashboard

```bash
# Port forward Ray dashboard
kubectl port-forward -n default svc/gpu-tpu-mix-xxxxx-head-svc 8265:8265

# Open in browser
# http://localhost:8265
```

### Check Logs

```bash
# Get pod names
kubectl get pods -n default | grep gpu-tpu-mix

# Head pod logs
kubectl logs -n default <head-pod-name> --tail=100

# GPU worker logs
kubectl logs -n default <gpu-worker-pod-name> --tail=100

# TPU worker logs
kubectl logs -n default <tpu-worker-pod-name> --tail=100

# Serve controller logs
kubectl exec -n default <head-pod> -- \
  tail -100 /tmp/ray/session_latest/logs/serve/controller.log
```

### Ray Cluster Status

```bash
# Check Ray cluster resources
kubectl exec -n default <head-pod> -- ray status

# Check serve configuration
kubectl exec -n default <head-pod> -- serve config
```

## Configuration

### Key Files

- **`gpu-tpu-app.rayservice.yaml`**: RayService definition with GPU and TPU worker groups
- **`repvgg_gpu_app.py`**: GPU inference application code
- **`repvgg_tpu_app.py`**: TPU inference application code
- **`test_repvgg_apps.py`**: Automated test script
- **`test_curl.sh`**: Simple curl-based test script

### Environment Variables

Applications can be configured via environment variables in the RayService:

**GPU App**:
- `MODEL_VARIANT`: RepVGG model variant (A0, B0) - default: B0
- `NUM_CLASSES`: Number of output classes - default: 1000
- `USE_FP16`: Enable FP16 precision - default: true

**TPU App**:
- `MODEL_VARIANT`: RepVGG model variant (A0, B0) - default: B0
- `NUM_CLASSES`: Number of output classes - default: 1000
- `TPU_NODE`: Indicates running on TPU node - default: true

### Resource Allocation

**Head Node**:
- CPU: 2 cores
- Memory: 8Gi

**GPU Worker**:
- CPU: 4 cores
- Memory: 20Gi
- GPU: 1x NVIDIA L4

**TPU Worker**:
- CPU: 1 core
- Memory: 20Gi
- TPU: 1x TPU v6e chip

## Updating the Deployment

### Update Application Code

```bash
# Edit the Python files
vim repvgg_gpu_app.py
vim repvgg_tpu_app.py

# Delete old ConfigMap
kubectl delete configmap repvgg-apps -n default

# Create new ConfigMap
kubectl create configmap repvgg-apps \
  --from-file=repvgg_gpu_app.py=repvgg_gpu_app.py \
  --from-file=repvgg_tpu_app.py=repvgg_tpu_app.py \
  -n default

# Apply updated RayService (will trigger redeployment)
kubectl apply -f gpu-tpu-app.rayservice.yaml
```

### Update RayService Configuration

```bash
# Edit YAML file
vim gpu-tpu-app.rayservice.yaml

# Apply changes
kubectl apply -f gpu-tpu-app.rayservice.yaml

# Monitor rollout
kubectl get rayservice gpu-tpu-mix -n default -w
```

### Force Redeploy

If the deployment gets stuck, delete and recreate:

```bash
# Delete RayService
kubectl delete rayservice gpu-tpu-mix -n default

# Wait for pods to terminate
kubectl get pods -n default | grep gpu-tpu-mix

# Reapply
kubectl apply -f gpu-tpu-app.rayservice.yaml
```

## Troubleshooting

### Issue: Pods not starting

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

### Issue: Serve applications stuck in DEPLOYING

```bash
# Check serve status
kubectl exec -n default <head-pod> -- serve status

# Check if dependencies are installing
kubectl logs -n default <worker-pod> --tail=200

# Common causes:
# - pip install timeout (large packages)
# - Python import errors
# - Resource constraints
```

### Issue: Health endpoint returns error

```bash
# Check if deployment is healthy
kubectl exec -n default <head-pod> -- serve status

# Check application logs
kubectl logs -n default <worker-pod> | grep -A 10 "RepVGG"

# Test import manually
kubectl exec -n default <worker-pod> -- \
  python3 -c "import sys; sys.path.append('/rayserve/apps'); import repvgg_gpu_app"
```

### Issue: ConfigMap not mounted

```bash
# Check if ConfigMap exists
kubectl get configmap repvgg-apps -n default

# Check mount in pod
kubectl exec -n default <pod-name> -- ls -la /rayserve/apps/

# Should show:
# repvgg_gpu_app.py
# repvgg_tpu_app.py
```

### Issue: Port forward fails

```bash
# Kill existing port-forward
pkill -f "port-forward.*8000"

# Find correct service name
kubectl get svc -n default | grep gpu-tpu-mix

# Restart port-forward with correct name
kubectl port-forward -n default svc/<actual-service-name> 8000:8000
```

## Cleanup

```bash
# Delete RayService (will delete all associated resources)
kubectl delete rayservice gpu-tpu-mix -n default

# Delete ConfigMap
kubectl delete configmap repvgg-apps -n default

# Verify cleanup
kubectl get pods -n default | grep gpu-tpu-mix
kubectl get svc -n default | grep gpu-tpu-mix
```

## Performance Notes

### GPU App
- **Device**: NVIDIA L4 GPU
- **Precision**: FP16 (half precision) for faster inference
- **Performance**: ~2-5ms per image (depending on batch size)

### TPU App
- **Current**: Running on TPU node using CPU
- **Reason**: torch-xla2/JAX installation (~2GB) causes deployment timeouts
- **To enable full TPU**: Pre-build custom Docker image with dependencies

### Adding Full TPU Support

To enable actual TPU acceleration:

1. **Build custom image**:
```dockerfile
FROM rayproject/ray:2.40.0-py312-gpu

# Install TPU dependencies
RUN pip install jax[tpu] torch-xla2 -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
```

2. **Update RayService** to use custom image:
```yaml
spec:
  rayClusterConfig:
    workerGroupSpecs:
    - groupName: tpu-group
      template:
        spec:
          containers:
          - name: ray-worker
            image: <your-custom-image>
```

3. **Update TPU app** to use torch-xla2 (code is ready, just needs dependencies)

## Additional Resources

- **Ray Serve Documentation**: https://docs.ray.io/en/latest/serve/
- **KubeRay Documentation**: https://docs.ray.io/en/latest/cluster/kubernetes/
- **RepVGG Paper**: https://arxiv.org/abs/2101.03697
- **Test Scripts**: `test_repvgg_apps.py`, `test_curl.sh`
- **Detailed Testing Guide**: `TESTING.md`

## Architecture Diagram

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

## Files Overview

```
gke-ray-solution/test/
├── README.md                      # This file
├── TESTING.md                     # Detailed testing guide
├── gpu-tpu-app.rayservice.yaml    # RayService definition
├── repvgg_gpu_app.py              # GPU inference app
├── repvgg_tpu_app.py              # TPU inference app
├── test_repvgg_apps.py            # Python test script
└── test_curl.sh                   # Bash test script
```
