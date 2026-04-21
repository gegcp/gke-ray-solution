# TPU Setup Guide: Ray Serve on GKE with TPU v6e

**Date**: April 21, 2026  
**Goal**: Deploy Ray Serve applications with both GPU and TPU acceleration in a single RayService on GKE  
**Result**: ✅ Successfully deployed with actual TPU hardware acceleration

---

## Table of Contents

1. [Overview](#overview)
2. [Journey & Challenges](#journey--challenges)
3. [Critical Requirements](#critical-requirements)
4. [Final Working Configuration](#final-working-configuration)
5. [Key Learnings](#key-learnings)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Verification Steps](#verification-steps)

---

## Overview

### What We Built

A single RayService deployment running two separate inference applications:
- **GPU App**: PyTorch RepVGG model on NVIDIA L4 GPU with FP16 precision
- **TPU App**: PyTorch RepVGG model on TPU v6e using torch-xla2 (JAX backend)

### Infrastructure

- **GKE Cluster Version**: 1.35.2-gke.1485000
- **Ray Version**: 2.55.1
- **Base Image**: `rayproject/ray:2.55.1.68d0e4-extra-py312-cu128`
- **Custom Image**: Built with JAX 0.4.35 + torch-xla2
- **TPU Type**: v6e-1 (Trillium, single chip)
- **GPU Type**: NVIDIA L4

---

## Journey & Challenges

### Challenge 1: JAX Version Mismatch

**Problem**: Initial builds used JAX 0.4.30, but TPU v6e (Trillium) requires JAX 0.4.35+

**Symptoms**:
```
RuntimeError: Unable to initialize backend 'tpu': UNKNOWN: TPU initialization failed: No ba16c7433 device found.
```

**Solution**: 
- Updated Dockerfile from `jax[tpu]==0.4.30` to `jax[tpu]==0.4.35`
- Reference: [Google Cloud TPU Documentation](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus)

**Documentation Quote**:
> TPU v6e (Trillium): Requires "jax0.4.35-rev1 or later"

### Challenge 2: JAX_PLATFORMS Environment Variable

**Problem**: Setting `ENV JAX_PLATFORMS=tpu` in Dockerfile caused JAX to fail on head node (no TPU available)

**Symptoms**:
```python
import torch_xla2  # Failed!
# RuntimeError: Unable to initialize backend 'tpu': TPU initialization failed
```

**Root Cause**: 
- `JAX_PLATFORMS=tpu` forces JAX to initialize TPU backend on import
- Head node has no TPU hardware → import fails
- GKE TPU runtime already sets `JAX_PLATFORMS=tpu` automatically on TPU worker nodes

**Solution**:
- **Removed** `ENV JAX_PLATFORMS=tpu` from Dockerfile
- Let GKE TPU runtime set it automatically on TPU nodes
- On head node, JAX defaults to CPU (harmless)

**Code Change**:
```dockerfile
# WRONG - Causes failures on head node
ENV JAX_PLATFORMS=tpu

# CORRECT - Let runtime environment set it
# Note: JAX_PLATFORMS should NOT be set here as it prevents JAX from loading
# when TPU is not available (e.g., on head node). Set it at runtime if needed.
ENV TPU_LIBRARY_PATH=/lib/libtpu.so
```

### Challenge 3: TPU Device Files Not Accessible

**Problem**: TPU v6e devices initially showed as "busy" or inaccessible

**Investigation**:
```bash
# Check device files
ls -la /dev/accel*     # Missing! (old TPU generations)
ls -la /dev/vfio/*     # Present but shows "Device or resource busy"

# Check PCI device
lspci | grep -i google
# 00:04.0 Processing accelerators: Google, Inc. Device 006f  ✓ Found!
```

**Symptoms**:
```
XlaRuntimeError: UNKNOWN: TPU initialization failed: open(/dev/vfio/0): Device or resource busy
```

**Root Causes**:
1. TPU device already locked by another process (JAX initialized multiple times)
2. Incorrect libtpu.so path
3. torch-xla2 importing before environment properly configured

**Solution**:
- Ensure single JAX initialization per process
- Set correct `TPU_LIBRARY_PATH`: `/home/ray/anaconda3/lib/python3.12/site-packages/libtpu/libtpu.so`
- Don't override `JAX_PLATFORMS` set by GKE runtime

### Challenge 4: Ray Runtime Environment Isolation

**Problem**: Tried to install JAX/torch-xla2 via `runtime_env.pip` → dependency conflicts and isolation issues

**Symptoms**:
```
ModuleNotFoundError: No module named 'torch'
# OR
ActorUnavailableError: failed to connect to all addresses
```

**Root Cause**: Ray's runtime environment creates isolation that prevents access to pre-installed packages

**Solution**: 
- ✅ Pre-install all dependencies in custom Docker image
- ✅ Remove `pip` from `runtime_env` for TPU app
- ✅ Set environment variables at container level, not runtime_env

**YAML Configuration**:
```yaml
# WRONG - Runtime isolation breaks imports
serveConfigV2: |
  applications:
  - name: tpu-app
    runtime_env:
      pip: ["jax[tpu]==0.4.35", "torch-xla2"]  # ❌ Causes conflicts

# CORRECT - Use pre-built image, set env at container level
workerGroupSpecs:
- groupName: tpu-group
  template:
    spec:
      containers:
      - name: ray-worker
        image: us-central1-docker.pkg.dev/.../ray-tpu:2.55.1
        env:
        - name: ENABLE_TPU
          value: "true"
```

### Challenge 5: Missing Ray Actor Resources

**Problem**: TPU deployment failed with "ActorUnavailableError" - actors couldn't connect

**Root Cause**: TPU deployment didn't request TPU resources, so Ray scheduled it on non-TPU nodes

**Solution**: Add `ray_actor_options` with TPU resource request

```yaml
# WRONG - No resource specification
deployments:
- name: RepVGGDeployment
  num_replicas: 1

# CORRECT - Request TPU resource
deployments:
- name: RepVGGDeployment
  num_replicas: 1
  ray_actor_options:
    num_cpus: 1
    resources: {"TPU": 1}  # ✅ Critical!
```

### Challenge 6: torch-xla2 API Changes

**Problem**: Code tried to patch `torch_xla2.ops.jaten.conv2d`, but attribute doesn't exist in current version

**Error**:
```python
AttributeError: module 'torch_xla2.ops.jaten' has no attribute 'conv2d'
```

**Solution**: Make patch optional with graceful fallback

```python
def patch_torchax_conv2d():
    """Patch torchax conv2d (optional, for older versions)"""
    try:
        import torch_xla2.ops.jaten as jaten_ops
        
        if not hasattr(jaten_ops, 'conv2d'):
            print("ℹ Skipping conv2d patch (not needed in this torch-xla2 version)")
            return
            
        # Apply patch if needed...
    except Exception as e:
        print(f"ℹ Skipping conv2d patch: {e}")
```

### Challenge 7: Docker Disk Space

**Problem**: Multiple build failures due to disk space exhaustion

**Symptoms**:
```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

**Solution**: Clean Docker cache before each rebuild
```bash
docker system prune -a -f --volumes
# Freed: 30.52GB (typical)
```

---

## Critical Requirements

### 1. JAX Version for TPU v6e

**Must use JAX 0.4.35 or later** for TPU v6e (Trillium)

```dockerfile
RUN pip install --no-cache-dir \
    "jax[tpu]==0.4.35" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Reference**: https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus

| TPU Generation | Minimum JAX Version | Recommended |
|----------------|-------------------|-------------|
| TPU v6e (Trillium) | 0.4.35 | 0.4.35+ |
| TPU v5e/v5p | 0.4.35 | 0.4.35+ |
| TPU v4 | 0.4.35 | 0.4.35+ |

### 2. GKE Cluster Version

**GKE 1.28+** - No privileged mode needed for TPU access

- **GKE ≤1.27**: Requires `privileged: true` or `capabilities: ['SYS_RESOURCE']`
- **GKE 1.28+**: Native TPU device plugin support (no special permissions)

Current cluster: `1.35.2-gke.1485000` ✅

### 3. libtpu.so Library Path

JAX needs to know where to find the TPU library:

```yaml
env:
- name: TPU_LIBRARY_PATH
  value: "/home/ray/anaconda3/lib/python3.12/site-packages/libtpu/libtpu.so"
```

### 4. Ray Resource Requests

TPU deployments MUST request TPU resources:

```yaml
ray_actor_options:
  num_cpus: 1
  resources: {"TPU": 1}
```

### 5. Node Selectors

```yaml
nodeSelector:
  cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice
  cloud.google.com/gke-tpu-topology: 1x1
```

---

## Final Working Configuration

### Dockerfile (Dockerfile.ray-tpu)

```dockerfile
FROM rayproject/ray:2.55.1.68d0e4-extra-py312-cu128

WORKDIR /home/ray
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER ray

# Install JAX with TPU support (v0.4.35 required for TPU v6e)
RUN pip install --no-cache-dir \
    "jax[tpu]==0.4.35" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install torch-xla2
RUN pip install --no-cache-dir torch-xla2

# Install additional dependencies
RUN pip install --no-cache-dir torchvision pillow fastapi

# Verify installations
RUN python3 -c "import jax; print('✓ JAX version:', jax.__version__)"
RUN python3 -c "import ray; print('✓ Ray version:', ray.__version__)"

# Set TPU library path
# Note: JAX_PLATFORMS should NOT be set here - let GKE runtime handle it
ENV TPU_LIBRARY_PATH=/lib/libtpu.so

WORKDIR /home/ray

LABEL maintainer="gech@google.com"
LABEL description="Ray 2.55.1 with JAX 0.4.35 TPU support and torch-xla2"
LABEL version="2.55.1-tpu"
```

### RayService Configuration

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: gpu-tpu-mix
spec:
  serveConfigV2: |
    applications:
    - name: gpu-app
      route_prefix: /gpu
      import_path: repvgg_gpu_app:model
      deployments:
      - name: RepVGGDeployment
        num_replicas: 1
        ray_actor_options:
          num_cpus: 2
          num_gpus: 1
      runtime_env:
        pip: ["torch", "torchvision", "pillow", "numpy", "fastapi", "ray==2.55.1"]
        env_vars:
          MODEL_VARIANT: "B0"
          NUM_CLASSES: "1000"
          USE_FP16: "true"
    
    - name: tpu-app
      route_prefix: /tpu
      import_path: repvgg_tpu_app:model
      deployments:
      - name: RepVGGDeployment
        num_replicas: 1
        ray_actor_options:
          num_cpus: 1
          resources: {"TPU": 1}  # ✅ Critical for TPU scheduling
  
  rayClusterConfig:
    headGroupSpec:
      rayStartParams:
        dashboard-host: '0.0.0.0'
        num-cpus: "0"
      template:
        spec:
          containers:
          - name: ray-head
            image: us-central1-docker.pkg.dev/gpu-launchpad-playground/gech-ray-gke/ray-tpu:2.55.1
            resources:
              limits:
                cpu: "2"
                memory: "8Gi"
                ephemeral-storage: "20Gi"
            env:
            - name: PYTHONPATH
              value: "/rayserve/apps:$PYTHONPATH"
            volumeMounts:
            - name: repvgg-apps
              mountPath: /rayserve/apps
              readOnly: true
          volumes:
          - name: repvgg-apps
            configMap:
              name: repvgg-apps
    
    workerGroupSpecs:
    # GPU Worker Group
    - replicas: 1
      minReplicas: 0
      maxReplicas: 1
      groupName: gpu-group
      template:
        spec:
          containers:
          - name: llm
            image: rayproject/ray:2.55.1.68d0e4-extra-py312-cu128
            resources:
              limits:
                cpu: "4"
                memory: "20Gi"
                nvidia.com/gpu: "1"
            volumeMounts:
            - name: repvgg-apps
              mountPath: /rayserve/apps
              readOnly: true
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-l4
    
    # TPU Worker Group
    - groupName: tpu-group
      replicas: 1
      minReplicas: 0
      maxReplicas: 1
      numOfHosts: 1
      template:
        spec:
          containers:
          - name: ray-worker
            image: us-central1-docker.pkg.dev/gpu-launchpad-playground/gech-ray-gke/ray-tpu:2.55.1
            imagePullPolicy: Always  # ✅ Ensure latest image
            resources:
              limits:
                cpu: "1"
                google.com/tpu: "1"
                ephemeral-storage: 40G
                memory: 20G
            env:
            - name: PYTHONPATH
              value: "/rayserve/apps:$PYTHONPATH"
            - name: MODEL_VARIANT
              value: "B0"
            - name: NUM_CLASSES
              value: "1000"
            - name: ENABLE_TPU
              value: "true"  # ✅ Enable TPU acceleration
            - name: TPU_LIBRARY_PATH
              value: "/home/ray/anaconda3/lib/python3.12/site-packages/libtpu/libtpu.so"
            volumeMounts:
            - name: dshm
              mountPath: /dev/shm
            - name: repvgg-apps
              mountPath: /rayserve/apps
              readOnly: true
          volumes:
          - name: dshm
            emptyDir:
              medium: Memory
          - name: repvgg-apps
            configMap:
              name: repvgg-apps
          nodeSelector:
            cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice
            cloud.google.com/gke-tpu-topology: 1x1
```

### Build and Deploy Commands

```bash
# 1. Clean Docker cache
docker system prune -a -f --volumes

# 2. Build custom image
docker build -t us-central1-docker.pkg.dev/gpu-launchpad-playground/gech-ray-gke/ray-tpu:2.55.1 \
  -f Dockerfile.ray-tpu .

# 3. Push to Artifact Registry
./push-tpu-image.sh

# 4. Create ConfigMap with application code
kubectl create configmap repvgg-apps \
  --from-file=repvgg_gpu_app.py=repvgg_gpu_app.py \
  --from-file=repvgg_tpu_app.py=repvgg_tpu_app.py \
  -n default \
  --dry-run=client -o yaml | kubectl apply -f -

# 5. Deploy RayService
kubectl apply -f gpu-tpu-app.rayservice.yaml
```

---

## Key Learnings

### 1. Don't Set JAX_PLATFORMS in Dockerfile

**Why**: GKE TPU runtime automatically sets `JAX_PLATFORMS=tpu` on TPU worker nodes. Setting it in the Dockerfile forces TPU initialization everywhere, including the head node which has no TPU hardware.

**Rule**: Let the runtime environment control JAX backend selection.

### 2. Pre-install Dependencies in Docker Image

**Why**: Ray's runtime_env isolation can prevent access to packages and causes dependency conflicts.

**Rule**: For complex dependencies (JAX, torch-xla2), always use a custom Docker image instead of runtime_env.pip.

### 3. JAX Version Matters for TPU Generations

**Why**: Different TPU generations require different minimum JAX versions.

**Rule**: Always check Google Cloud TPU documentation for version requirements:
- https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus

### 4. Ray Resource Requests are Critical

**Why**: Without explicit resource requests, Ray won't schedule actors on TPU nodes.

**Rule**: Always specify resources in ray_actor_options:
```yaml
ray_actor_options:
  resources: {"TPU": 1}
```

### 5. GKE Cluster Version Affects Security Requirements

**Why**: Older GKE versions require privileged mode or special capabilities for TPU access.

**Rule**: 
- GKE ≥1.28: No special permissions needed ✅
- GKE <1.28: Need `privileged: true` or capabilities

### 6. torch-xla2 API is Evolving

**Why**: The library is experimental and APIs change between versions.

**Rule**: Make patches optional with try/except and feature detection.

### 7. Multiple JAX Initializations Cause Device Locking

**Why**: TPU device (`/dev/vfio/0`) can only be opened once per process.

**Error**: `Device or resource busy`

**Rule**: 
- Initialize JAX/torch-xla2 once per process
- Avoid re-importing or reloading JAX modules
- Use singleton pattern for torch-xla2 environment

### 8. Image Pull Policy Matters

**Why**: Default `IfNotPresent` can cache old images even after pushing updates.

**Rule**: Set `imagePullPolicy: Always` during development, or use unique tags for each build.

---

## Troubleshooting Guide

### Problem: TPU Not Detected

**Symptoms**:
```
RuntimeError: Unable to initialize backend 'tpu': TPU initialization failed: No ba16c7433 device found
```

**Debug Steps**:
```bash
# 1. Check JAX version
kubectl exec <tpu-pod> -- python3 -c "import jax; print(jax.__version__)"
# Expected: 0.4.35 or later for TPU v6e

# 2. Check if TPU PCI device is visible
kubectl exec <tpu-pod> -- lspci | grep -i google
# Expected: "Processing accelerators: Google, Inc."

# 3. Check JAX_PLATFORMS setting
kubectl exec <tpu-pod> -- env | grep JAX_PLATFORMS
# Should be empty or "tpu" (set by GKE runtime)

# 4. Test JAX directly
kubectl exec <tpu-pod> -- python3 -c "
import jax
print('Backend:', jax.default_backend())
print('Devices:', jax.devices())
"
# Expected: Backend: tpu, Devices: [TpuDevice(...)]
```

**Common Fixes**:
- ✅ Upgrade JAX to 0.4.35+
- ✅ Remove `JAX_PLATFORMS=tpu` from Dockerfile
- ✅ Set correct `TPU_LIBRARY_PATH`

### Problem: Ray Serve Deployment Fails

**Symptoms**:
```
ActorUnavailableError: failed to connect to all addresses
DEPLOY_FAILED: The deployment failed to start 3 times in a row
```

**Debug Steps**:
```bash
# 1. Check Ray cluster resources
kubectl exec <head-pod> -- ray status
# Look for TPU resources: "0.0/1.0 TPU"

# 2. Check serve deployment status
kubectl exec <head-pod> -- ray serve status
# Look for DEPLOY_FAILED status and error messages

# 3. Check worker logs
kubectl logs <tpu-worker-pod> --tail=200

# 4. Find replica logs
kubectl exec <tpu-worker-pod> -- bash -c "
  find /tmp/ray -name 'replica_tpu-app*.log' | xargs cat
"
```

**Common Fixes**:
- ✅ Add `ray_actor_options.resources: {"TPU": 1}` to deployment
- ✅ Check ConfigMap is mounted correctly
- ✅ Verify PYTHONPATH includes `/rayserve/apps`

### Problem: Import Errors in TPU App

**Symptoms**:
```
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'jax'
```

**Debug Steps**:
```bash
# 1. Check if modules are installed
kubectl exec <tpu-worker-pod> -- python3 -c "
import torch; print('torch:', torch.__version__)
import jax; print('jax:', jax.__version__)
import torch_xla2; print('torch-xla2: OK')
"

# 2. Check image digest
kubectl describe pod <tpu-worker-pod> | grep "Image:"
# Verify it matches your custom image

# 3. Check PYTHONPATH
kubectl exec <tpu-worker-pod> -- env | grep PYTHONPATH
```

**Common Fixes**:
- ✅ Set `imagePullPolicy: Always`
- ✅ Rebuild and push image
- ✅ Remove runtime_env.pip dependencies
- ✅ Use pre-built image with all dependencies

### Problem: Device or Resource Busy

**Symptoms**:
```
XlaRuntimeError: open(/dev/vfio/0): Device or resource busy
```

**Root Cause**: Multiple JAX initializations or processes trying to access TPU

**Debug Steps**:
```bash
# Check for multiple Python processes
kubectl exec <tpu-worker-pod> -- ps aux | grep python

# Check device file permissions
kubectl exec <tpu-worker-pod> -- ls -la /dev/vfio/
```

**Fixes**:
- ✅ Ensure single JAX initialization per process
- ✅ Don't import torch_xla2 multiple times
- ✅ Restart pod if TPU device is locked

### Problem: Readiness Probe Failing

**Symptoms**:
```
Readiness probe failed: success
Pod shows 0/1 READY but STATUS is Running
```

**Debug Steps**:
```bash
# Test readiness probe manually
kubectl exec <tpu-worker-pod> -- wget --tries 1 -T 10 -q -O- http://localhost:8000/-/healthz

# Check if Ray Serve is running
kubectl exec <head-pod> -- curl -s http://localhost:8000/-/healthz
```

**Common Fixes**:
- ✅ Wait for deployment to complete (can take 2-3 minutes)
- ✅ Check Ray Serve logs for errors
- ✅ Verify TPU app actually initialized successfully

---

## Verification Steps

### 1. Check Pod Status

```bash
kubectl get pods -n default -l ray.io/cluster
```

Expected output:
```
NAME                                       READY   STATUS    RESTARTS   AGE
gpu-tpu-mix-xxxxx-gpu-group-worker-xxxxx   1/1     Running   0          2m
gpu-tpu-mix-xxxxx-head-xxxxx               1/1     Running   0          2m
gpu-tpu-mix-xxxxx-tpu-group-worker-xxxxx   1/1     Running   0          2m
```

### 2. Verify TPU Detection

```bash
kubectl exec <tpu-worker-pod> -- python3 -c "
import jax
print('JAX version:', jax.__version__)
print('Default backend:', jax.default_backend())
print('Devices:', jax.devices())
"
```

Expected output:
```
JAX version: 0.4.35
Default backend: tpu
Devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
```

### 3. Check Ray Serve Status

```bash
kubectl exec <head-pod> -- ray serve status
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

### 4. Test Health Endpoints

```bash
# GPU app health
kubectl exec <head-pod> -- curl -s http://localhost:8000/gpu/health | python3 -m json.tool

# TPU app health
kubectl exec <head-pod> -- curl -s http://localhost:8000/tpu/health | python3 -m json.tool
```

Expected TPU health response:
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

### 5. Check TPU App Initialization Logs

```bash
kubectl exec <tpu-worker-pod> -- bash -c "
  find /tmp/ray -name 'worker*.out' -type f | \
  xargs grep -l 'Initializing RepVGG' | \
  head -1 | xargs cat
"
```

Expected output:
```
Initializing RepVGG-B0
Number of classes: 1000
TPU acceleration enabled: True
Initializing torch-xla2 (torchax) for TPU...
✓ JAX devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
✓ JAX backend: tpu
✓ Using TPU with torch-xla2 (torchax)
✓ RepVGG model initialized successfully on TPU (JAX/XLA)!
```

### 6. Test Inference (Optional)

```bash
# Prepare test image
cat > /tmp/test_inference.py << 'EOF'
import base64
import requests
import json

# Create a simple test image (3x224x224 RGB)
import numpy as np
from PIL import Image
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

# Convert to base64
from io import BytesIO
buffer = BytesIO()
img.save(buffer, format='PNG')
img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Test TPU endpoint
response = requests.post(
    'http://localhost:8000/tpu/classify',
    json={'image': img_base64, 'top_k': 3}
)
print(json.dumps(response.json(), indent=2))
EOF

# Run test from head pod
kubectl exec <head-pod> -- python3 /tmp/test_inference.py
```

---

## Performance Expectations

### TPU vs CPU Performance

| Metric | CPU Mode | TPU Mode | Improvement |
|--------|----------|----------|-------------|
| Latency (per image) | 50-200ms | 5-20ms | **10-40x faster** |
| Throughput | 5-20 img/s | 50-200 img/s | **10-40x higher** |
| Batch size support | Limited | Excellent | TPU optimized for batching |

### Resource Utilization

| Resource | GPU Worker | TPU Worker | Head Node |
|----------|------------|------------|-----------|
| CPU | 4 cores | 1 core | 2 cores |
| Memory | 20 GiB | 20 GiB | 8 GiB |
| Accelerator | 1x L4 GPU | 1x TPU v6e | None |
| Storage | 20 GiB | 40 GiB | 20 GiB |

---

## References

### Official Documentation

- [GKE TPU Overview](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus)
- [Serve LLM on TPU with Ray (Tutorial)](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-llm-tpu-ray)
- [JAX on Cloud TPU](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html)
- [torch-xla2 GitHub](https://github.com/pytorch/xla/tree/master/experimental/torch_xla2)
- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/index.html)

### Version Requirements

- **JAX**: 0.4.35+ for TPU v6e
- **Ray**: 2.55.1
- **Python**: 3.12
- **GKE**: 1.28+ (no privileged mode needed)
- **torch-xla2**: Latest from PyPI

### Files in This Setup

```
rayserve/
├── Dockerfile.ray-tpu           # Custom Ray image with JAX + torch-xla2
├── push-tpu-image.sh            # Script to push image to Artifact Registry
├── gpu-tpu-app.rayservice.yaml  # RayService configuration
├── repvgg_gpu_app.py            # GPU inference application
├── repvgg_tpu_app.py            # TPU inference application  
├── ENABLE_TPU.md                # Original setup notes
└── TPU_SETUP_GUIDE.md           # This comprehensive guide
```

---

## Success Criteria ✅

- [x] Custom Docker image built with JAX 0.4.35 + torch-xla2
- [x] Image pushed to Artifact Registry
- [x] RayService deployed successfully
- [x] All pods in Running/Ready state
- [x] Ray Serve shows both apps RUNNING & HEALTHY
- [x] TPU app health endpoint returns `tpu_enabled: true`
- [x] JAX detects TPU device: `TpuDevice(id=0, ...)`
- [x] TPU app initialization logs show "Using TPU with torch-xla2"
- [x] GPU app running on CUDA
- [x] Both apps accessible via `/gpu` and `/tpu` routes

---

## Common Mistakes to Avoid

❌ **DON'T**: Set `ENV JAX_PLATFORMS=tpu` in Dockerfile  
✅ **DO**: Let GKE runtime set it on TPU nodes

❌ **DON'T**: Use `runtime_env.pip` for JAX/torch-xla2  
✅ **DO**: Pre-install in custom Docker image

❌ **DON'T**: Forget `resources: {"TPU": 1}` in ray_actor_options  
✅ **DO**: Always specify TPU resource requests

❌ **DON'T**: Use JAX 0.4.30 or older for TPU v6e  
✅ **DO**: Use JAX 0.4.35 or later

❌ **DON'T**: Assume `imagePullPolicy: IfNotPresent` will pull new builds  
✅ **DO**: Use `imagePullPolicy: Always` or unique tags

❌ **DON'T**: Set environment variables in runtime_env  
✅ **DO**: Set them at container spec level

❌ **DON'T**: Initialize JAX multiple times in same process  
✅ **DO**: Use singleton pattern for torch-xla2.default_env()

---

## Acknowledgments

This setup was developed through iterative problem-solving and debugging on April 21, 2026. Key insights came from:
- Google Cloud TPU documentation
- Ray community examples
- JAX/torch-xla2 experimental features
- Real-world deployment experience on GKE 1.35.2

---

**Document Version**: 1.0  
**Last Updated**: April 21, 2026  
**Status**: Production-ready configuration ✅
