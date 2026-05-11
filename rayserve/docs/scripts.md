# Deployment Scripts

This directory contains automation scripts for building and deploying the GPU+TPU RayService.

## Scripts Overview

### 1. `build-tpu-image.sh`

Builds the custom Ray Docker image with JAX and torch-xla2 for TPU support.

**Usage**:
```bash
./build-tpu-image.sh [OPTIONS]

OPTIONS:
  --clean      Clean Docker cache before building (frees ~30GB)
  --no-cache   Build without using Docker cache
```

**Examples**:
```bash
# Standard build
./build-tpu-image.sh

# Clean cache and rebuild
./build-tpu-image.sh --clean

# Force complete rebuild without cache
./build-tpu-image.sh --clean --no-cache
```

**What it does**:
- Checks for Dockerfile.ray-tpu
- Optionally cleans Docker cache
- Builds image: `us-central1-docker.pkg.dev/gpu-launchpad-playground/gech-ray-gke/ray-tpu:2.55.1`
- Verifies JAX version in built image
- Shows next steps

**Output**:
- Docker image with JAX 0.4.35, torch-xla2, and dependencies
- Build verification (JAX version check)

---

### 2. `push-tpu-image.sh`

Pushes the built Docker image to Google Artifact Registry.

**Usage**:
```bash
./push-tpu-image.sh
```

**What it does**:
- Configures Docker authentication for Artifact Registry
- Checks if image exists locally
- Pushes image with tag `2.55.1`
- Tags and pushes as `latest`

**Prerequisites**:
- Image must be built first (`./build-tpu-image.sh`)
- Authenticated to GCP (`gcloud auth login`)
- Permissions to push to Artifact Registry

**Output**:
```
Image URIs:
  - us-central1-docker.pkg.dev/gpu-launchpad-playground/gech-ray-gke/ray-tpu:2.55.1
  - us-central1-docker.pkg.dev/gpu-launchpad-playground/gech-ray-gke/ray-tpu:latest
```

---

### 3. `deploy-tpu-rayservice.sh`

Complete end-to-end deployment automation.

**Usage**:
```bash
./deploy-tpu-rayservice.sh [OPTIONS]

OPTIONS:
  --rebuild     Build Docker image before deploying
  --skip-push   Skip pushing image to registry (local testing)
  --clean       Clean Docker cache before building
```

**Examples**:
```bash
# Deploy with existing image (ConfigMap update only)
./deploy-tpu-rayservice.sh

# Full rebuild and deploy
./deploy-tpu-rayservice.sh --rebuild

# Rebuild with clean cache and deploy
./deploy-tpu-rayservice.sh --rebuild --clean

# Build but don't push (local testing)
./deploy-tpu-rayservice.sh --rebuild --skip-push
```

**What it does**:
1. **Build Docker image** (if `--rebuild` specified)
2. **Push to Artifact Registry** (if rebuilt and not `--skip-push`)
3. **Update ConfigMap** with application code
4. **Check for existing RayService** (prompts to delete if exists)
5. **Deploy RayService** from YAML
6. **Wait for pods** to be ready
7. **Verify deployment** (health checks, Ray Serve status)

**Interactive prompts**:
- Asks before deleting existing RayService
- Shows verification commands at the end

**Output**:
```
==========================================================="
✅ Deployment complete!
===========================================================

Verification commands:
  kubectl get pods -n default -l ray.io/cluster
  kubectl exec -n default <head-pod> -- ray serve status
  ...
```

---

## Quick Start Workflows

### First-time deployment

```bash
# 1. Build image
./build-tpu-image.sh --clean

# 2. Push to registry
./push-tpu-image.sh

# 3. Deploy
kubectl create configmap repvgg-apps \
  --from-file=repvgg_gpu_app.py \
  --from-file=repvgg_tpu_app.py \
  -n default --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f gpu-tpu-app.rayservice.yaml
```

**OR** use the all-in-one script:
```bash
./deploy-tpu-rayservice.sh --rebuild --clean
```

### Update application code only

```bash
# Just update ConfigMap and restart pods
kubectl create configmap repvgg-apps \
  --from-file=repvgg_gpu_app.py \
  --from-file=repvgg_tpu_app.py \
  -n default --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up changes
kubectl delete pod -l ray.io/cluster -n default
```

**OR**:
```bash
./deploy-tpu-rayservice.sh
# (Answer 'y' to delete existing RayService)
```

### Update Docker image

```bash
# 1. Build new image
./build-tpu-image.sh

# 2. Push to registry
./push-tpu-image.sh

# 3. Force pod restart
kubectl delete rayservice gpu-tpu-mix -n default
kubectl apply -f gpu-tpu-app.rayservice.yaml
```

**OR**:
```bash
./deploy-tpu-rayservice.sh --rebuild
```

### Debug build issues

```bash
# Clean everything and rebuild from scratch
docker system prune -a -f --volumes
./build-tpu-image.sh --no-cache

# Check image
docker images | grep ray-tpu
docker run --rm us-central1-docker.pkg.dev/gpu-launchpad-playground/gech-ray-gke/ray-tpu:2.55.1 \
  python3 -c "import jax; print(jax.__version__)"
```

---

## Verification Commands

After deployment, verify everything is working:

```bash
# Get head pod name
HEAD_POD=$(kubectl get pods -n default -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

# Check all pods
kubectl get pods -n default -l ray.io/cluster

# Check Ray Serve status
kubectl exec -n default $HEAD_POD -- ray serve status

# Test GPU health
kubectl exec -n default $HEAD_POD -- curl -s http://localhost:8000/gpu/health | python3 -m json.tool

# Test TPU health (should show tpu_enabled: true)
kubectl exec -n default $HEAD_POD -- curl -s http://localhost:8000/tpu/health | python3 -m json.tool

# Check TPU detection on TPU worker
TPU_POD=$(kubectl get pods -n default -l ray.io/group-name=tpu-group -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n default $TPU_POD -- python3 -c "import jax; print(jax.devices())"
# Expected: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
```

---

## Configuration

All scripts use these default values (edit scripts to change):

```bash
PROJECT_ID="gpu-launchpad-playground"
REGION="us-central1"
REPO="gech-ray-gke"
IMAGE_NAME="ray-tpu"
TAG="2.55.1"
NAMESPACE="default"
```

---

## Troubleshooting

### Build fails with "No space left on device"

```bash
# Clean Docker cache
docker system prune -a -f --volumes

# Then rebuild
./build-tpu-image.sh --no-cache
```

### Push fails with authentication error

```bash
# Re-authenticate
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

# Or login to GCP
gcloud auth login
```

### Deployment stuck with pods not ready

```bash
# Check pod status
kubectl get pods -n default -l ray.io/cluster

# Check pod logs
kubectl logs <pod-name> --tail=100

# Check events
kubectl describe pod <pod-name>

# For TPU worker specifically
kubectl logs -n default -l ray.io/group-name=tpu-group --tail=200
```

### ConfigMap not updating

ConfigMap changes require pod restart:

```bash
# Update ConfigMap
kubectl create configmap repvgg-apps \
  --from-file=repvgg_gpu_app.py \
  --from-file=repvgg_tpu_app.py \
  -n default --dry-run=client -o yaml | kubectl apply -f -

# Restart all pods
kubectl delete rayservice gpu-tpu-mix -n default
kubectl apply -f gpu-tpu-app.rayservice.yaml
```

### Image not pulling latest version

Force image pull:

```bash
# In gpu-tpu-app.rayservice.yaml, ensure:
imagePullPolicy: Always

# Or use unique tags for each build
TAG="2.55.1-$(date +%Y%m%d-%H%M%S)"
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `build-tpu-image.sh` | Build Docker image |
| `push-tpu-image.sh` | Push to Artifact Registry |
| `deploy-tpu-rayservice.sh` | Complete deployment automation |
| `Dockerfile.ray-tpu` | Docker image definition |
| `gpu-tpu-app.rayservice.yaml` | Kubernetes RayService manifest |
| `repvgg_gpu_app.py` | GPU inference application |
| `repvgg_tpu_app.py` | TPU inference application |

---

## Related Documentation

- [TPU_SETUP_GUIDE.md](./TPU_SETUP_GUIDE.md) - Detailed setup guide with troubleshooting
- [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) - Quick reference for common tasks
- [ENABLE_TPU.md](./ENABLE_TPU.md) - Original TPU enablement notes

---

**Last Updated**: April 21, 2026
