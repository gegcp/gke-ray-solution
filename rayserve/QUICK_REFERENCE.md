# Quick Reference: TPU Setup on GKE

## TL;DR - Working Configuration

```bash
# What works:
✅ Ray 2.55.1 + JAX 0.4.35 + torch-xla2
✅ TPU v6e (Trillium) on GKE 1.35.2
✅ Both GPU andU in single RayService
✅ Actual TPU hardware acceleration enabled
```

## Critical Success Factors

1. **JAX 0.4.35+** (not 0.4.30) for TPU v6e
2. **No JAX_PLATFORMS in Dockerfile** (let GKE runtime set it)
3. **Pre-install deps in Docker image** (not runtime_env.pip)
4. **Ray actor resources: `{"TPU": 1}`** (mandatory)
5. **imagePullPolicy: Always** (during development)

## Quick Commands

### Build & Deploy

```bash
# Clean, build, push
docker system prune -a -f --volumes
docker build -t us-central1-docker.pkg.dev/gpu-launchpad-playground/gech-ray-gke/ray-tpu:2.55.1 -f Dockerfile.ray-tpu .
./push-tpu-image.sh

# Update ConfigMap
kubectl create configmap repvgg-apps \
  --from-file=repvgg_gpu_app.py \
  --from-file=repvgg_tpu_app.py \
  -n default --dry-run=client -o yaml | kubectl apply -f -

# Deploy
kubectl apply -f gpu-tpu-app.rayservice.yaml
```

### Verification

```bash
# Check pods
kubectl get pods -n default -l ray.io/cluster

# Test TPU detection
POD=$(kubectl get pods -n default -l ray.io/group-name=tpu-group -o name | head -1)
kubectl exec $POD -- python3 -c "import jax; print(jax.devices())"
# Expected: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]

# Check health
HEAD=$(kubectl get pods -n default -l ray.io/node-type=head -o name | head -1)
kubectl exec $HEAD -- curl -s http://localhost:8000/tpu/health | python3 -m json.tool
# Expected: "tpu_enabled": true, "jax_backend": "tpu"

# Ray Serve status
kubectl exec $HEAD -- ray serve status
# Expected: Both apps RUNNING & HEALTHY
```

### Debugging

```bash
# Check initialization logs
kubectl exec <tpu-worker-pod> -- bash -c "
  find /tmp/ray -name 'worker*.out' | xargs grep 'Initializing RepVGG' -A 10
"

# Check Ray resources
kubectl exec <head-pod> -- ray status
# Look for: "0.0/1.0 TPU" (1.0 when app running)

# Controller logs
kubectl exec <head-pod> -- bash -c "
  cat /tmp/ray/session_latest/logs/serve/controller*.log | tail -100
"

# Replica logs
kubectl exec <tpu-worker-pod> -- bash -c "
  find /tmp/ray -name 'replica_tpu-app*.log' | xargs cat
"
```

## Common Errors → Solutions

| Error | Cause | Fix |
|-------|-------|-----|
| `No ba16c7433 device found` | Wrong JAX version | Use JAX 0.4.35+ |
| `Device or resource busy` | Multiple JAX inits | Restart pod, check code |
| `ModuleNotFoundError: torch` | Wrong image or cache | `imagePullPolicy: Always` |
| `ActorUnavailableError` | Missing TPU resources | Add `resources: {"TPU": 1}` |
| `conv2d not found` | API changed | Make patch optional |
| `DEPLOY_FAILED` | Check controller logs | See debugging commands |

## Minimal Working Dockerfile

```dockerfile
FROM rayproject/ray:2.55.1.68d0e4-extra-py312-cu128
USER ray

# Critical: JAX 0.4.35+ for TPU v6e
RUN pip install --no-cache-dir \
    "jax[tpu]==0.4.35" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

RUN pip install --no-cache-dir torch-xla2 torchvision pillow fastapi

# Don't set JAX_PLATFORMS here!
ENV TPU_LIBRARY_PATH=/lib/libtpu.so
```

## Minimal RayService Config

```yaml
spec:
  serveConfigV2: |
    applications:
    - name: tpu-app
      route_prefix: /tpu
      import_path: repvgg_tpu_app:model
      deployments:
      - name: RepVGGDeployment
        num_replicas: 1
        ray_actor_options:
          resources: {"TPU": 1}  # ← CRITICAL
  
  rayClusterConfig:
    workerGroupSpecs:
    - groupName: tpu-group
      template:
        spec:
          containers:
          - name: ray-worker
            image: <your-custom-image>
            imagePullPolicy: Always  # ← IMPORTANT for dev
            resources:
              limits:
                google.com/tpu: "1"
            env:
            - name: ENABLE_TPU
              value: "true"
            - name: TPU_LIBRARY_PATH
              value: "/home/ray/anaconda3/lib/python3.12/site-packages/libtpu/libtpu.so"
          nodeSelector:
            cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice
            cloud.google.com/gke-tpu-topology: 1x1
```

## Health Check Template

Expected healthy TPU response:
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

## Version Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| GKE | 1.35.2 | No privileged mode needed (1.28+) |
| Ray | 2.55.1 | Latest stable |
| JAX | 0.4.35 | **Required** for TPU v6e |
| Python | 3.12 | From base image |
| torch-xla2 | Latest | Experimental, API evolving |
| TPU | v6e-1 | Trillium, single chip |

## Links

- [Full Guide](./TPU_SETUP_GUIDE.md)
- [GKE TPU Docs](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus)
- [JAX Releases](https://storage.googleapis.com/jax-releases/libtpu_releases.html)
