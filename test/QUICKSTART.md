# Quick Start Guide - RepVGG GPU/TPU Inference

Get up and running in 5 minutes.

## 1. Deploy (2 minutes)

```bash
cd /home/gech/workspace/oppo/gke-ray-solution/test

# Create ConfigMap
kubectl create configmap repvgg-apps \
  --from-file=repvgg_gpu_app.py=repvgg_gpu_app.py \
  --from-file=repvgg_tpu_app.py=repvgg_tpu_app.py \
  -n default

# Deploy RayService
kubectl apply -f gpu-tpu-app.rayservice.yaml

# Wait for pods (should take ~2 minutes)
kubectl get pods -n default -w | grep gpu-tpu-mix
```

**Wait for all pods to show `1/1 Running`**

## 2. Verify (30 seconds)

```bash
# Get head pod
HEAD_POD=$(kubectl get pod -n default -l ray.io/node-type=head,ray.io/cluster | grep gpu-tpu-mix | awk '{print $1}')

# Check status (should show RUNNING/HEALTHY)
kubectl exec -n default $HEAD_POD -- serve status
```

Expected:
```
applications:
  gpu-app: RUNNING (HEALTHY)
  tpu-app: RUNNING (HEALTHY)
```

## 3. Test (1 minute)

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
TPU: {'status': 'healthy', 'model': 'RepVGG-TPU', 'device': 'TPU-node (CPU mode)', ...}
```

## 4. Full Test (Optional)

```bash
# Setup port-forward
kubectl port-forward -n default svc/$(kubectl get svc -n default | grep gpu-tpu-mix | grep head-svc | awk '{print $1}') 8000:8000 &

# Run automated tests
python3 test_repvgg_apps.py
```

## That's it! 🎉

Your GPU and TPU inference apps are running!

- **GPU endpoint**: `http://localhost:8000/gpu/classify`
- **TPU endpoint**: `http://localhost:8000/tpu/classify`

## Quick Commands

```bash
# Check status
kubectl get rayservice gpu-tpu-mix -n default

# View logs
kubectl logs -n default $HEAD_POD --tail=50

# Access dashboard
kubectl port-forward -n default svc/<service-name> 8265:8265
# Open: http://localhost:8265

# Delete everything
kubectl delete rayservice gpu-tpu-mix -n default
kubectl delete configmap repvgg-apps -n default
```

## Troubleshooting

**Pods stuck in Pending?**
- Check if GPU/TPU nodes are available: `kubectl get nodes`
- Check events: `kubectl describe rayservice gpu-tpu-mix -n default`

**Apps stuck in DEPLOYING?**
- Wait 2-3 minutes for pip install
- Check logs: `kubectl logs -n default <worker-pod> --tail=100`

**Port-forward not working?**
- Kill existing: `pkill -f "port-forward.*8000"`
- Get service name: `kubectl get svc -n default | grep gpu-tpu-mix`
- Try again with correct service name

## Next Steps

- 📖 Read full guide: `README.md`
- 🧪 Advanced testing: `TESTING.md`
- 🔧 Customize apps: Edit `repvgg_gpu_app.py` and `repvgg_tpu_app.py`

## Support

- Issues: Check troubleshooting sections in `README.md` and `TESTING.md`
- Logs: `kubectl logs -n default <pod-name>`
- Status: `kubectl exec -n default $HEAD_POD -- serve status`
