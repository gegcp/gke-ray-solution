# Quick Start Guide - Gemma Inference Gateway

This is a condensed guide for quickly deploying the Gemma 3 1B inference gateway.

## Prerequisites Checklist

- [ ] GKE Standard cluster (not Autopilot)
- [ ] Gateway API enabled on cluster
- [ ] Helm 3 installed
- [ ] kubectl configured
- [ ] HuggingFace token

## 5-Minute Setup

### 1. Enable Gateway API (Manual - via Console or gcloud)

```bash
gcloud container clusters update CLUSTER_NAME --zone=ZONE --gateway-api=standard
```

Wait ~5-10 minutes for completion.

### 2. Install CRDs

```bash
kubectl kustomize "github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd?ref=v1.0.1" | kubectl apply -f -
```

### 3. Create Secret

```bash
kubectl create secret generic hf-secret --from-literal=hf_api_token=YOUR_TOKEN
```

### 4. Deploy Everything

```bash
cd llm

# Create ComputeClass
kubectl apply -f gpu-2-compute-class.yaml

# Deploy vLLM
kubectl apply -f vllm-gemma-deployment.yaml

# Wait for node and pod (3-5 min)
kubectl get pods -l app=vllm-gemma-3-1b -w

# Install BBR
helm install bbr \
  --version v1.0.1 \
  --set provider.name=gke \
  --set inferenceGateway.name=vllm-xlb \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/body-based-routing

# Install Inference Pool
INFERENCE_POOL=vllm-gemma-3-1b
helm install ${INFERENCE_POOL} \
  --set inferencePool.modelServers.matchLabels.app=vllm-gemma-3-1b \
  --set provider.name=gke \
  --version v1.0.1 \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool \
  -f epp-values.yaml

# Scale EPP to 1 replica
kubectl scale deployment vllm-gemma-3-1b-epp --replicas=1

# Deploy remaining components
kubectl apply -f inference-objective.yaml
kubectl apply -f gateway.yaml
kubectl apply -f httproute.yaml

# Wait for Gateway (2-3 min)
kubectl wait --for=condition=Programmed --timeout=300s gateway/vllm-xlb
```

## Test

```bash
# Get Gateway IP
GATEWAY_IP=$(kubectl get gateway vllm-xlb -o jsonpath='{.status.addresses[0].value}')

# Test
kubectl run -it --rm test-client --image=curlimages/curl --restart=Never -- \
  curl -X POST http://$GATEWAY_IP/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "google/gemma-3-1b-it", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

## Verify Deployment

```bash
kubectl get computeclass gpu-2-spot
kubectl get nodes -l cloud.google.com/compute-class=gpu-2-spot
kubectl get pods -l app=vllm-gemma-3-1b
kubectl get gateway vllm-xlb
```

## Troubleshooting

**EPP pods crashing?**
- Delete and let them recreate: `kubectl delete pods -l inferencepool=vllm-gemma-3-1b-epp`

**Gateway stuck?**
- Check GatewayClasses exist: `kubectl get gatewayclass`
- Verify using internal LB: Gateway should use `gke-l7-rilb`

**Pod not scheduling?**
- Wait 2-3 minutes for node auto-provisioning
- Check events: `kubectl describe pod -l app=vllm-gemma-3-1b`

## Optional: Deploy Gradio UI

```bash
kubectl apply -f gradio.yaml

# Get LoadBalancer IP (wait for EXTERNAL-IP)
kubectl get svc gradio-chatbot

# Or use port-forward for immediate access
kubectl port-forward svc/gradio-chatbot 8080:8080
# Then open http://localhost:8080
```

## Optional: Deploy Hermes Agents (Demo)

```bash
kubectl create secret generic hermes-api-key \
  --from-literal=api-key="$(openssl rand -hex 32)"

kubectl apply -f agent.yaml

# Check distribution across nodes
kubectl get pods -l app=hermes-agent -o wide
```

## Clean Up

```bash
# Optional components
kubectl delete -f gradio.yaml 2>/dev/null
kubectl delete -f agent.yaml 2>/dev/null
kubectl delete secret hermes-api-key 2>/dev/null

# Core components
kubectl delete -f httproute.yaml
kubectl delete -f gateway.yaml
kubectl delete -f inference-objective.yaml
helm uninstall vllm-gemma-3-1b
helm uninstall bbr
kubectl delete -f vllm-gemma-deployment.yaml
kubectl delete -f gpu-2-compute-class.yaml
```

For detailed documentation, see [README.md](README.md)
