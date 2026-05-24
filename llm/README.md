# Gemma Inference Pool and Gateway Deployment

This directory contains the configuration files and scripts to deploy a Gemma 3 1B inference pool with GKE Inference Gateway using the `gpu-2-spot` ComputeClass.

## Architecture

The deployment consists of:

1. **ComputeClass (gpu-2-spot)**: Auto-provisions g2-standard-8 instances with L4 GPUs
   - Priority 1: Spot instances (lower cost)
   - Priority 2: On-demand instances (fallback if no spot capacity)
2. **vLLM Deployment**: Runs Gemma 3 1B model with vLLM on auto-provisioned GPU nodes
3. **Inference Pool**: Manages multiple vLLM instances with intelligent endpoint picking
4. **Body-Based Routing (BBR)**: Extracts model name from request body for routing
5. **Gateway**: Regional internal load balancer for inference requests (gke-l7-rilb)
6. **HTTPRoute**: Routes requests to the appropriate InferencePool based on model name
7. **Gradio UI** (Optional): Web-based chat interface for interacting with the model
8. **Hermes Agents** (Optional): Demo containers distributed across GPU/TPU nodes

## Files

- `gpu-2-compute-class.yaml` - ComputeClass definition for g2-standard-8 (spot + on-demand fallback)
- `vllm-gemma-deployment.yaml` - vLLM deployment, service, and HPA using gpu-2-spot
- `epp-values.yaml` - Endpoint Picker Plugin configuration with prefix caching
- `inference-objective.yaml` - InferenceObjective for routing to the pool
- `gateway.yaml` - Gateway resource (regional internal load balancer)
- `httproute.yaml` - HTTPRoute for model-based routing
- `agent.yaml` - Hermes agent deployment (demo containers on GPU/TPU nodes)
- `gradio.yaml` - Gradio chat UI to interact with the inference pool
- `llm-inference.html` - Reference documentation from GKE lab

## Prerequisites

1. **GKE cluster** (Standard mode, not Autopilot)
2. **Gateway API enabled** on the cluster
3. **Helm 3** installed
4. **kubectl** configured to access your cluster
5. **HuggingFace token** for model access

## Setup Steps

### Step 1: Enable Gateway API on the Cluster

**Important**: This must be done manually via GCloud Console or gcloud CLI.

```bash
gcloud container clusters update CLUSTER_NAME \
  --zone=ZONE \
  --gateway-api=standard
```

Wait for the cluster update to complete (~5-10 minutes).

Verify Gateway API is enabled:
```bash
kubectl get gatewayclass
```

You should see:
```
NAME                               CONTROLLER                  ACCEPTED   AGE
gke-l7-global-external-managed     networking.gke.io/gateway   True       ...
gke-l7-gxlb                        networking.gke.io/gateway   True       ...
gke-l7-regional-external-managed   networking.gke.io/gateway   True       ...
gke-l7-rilb                        networking.gke.io/gateway   True       ...
```

### Step 2: Install Gateway API Inference Extension CRDs

```bash
kubectl kustomize "github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd?ref=v1.0.1" | kubectl apply -f -
```

Verify CRDs are installed:
```bash
kubectl get crd | grep inference
```

You should see:
```
inferenceobjectives.inference.networking.x-k8s.io
inferencepools.inference.networking.k8s.io
inferencepools.inference.networking.x-k8s.io
```

### Step 3: Create HuggingFace Secret

```bash
kubectl create secret generic hf-secret \
  --from-literal=hf_api_token=YOUR_HF_TOKEN
```

Replace `YOUR_HF_TOKEN` with your actual HuggingFace token.

### Step 4: Create the ComputeClass

```bash
kubectl apply -f gpu-2-compute-class.yaml
```

Verify ComputeClass is created:
```bash
kubectl get computeclass gpu-2-spot
```

### Step 5: Deploy vLLM with Gemma 3 1B

```bash
kubectl apply -f vllm-gemma-deployment.yaml
```

This will:
- Create a Deployment with 1 replica
- Create a Service (ClusterIP)
- Create a HorizontalPodAutoscaler
- Auto-provision a g2-standard-8 spot instance via the ComputeClass

Wait for the deployment and node provisioning:
```bash
# Watch node provisioning
kubectl get nodes -l cloud.google.com/compute-class=gpu-2-spot -w

# Watch pod status
kubectl get pods -l app=vllm-gemma-3-1b -w
```

The pod should reach `Running` status in 3-5 minutes (includes node provisioning and model download).

### Step 6: Deploy Body-Based Routing Extension

```bash
helm install bbr \
  --version v1.0.1 \
  --set provider.name=gke \
  --set inferenceGateway.name=vllm-xlb \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/body-based-routing
```

Verify deployment:
```bash
kubectl get pods -l app=body-based-router
```

### Step 7: Deploy Inference Pool with Endpoint Picker

```bash
INFERENCE_POOL=vllm-gemma-3-1b
helm install ${INFERENCE_POOL} \
  --set inferencePool.modelServers.matchLabels.app=vllm-gemma-3-1b \
  --set provider.name=gke \
  --version v1.0.1 \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool \
  -f epp-values.yaml
```

The Endpoint Picker Extension (EPP) will deploy with 3 replicas by default. Scale it down to 1:
```bash
kubectl scale deployment vllm-gemma-3-1b-epp --replicas=1
```

Wait for EPP to be ready:
```bash
kubectl wait --for=condition=ready --timeout=120s pod -l inferencepool=vllm-gemma-3-1b-epp
```

Verify InferencePool:
```bash
kubectl get inferencepool
kubectl get pods -l inferencepool=vllm-gemma-3-1b-epp
```

### Step 8: Deploy Inference Objective

```bash
kubectl apply -f inference-objective.yaml
```

Verify:
```bash
kubectl get inferenceobjective gemma-3-1b-it-hf
```

### Step 9: Deploy Internal Gateway

```bash
kubectl apply -f gateway.yaml
```

Wait for the Gateway to be programmed (provisions an internal load balancer):
```bash
kubectl wait --for=condition=Programmed --timeout=300s gateway/vllm-xlb
```

Get the internal IP:
```bash
kubectl get gateway vllm-xlb -o jsonpath='{.status.addresses[0].value}'
```

### Step 10: Deploy HTTPRoute

```bash
kubectl apply -f httproute.yaml
```

Verify:
```bash
kubectl get httproute vllm-gemma-3-1b-route
```

### Step 11: (Optional) Deploy Gradio Chat UI

Deploy a web-based chat interface to interact with the Gemma model:

```bash
kubectl apply -f gradio.yaml
```

Wait for the LoadBalancer to get an IP:
```bash
kubectl get svc gradio -w
```

Access the Gradio UI:
- **Via LoadBalancer**: `http://EXTERNAL-IP:8080`
- **Via Port-forward**: 
  ```bash
  kubectl port-forward svc/gradio 8080:8080
  ```
  Then open http://localhost:8080

### Step 12: (Optional) Deploy Hermes Agents (Demo)

Deploy Hermes agent containers on GPU and TPU nodes for demo purposes:

```bash
# Create API key secret
kubectl create secret generic hermes-api-key \
  --from-literal=api-key="$(openssl rand -hex 32)"

# Deploy 10 replicas distributed across GPU/TPU nodes
kubectl apply -f agent.yaml
```

Check deployment:
```bash
# View pods and their node placement
kubectl get pods -l app=hermes-agent -o wide

# Count distribution
kubectl get pods -l app=hermes-agent -o wide | \
  awk 'NR>1 {if ($7~/gpu/) gpu++; else if ($7~/tpu/) tpu++} END {print "GPU nodes:", gpu; print "TPU nodes:", tpu}'
```

**Note**: The Hermes agents in this configuration run `sleep 356d` for demo purposes. To make them functional, change the command in `agent.yaml` to `["gateway", "run"]` and add health probes back.

## Testing the Deployment

### Test from within the cluster

Get the Gateway IP:
```bash
GATEWAY_IP=$(kubectl get gateway vllm-xlb -o jsonpath='{.status.addresses[0].value}')
echo "Gateway IP: $GATEWAY_IP"
```

Run a test pod with curl:
```bash
kubectl run -it --rm test-client --image=curlimages/curl --restart=Never -- curl -X POST http://$GATEWAY_IP/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "google/gemma-3-1b-it", "messages": [{"role": "user", "content": "What is Kubernetes?"}], "max_tokens": 100}'
```

### Interactive testing

For multiple tests, use an interactive shell:

```bash
kubectl run -it --rm test-client --image=curlimages/curl --restart=Never -- sh
```

Then inside the pod:
```bash
# Get Gateway IP (replace with actual IP)
GATEWAY_IP=10.128.0.203

# Simple test
curl -X POST http://$GATEWAY_IP/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/gemma-3-1b-it",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# More complex query
curl -X POST http://$GATEWAY_IP/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/gemma-3-1b-it",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Expected Response

A successful response should look like:
```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "created": 1234567890,
  "model": "google/gemma-3-1b-it",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Kubernetes is an open-source container orchestration platform..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 50,
    "total_tokens": 65
  }
}
```

### Testing with Gradio UI

If you deployed Gradio (optional), access the web interface:

1. **Get the LoadBalancer IP:**
   ```bash
   GRADIO_IP=$(kubectl get svc gradio -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
   echo "Gradio UI: http://$GRADIO_IP:8080"
   ```

2. **Or use port-forward:**
   ```bash
   kubectl port-forward svc/gradio 8080:8080
   ```
   Then open http://localhost:8080

3. **Use the chat interface:**
   - Type your message in the chat box
   - Press Enter or click Send
   - The response will come from the Gemma model via the inference gateway

**Configuration:**
- **Backend**: Internal Gateway at `http://10.128.0.203`
- **Model**: `google/gemma-3-1b-it`
- **API**: OpenAI-compatible chat completions

## Verification and Status Checks

### Check all components

```bash
# 1. ComputeClass
kubectl get computeclass gpu-2-spot

# 2. Auto-provisioned node
kubectl get nodes -l cloud.google.com/compute-class=gpu-2-spot

# 3. vLLM Deployment
kubectl get deployment vllm-gemma-3-1b
kubectl get pods -l app=vllm-gemma-3-1b

# 4. Body-Based Router
kubectl get pods -l app=body-based-router

# 5. Inference Pool EPP
kubectl get pods -l inferencepool=vllm-gemma-3-1b-epp

# 6. InferencePool
kubectl get inferencepool

# 7. InferenceObjective
kubectl get inferenceobjective

# 8. Gateway
kubectl get gateway vllm-xlb

# 9. HTTPRoute
kubectl get httproute
```

### Complete deployment status

```bash
echo "======================================"
echo "  Deployment Status"
echo "======================================"
kubectl get computeclass gpu-2-spot
kubectl get nodes -l cloud.google.com/compute-class=gpu-2-spot
kubectl get deployment vllm-gemma-3-1b
kubectl get pods -l app=vllm-gemma-3-1b
kubectl get pods -l app=body-based-router
kubectl get pods -l inferencepool=vllm-gemma-3-1b-epp
kubectl get inferencepool
kubectl get inferenceobjective
kubectl get gateway vllm-xlb
kubectl get httproute

# Optional components
echo ""
echo "Optional Components:"
kubectl get deployment gradio 2>/dev/null && echo "✓ Gradio UI deployed"
kubectl get deployment hermes-agent 2>/dev/null && echo "✓ Hermes agents deployed"
```

## Monitoring

### Check Pod Status

```bash
# vLLM pods
kubectl get pods -l app=vllm-gemma-3-1b

# Body-Based Router
kubectl get pods -l app=body-based-router

# Endpoint Picker Extension (EPP)
kubectl get pods -l inferencepool=vllm-gemma-3-1b-epp
```

### Check Logs

```bash
# vLLM logs (check for model loading and health checks)
kubectl logs -l app=vllm-gemma-3-1b --tail=50

# BBR logs (check for request routing)
kubectl logs -l app=body-based-router --tail=50

# EPP logs (check for endpoint selection)
kubectl logs -l inferencepool=vllm-gemma-3-1b-epp --tail=50

# Follow logs in real-time
kubectl logs -l app=vllm-gemma-3-1b -f
```

### Check Gateway Status

```bash
# Gateway details
kubectl get gateway vllm-xlb -o yaml

# Gateway IP address
kubectl get gateway vllm-xlb -o jsonpath='{.status.addresses[0].value}'

# Gateway conditions
kubectl get gateway vllm-xlb -o jsonpath='{.status.conditions[*].type}' && echo
```

### Check vLLM Health

```bash
# Get pod name
POD_NAME=$(kubectl get pods -l app=vllm-gemma-3-1b -o jsonpath='{.items[0].metadata.name}')

# Port-forward to vLLM
kubectl port-forward $POD_NAME 8000:8000

# In another terminal, check health endpoint
curl http://localhost:8000/health

# Check vLLM metrics
curl http://localhost:8000/metrics
```

## Autoscaling

The HorizontalPodAutoscaler is configured to scale based on the `inference_pool_average_kv_cache_utilization` metric:

- **Min replicas**: 1
- **Max replicas**: 2
- **Target**: 10m (10 millicores) average KV cache utilization

Check HPA status:

```bash
kubectl get hpa vllm-gemma-3-1b
kubectl describe hpa vllm-gemma-3-1b
```

## ComputeClass Benefits

Using the `gpu-2-spot` ComputeClass provides:

1. **Automatic Node Provisioning**: Nodes are created automatically when pods are scheduled
2. **Cost Optimization**: Uses spot instances for lower costs
3. **Resource Matching**: Ensures g2-standard-8 machines with L4 GPUs
4. **Simplified Management**: No manual node pool creation needed

## Cleanup

To remove the deployment, follow these steps in order:

### Step 1: Delete Optional Components

```bash
# Delete Gradio UI (if deployed)
kubectl delete -f gradio.yaml 2>/dev/null || echo "Gradio not deployed"

# Delete Hermes agents (if deployed)
kubectl delete -f agent.yaml 2>/dev/null || echo "Hermes agents not deployed"
kubectl delete secret hermes-api-key 2>/dev/null || echo "Hermes secret not found"
```

### Step 2: Delete HTTPRoute and Gateway

```bash
# Delete HTTPRoute first
kubectl delete -f httproute.yaml

# Delete Gateway (this removes the load balancer)
kubectl delete -f gateway.yaml
```

Wait for the Gateway to be fully deleted before proceeding.

### Step 3: Delete Inference Components

```bash
# Delete Inference Objective
kubectl delete -f inference-objective.yaml

# Uninstall Inference Pool (this removes EPP)
helm uninstall vllm-gemma-3-1b

# Uninstall Body-Based Router
helm uninstall bbr
```

### Step 4: Delete vLLM Deployment

```bash
# Delete deployment, service, and HPA
kubectl delete -f vllm-gemma-deployment.yaml
```

### Step 5: Delete Auto-provisioned Node (Optional)

The node will be automatically removed after the pods are deleted, but you can also manually delete it:

```bash
# Get node name
NODE_NAME=$(kubectl get nodes -l cloud.google.com/compute-class=gpu-2-spot -o jsonpath='{.items[0].metadata.name}')

# Drain and delete the node (if exists)
if [ ! -z "$NODE_NAME" ]; then
  kubectl drain $NODE_NAME --ignore-daemonsets --delete-emptydir-data
  kubectl delete node $NODE_NAME
fi
```

### Step 6: Delete ComputeClass (Optional)

Only delete the ComputeClass if you no longer need it:

```bash
kubectl delete -f gpu-2-compute-class.yaml
```

### Complete Cleanup Script

```bash
#!/bin/bash
echo "Cleaning up Gemma Inference deployment..."

# Delete optional components
kubectl delete -f gradio.yaml 2>/dev/null
kubectl delete -f agent.yaml 2>/dev/null
kubectl delete secret hermes-api-key 2>/dev/null

# Delete in reverse order
kubectl delete -f httproute.yaml
kubectl delete -f gateway.yaml
kubectl delete -f inference-objective.yaml
helm uninstall vllm-gemma-3-1b
helm uninstall bbr
kubectl delete -f vllm-gemma-deployment.yaml

# Wait for resources to be deleted
sleep 10

# Check for remaining resources
echo "Remaining resources:"
kubectl get pods -l app=vllm-gemma-3-1b
kubectl get pods -l app=body-based-router
kubectl get pods -l inferencepool=vllm-gemma-3-1b-epp
kubectl get pods -l app=gradio 2>/dev/null
kubectl get pods -l app=hermes-agent 2>/dev/null

echo "Cleanup complete!"
```

### Verify Cleanup

```bash
# Check for remaining pods
kubectl get pods -l app=vllm-gemma-3-1b
kubectl get pods -l app=body-based-router
kubectl get pods -l inferencepool=vllm-gemma-3-1b-epp
kubectl get pods -l app=gradio 2>/dev/null
kubectl get pods -l app=hermes-agent 2>/dev/null

# Check for remaining gateway
kubectl get gateway

# Check for remaining services
kubectl get svc gradio 2>/dev/null
kubectl get svc hermes-agent 2>/dev/null

# Check for remaining nodes
kubectl get nodes -l cloud.google.com/compute-class=gpu-2-spot
```

## Troubleshooting

### Issue 1: Gateway stuck in "Unknown" status

**Symptoms:**
```bash
kubectl get gateway vllm-xlb
# Shows: PROGRAMMED = Unknown
```

**Cause:** Gateway API not enabled on the cluster, or wrong GatewayClass.

**Solution:**
1. Enable Gateway API on the cluster (see Step 1 in Setup)
2. Verify GatewayClasses are available:
   ```bash
   kubectl get gatewayclass
   ```
3. For internal load balancer, use `gke-l7-rilb` (not `gke-l7-regional-internal-managed`)
4. Delete and recreate the gateway:
   ```bash
   kubectl delete gateway vllm-xlb
   kubectl apply -f gateway.yaml
   ```

### Issue 2: EPP pods crashing (CrashLoopBackOff)

**Symptoms:**
```bash
kubectl get pods -l inferencepool=vllm-gemma-3-1b-epp
# Shows: CrashLoopBackOff
```

**Cause:** InferenceObjective CRD not installed.

**Solution:**
1. Install the Gateway API Inference Extension CRDs:
   ```bash
   kubectl kustomize "github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd?ref=v1.0.1" | kubectl apply -f -
   ```
2. Delete EPP pods to force recreation:
   ```bash
   kubectl delete pods -l inferencepool=vllm-gemma-3-1b-epp
   ```
3. Wait for pods to restart:
   ```bash
   kubectl get pods -l inferencepool=vllm-gemma-3-1b-epp -w
   ```

### Issue 3: Pods not scheduling on gpu-2-spot nodes

**Symptoms:**
```bash
kubectl get pods -l app=vllm-gemma-3-1b
# Shows: Pending
kubectl describe pod -l app=vllm-gemma-3-1b
# Shows: node(s) didn't match Pod's node affinity/selector
```

**Cause:** No nodes with the `cloud.google.com/compute-class=gpu-2-spot` label exist.

**Solution:**
1. Check if ComputeClass exists:
   ```bash
   kubectl get computeclass gpu-2-spot
   ```
2. Verify `whenUnsatisfiable: ScaleUpAnyway` is set:
   ```bash
   kubectl get computeclass gpu-2-spot -o yaml | grep whenUnsatisfiable
   ```
3. Check for node provisioning events:
   ```bash
   kubectl get events --sort-by='.lastTimestamp' | grep -i node
   ```
4. Wait 2-3 minutes for node auto-provisioning to create a new node

### Issue 4: vLLM pod stuck in ContainerCreating

**Symptoms:**
```bash
kubectl get pods -l app=vllm-gemma-3-1b
# Shows: ContainerCreating for several minutes
```

**Possible Causes:**
- Model download in progress (Gemma 3 1B is ~2GB)
- HuggingFace token missing or invalid
- Image pull issues

**Solution:**
1. Check pod events:
   ```bash
   kubectl describe pod -l app=vllm-gemma-3-1b
   ```
2. Check logs (if available):
   ```bash
   kubectl logs -l app=vllm-gemma-3-1b --tail=50
   ```
3. Verify HuggingFace secret exists:
   ```bash
   kubectl get secret hf-secret
   ```
4. Wait 3-5 minutes for model download to complete

### Issue 5: Gateway shows org policy constraint error

**Symptoms:**
```bash
kubectl describe gateway vllm-xlb
# Shows: Constraint constraints/compute.restrictLoadBalancerCreationForTypes violated
```

**Cause:** Trying to create an external load balancer when org policy only allows internal.

**Solution:**
Use `gke-l7-rilb` GatewayClass for internal load balancer:
```bash
kubectl delete gateway vllm-xlb
# Edit gateway.yaml to use gke-l7-rilb
kubectl apply -f gateway.yaml
```

### Issue 6: 503 errors when testing inference

**Symptoms:**
```bash
curl http://GATEWAY_IP/v1/chat/completions ...
# Returns: 503 Service Unavailable
```

**Possible Causes:**
- vLLM pod not ready
- InferencePool not configured correctly
- HTTPRoute not attached to Gateway

**Solution:**
1. Check vLLM pod status:
   ```bash
   kubectl get pods -l app=vllm-gemma-3-1b
   kubectl logs -l app=vllm-gemma-3-1b --tail=20
   ```
2. Check InferencePool:
   ```bash
   kubectl get inferencepool vllm-gemma-3-1b -o yaml
   ```
3. Check HTTPRoute attachment:
   ```bash
   kubectl get httproute vllm-gemma-3-1b-route -o yaml
   ```
4. Verify Gateway has the HTTPRoute attached:
   ```bash
   kubectl get gateway vllm-xlb -o jsonpath='{.status.listeners[0].attachedRoutes}'
   # Should show: 1
   ```

### General Debugging Commands

```bash
# Check all resources
kubectl get all -l app=vllm-gemma-3-1b

# Check events in the namespace
kubectl get events --sort-by='.lastTimestamp' | tail -20

# Check node resources
kubectl describe node -l cloud.google.com/compute-class=gpu-2-spot

# Check GPU allocation
kubectl describe node -l cloud.google.com/compute-class=gpu-2-spot | grep -A 5 "Allocated resources"
```

## Quick Reference

### Common Commands

```bash
# Get Gateway IP
kubectl get gateway vllm-xlb -o jsonpath='{.status.addresses[0].value}'

# Check vLLM pod status
kubectl get pods -l app=vllm-gemma-3-1b

# Check vLLM logs
kubectl logs -l app=vllm-gemma-3-1b --tail=50

# Check all inference components
kubectl get inferencepool,inferenceobjective,gateway,httproute

# Scale EPP replicas
kubectl scale deployment vllm-gemma-3-1b-epp --replicas=1

# Port-forward to vLLM for direct access
kubectl port-forward svc/vllm-gemma-3-1b 8000:8000

# Gradio UI commands
kubectl get svc gradio
kubectl port-forward svc/gradio 8080:8080
kubectl logs -l app=gradio --tail=20

# Hermes agents commands
kubectl get pods -l app=hermes-agent -o wide
kubectl get secret hermes-api-key -o jsonpath='{.data.api-key}' | base64 -d
```

### Test Command

**One-line test:**
```bash
kubectl run -it --rm test-client --image=curlimages/curl --restart=Never -- curl -X POST http://10.128.0.203/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "google/gemma-3-1b-it", "messages": [{"role": "user", "content": "What is Kubernetes?"}], "max_tokens": 100}'
```

**Interactive shell:**
```bash
kubectl run -it --rm test-client --image=curlimages/curl --restart=Never -- sh
```

Then inside:
```bash
curl -X POST http://10.128.0.203/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/gemma-3-1b-it",
    "messages": [{"role": "user", "content": "What is Kubernetes?"}],
    "max_tokens": 100
  }'
```

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Client Applications                                         │
├─────────────────────────────────────────────────────────────┤
│  • Gradio Web UI (http://EXTERNAL-IP:8080)                 │
│  • Hermes Agents (10 replicas on GPU/TPU nodes)            │
│  • Direct API calls (kubectl run test-client...)           │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Internal Load Balancer (10.128.0.203)                      │
│ Gateway (gke-l7-rilb)                                       │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Body-Based Router (extracts model name from JSON body)     │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ HTTPRoute (routes by X-Gateway-Model-Name header)          │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ InferenceObjective (priority 10) → InferencePool           │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Endpoint Picker Extension (intelligent selection)          │
│  • Queue depth scoring                                      │
│  • KV cache utilization                                     │
│  • Prefix cache matching                                    │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ vLLM Pod (Gemma 3 1B on L4 GPU)                            │
│  • Runs on gpu-2-spot ComputeClass node                    │
│  • Auto-scaled based on KV cache utilization               │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
                      Response
```

### Key Metrics

- **Model**: google/gemma-3-1b-it (~2GB)
- **GPU**: NVIDIA L4 (g2-standard-8)
- **Node Provisioning**: Auto-provisioned via ComputeClass
  - Priority 1: Spot instances (cost-optimized)
  - Priority 2: On-demand instances (reliability fallback)
- **ComputeClass**: gpu-2-spot with `whenUnsatisfiable: ScaleUpAnyway`
- **Gateway Type**: Internal Regional Load Balancer (gke-l7-rilb)
- **Gateway IP**: 10.128.0.203 (internal only)
- **Auto-scaling**: HPA based on KV cache utilization (1-2 replicas)
- **Optional Components**:
  - Gradio UI: 1 replica, LoadBalancer on port 8080
  - Hermes Agents: 10 replicas across GPU/TPU nodes (demo)

## References

- [GKE Inference Gateway Documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/deploy-gke-inference-gateway)
- [ComputeClass Documentation](https://cloud.google.com/kubernetes-engine/docs/concepts/about-custom-compute-classes)
- [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Gateway API Documentation](https://gateway-api.sigs.k8s.io/)
- [Gemma Model Documentation](https://ai.google.dev/gemma)
