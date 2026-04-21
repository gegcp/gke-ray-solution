#!/bin/bash
# Complete deployment script for GPU+TPU RayService
# Usage: ./deploy-tpu-rayservice.sh [--rebuild] [--skip-push] [--clean]

set -e

# Configuration
PROJECT_ID="gpu-launchpad-playground"
REGION="us-central1"
REPO="gech-ray-gke"
IMAGE_NAME="ray-tpu"
TAG="2.55.1"
NAMESPACE="default"

FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"

# Parse arguments
REBUILD=false
SKIP_PUSH=false
CLEAN_CACHE=false

for arg in "$@"; do
    case $arg in
        --rebuild)
            REBUILD=true
            shift
            ;;
        --skip-push)
            SKIP_PUSH=true
            shift
            ;;
        --clean)
            CLEAN_CACHE=true
            shift
            ;;
        *)
            ;;
    esac
done

echo "==========================================================="
echo "Deploy GPU+TPU RayService"
echo "==========================================================="
echo "Image: ${FULL_IMAGE}"
echo "Namespace: ${NAMESPACE}"
echo "Rebuild: ${REBUILD}"
echo "Skip push: ${SKIP_PUSH}"
echo "Clean cache: ${CLEAN_CACHE}"
echo "==========================================================="
echo ""

# Step 1: Build Docker image (if requested)
if [ "$REBUILD" = true ]; then
    echo "Step 1: Building Docker image..."
    echo "-----------------------------------------------------------"

    BUILD_ARGS=""
    if [ "$CLEAN_CACHE" = true ]; then
        BUILD_ARGS="--clean"
    fi

    ./build-tpu-image.sh $BUILD_ARGS

    if [ $? -ne 0 ]; then
        echo "❌ Build failed!"
        exit 1
    fi
    echo ""
else
    echo "Step 1: Skipping Docker build (use --rebuild to build)"
    echo ""
fi

# Step 2: Push image to Artifact Registry (if requested)
if [ "$SKIP_PUSH" = false ] && [ "$REBUILD" = true ]; then
    echo "Step 2: Pushing image to Artifact Registry..."
    echo "-----------------------------------------------------------"
    ./push-tpu-image.sh

    if [ $? -ne 0 ]; then
        echo "❌ Push failed!"
        exit 1
    fi
    echo ""
else
    if [ "$SKIP_PUSH" = true ]; then
        echo "Step 2: Skipping image push (--skip-push specified)"
    else
        echo "Step 2: Skipping image push (no rebuild)"
    fi
    echo ""
fi

# Step 3: Update ConfigMap with application code
echo "Step 3: Updating ConfigMap with application code..."
echo "-----------------------------------------------------------"

if [ ! -f "repvgg_gpu_app.py" ] || [ ! -f "repvgg_tpu_app.py" ]; then
    echo "❌ Error: Application files not found"
    echo "   Expected: repvgg_gpu_app.py, repvgg_tpu_app.py"
    exit 1
fi

kubectl create configmap repvgg-apps \
  --from-file=repvgg_gpu_app.py=repvgg_gpu_app.py \
  --from-file=repvgg_tpu_app.py=repvgg_tpu_app.py \
  -n ${NAMESPACE} \
  --dry-run=client -o yaml | kubectl apply -f -

echo "✓ ConfigMap updated"
echo ""

# Step 4: Check if RayService exists
echo "Step 4: Checking existing RayService..."
echo "-----------------------------------------------------------"

if kubectl get rayservice gpu-tpu-mix -n ${NAMESPACE} &> /dev/null; then
    echo "⚠ RayService 'gpu-tpu-mix' already exists"
    read -p "Delete and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing RayService..."
        kubectl delete rayservice gpu-tpu-mix -n ${NAMESPACE}
        echo "Waiting for cleanup..."
        sleep 10
    else
        echo "Keeping existing RayService. Updating ConfigMap only."
        echo ""
        echo "Note: You may need to restart pods to pick up ConfigMap changes:"
        echo "  kubectl delete pod -l ray.io/cluster -n ${NAMESPACE}"
        exit 0
    fi
else
    echo "No existing RayService found"
fi
echo ""

# Step 5: Deploy RayService
echo "Step 5: Deploying RayService..."
echo "-----------------------------------------------------------"

if [ ! -f "gpu-tpu-app.rayservice.yaml" ]; then
    echo "❌ Error: gpu-tpu-app.rayservice.yaml not found"
    exit 1
fi

kubectl apply -f gpu-tpu-app.rayservice.yaml

echo "✓ RayService deployed"
echo ""

# Step 6: Wait for pods to be ready
echo "Step 6: Waiting for pods to be ready..."
echo "-----------------------------------------------------------"

echo "Waiting for pods to start (this may take 2-3 minutes)..."
for i in {1..60}; do
    POD_COUNT=$(kubectl get pods -n ${NAMESPACE} -l ray.io/cluster 2>/dev/null | grep Running | wc -l)
    if [ "$POD_COUNT" -ge 3 ]; then
        echo "✓ Pods are running"
        break
    fi
    echo -n "."
    sleep 3
done
echo ""

kubectl get pods -n ${NAMESPACE} -l ray.io/cluster
echo ""

# Step 7: Verify deployment
echo "Step 7: Verifying deployment..."
echo "-----------------------------------------------------------"

echo "Waiting for Ray Serve to be ready..."
sleep 30

HEAD_POD=$(kubectl get pods -n ${NAMESPACE} -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$HEAD_POD" ]; then
    echo "❌ Error: Head pod not found"
    exit 1
fi

echo "Head pod: ${HEAD_POD}"
echo ""

# Check Ray Serve status
echo "Ray Serve status:"
kubectl exec -n ${NAMESPACE} ${HEAD_POD} -- ray serve status 2>/dev/null || echo "⚠ Ray Serve not ready yet"
echo ""

# Check health endpoints
echo "Health endpoints:"
echo "  GPU: $(kubectl exec -n ${NAMESPACE} ${HEAD_POD} -- curl -s http://localhost:8000/gpu/health 2>/dev/null || echo 'Not ready')"
echo "  TPU: $(kubectl exec -n ${NAMESPACE} ${HEAD_POD} -- curl -s http://localhost:8000/tpu/health 2>/dev/null || echo 'Not ready')"
echo ""

echo "==========================================================="
echo "✅ Deployment complete!"
echo "==========================================================="
echo ""
echo "Verification commands:"
echo "  # Check pods"
echo "  kubectl get pods -n ${NAMESPACE} -l ray.io/cluster"
echo ""
echo "  # Check Ray Serve status"
echo "  kubectl exec -n ${NAMESPACE} ${HEAD_POD} -- ray serve status"
echo ""
echo "  # Test GPU health"
echo "  kubectl exec -n ${NAMESPACE} ${HEAD_POD} -- curl -s http://localhost:8000/gpu/health | python3 -m json.tool"
echo ""
echo "  # Test TPU health"
echo "  kubectl exec -n ${NAMESPACE} ${HEAD_POD} -- curl -s http://localhost:8000/tpu/health | python3 -m json.tool"
echo ""
echo "  # View logs"
echo "  kubectl logs -n ${NAMESPACE} ${HEAD_POD} --tail=100"
echo ""
