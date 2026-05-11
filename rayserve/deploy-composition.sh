#!/bin/bash
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Deploy Model Composition: Real-ESRGAN (TPU) → RepVGG (GPU)
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE=${NAMESPACE:-default}
CONFIGMAP_NAME="composition-apps"
RAYSERVICE_NAME="composition-esrgan-repvgg"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Model Composition Deployment${NC}"
echo -e "${BLUE}Real-ESRGAN (TPU) → RepVGG (GPU)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Step 1: Check prerequisites
echo -e "${YELLOW}[1/4] Checking prerequisites...${NC}"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl not found. Please install kubectl.${NC}"
    exit 1
fi

# Check if we can connect to cluster
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}Error: Cannot connect to Kubernetes cluster.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ kubectl connected to cluster${NC}"

# Step 2: Create/Update ConfigMap
echo -e "\n${YELLOW}[2/4] Creating ConfigMap...${NC}"

if kubectl get configmap $CONFIGMAP_NAME -n $NAMESPACE &> /dev/null; then
    echo -e "${YELLOW}ConfigMap $CONFIGMAP_NAME already exists. Updating...${NC}"
    kubectl delete configmap $CONFIGMAP_NAME -n $NAMESPACE
fi

kubectl create configmap $CONFIGMAP_NAME \
  --from-file=composition_esrgan_repvgg_app.py=composition_esrgan_repvgg_app.py \
  --from-file=realesrgan_tpu_backend.py=realesrgan_tpu_backend.py \
  --from-file=repvgg_gpu_backend.py=repvgg_gpu_backend.py \
  -n $NAMESPACE

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ ConfigMap created: $CONFIGMAP_NAME${NC}"
else
    echo -e "${RED}✗ Failed to create ConfigMap${NC}"
    exit 1
fi

# Step 3: Deploy RayService
echo -e "\n${YELLOW}[3/4] Deploying RayService...${NC}"

# Check if RayService already exists
if kubectl get rayservice $RAYSERVICE_NAME -n $NAMESPACE &> /dev/null; then
    echo -e "${YELLOW}RayService $RAYSERVICE_NAME already exists. Deleting...${NC}"
    kubectl delete rayservice $RAYSERVICE_NAME -n $NAMESPACE
    echo "Waiting for cleanup..."
    sleep 10
fi

kubectl apply -f composition-esrgan-repvgg.rayservice.yaml

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ RayService deployed: $RAYSERVICE_NAME${NC}"
else
    echo -e "${RED}✗ Failed to deploy RayService${NC}"
    exit 1
fi

# Step 4: Wait for pods to be ready
echo -e "\n${YELLOW}[4/4] Waiting for pods to be ready...${NC}"
echo "This may take 2-3 minutes..."
echo ""

# Wait for RayService to be ready
kubectl wait --for=condition=Ready rayservice/$RAYSERVICE_NAME -n $NAMESPACE --timeout=300s 2>/dev/null || true

echo -e "\n${BLUE}Checking pod status:${NC}"
kubectl get pods -n $NAMESPACE -l ray.io/cluster=$RAYSERVICE_NAME

# Get head pod
HEAD_POD=$(kubectl get pod -n $NAMESPACE -l ray.io/node-type=head,ray.io/cluster=$RAYSERVICE_NAME -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$HEAD_POD" ]; then
    echo -e "${YELLOW}Warning: Head pod not found yet. Cluster may still be starting.${NC}"
else
    echo -e "\n${GREEN}✓ Head pod found: $HEAD_POD${NC}"

    # Wait a bit more for serve to be ready
    echo "Waiting for Ray Serve to initialize..."
    sleep 20

    # Check Ray Serve status
    echo -e "\n${BLUE}Checking Ray Serve status:${NC}"
    kubectl exec -n $NAMESPACE $HEAD_POD -- ray serve status 2>/dev/null || echo "Serve status not available yet"

    # Test health endpoint
    echo -e "\n${BLUE}Testing composition health endpoint:${NC}"
    kubectl exec -n $NAMESPACE $HEAD_POD -- curl -s http://localhost:8000/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Not ready yet"
fi

# Summary
echo -e "\n${BLUE}============================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "RayService: $RAYSERVICE_NAME"
echo "Namespace: $NAMESPACE"
echo "ConfigMap: $CONFIGMAP_NAME"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Check deployment status:"
echo "   kubectl get rayservice $RAYSERVICE_NAME -n $NAMESPACE"
echo ""
echo "2. Monitor pods:"
echo "   kubectl get pods -n $NAMESPACE -l ray.io/cluster=$RAYSERVICE_NAME -w"
echo ""
echo "3. Check serve status:"
echo "   kubectl exec -n $NAMESPACE $HEAD_POD -- ray serve status"
echo ""
echo "4. Test the pipeline:"
echo "   python test_composition.py <image_path>"
echo ""
echo "5. View logs:"
echo "   kubectl logs -n $NAMESPACE $HEAD_POD --tail=100"
echo ""
echo -e "${BLUE}API Endpoint:${NC}"
echo "  POST http://localhost:8000/classify_upscaled"
echo ""
echo -e "${BLUE}Pipeline:${NC}"
echo "  User Image → Real-ESRGAN (TPU) → RepVGG (GPU) → Classification"
echo ""
