#!/bin/bash
# Script to build Ray TPU Docker image
# Usage: ./build-tpu-image.sh [--no-cache] [--clean]

set -e

# Configuration
PROJECT_ID="gpu-launchpad-playground"
REGION="us-central1"
REPO="gech-ray-gke"
IMAGE_NAME="ray-tpu"
TAG="2.55.1"

FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"

# Parse arguments
CLEAN_CACHE=false
NO_CACHE=""

for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN_CACHE=true
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        *)
            ;;
    esac
done

echo "==========================================================="
echo "Build Ray TPU Image"
echo "==========================================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Repository: ${REPO}"
echo "Full image: ${FULL_IMAGE}"
echo "==========================================================="
echo ""

# Clean Docker cache if requested
if [ "$CLEAN_CACHE" = true ]; then
    echo "Cleaning Docker cache..."
    docker system prune -a -f --volumes
    echo "✓ Docker cache cleaned"
    echo ""
fi

# Check if Dockerfile exists
if [ ! -f "Dockerfile.ray-tpu" ]; then
    echo "❌ Error: Dockerfile.ray-tpu not found in current directory"
    exit 1
fi

# Build image
echo "Building Docker image..."
echo "Command: docker build -t ${FULL_IMAGE} -f Dockerfile.ray-tpu ${NO_CACHE} ."
echo ""

docker build -t ${FULL_IMAGE} -f Dockerfile.ray-tpu ${NO_CACHE} .

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "==========================================================="
    echo "✅ Build succeeded!"
    echo "==========================================================="
    echo ""
    echo "Image: ${FULL_IMAGE}"
    echo ""

    # Show image details
    echo "Image details:"
    docker images ${FULL_IMAGE} --format "table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.CreatedAt}}"
    echo ""

    # Verify JAX version
    echo "Verifying JAX version..."
    docker run --rm ${FULL_IMAGE} python3 -c "import jax; print('JAX version:', jax.__version__)"
    echo ""

    echo "Next steps:"
    echo "  1. Push image: ./push-tpu-image.sh"
    echo "  2. Update ConfigMap: kubectl create configmap repvgg-apps --from-file=... | kubectl apply -f -"
    echo "  3. Deploy RayService: kubectl apply -f gpu-tpu-app.rayservice.yaml"
    echo ""
else
    echo ""
    echo "==========================================================="
    echo "❌ Build failed!"
    echo "==========================================================="
    exit 1
fi
