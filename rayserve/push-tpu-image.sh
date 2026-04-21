#!/bin/bash
# Script to push TPU Docker image to Artifact Registry

set -e

# Configuration
PROJECT_ID="gpu-launchpad-playground"
REGION="us-central1"
REPO="gech-ray-gke"
IMAGE_NAME="ray-tpu"
TAG="2.55.1"

FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"

echo "==========================================================="
echo "Push Ray TPU Image to Artifact Registry"
echo "==========================================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Repository: ${REPO}"
echo "Full image: ${FULL_IMAGE}"
echo "==========================================================="
echo ""

# Configure Docker authentication for Artifact Registry
echo "Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Check if image exists locally
if docker images ${FULL_IMAGE} | grep -q ${TAG}; then
    echo "✓ Image found locally"
else
    echo "✗ Image not found locally. Please build it first:"
    echo "  docker build -t ${FULL_IMAGE} -f Dockerfile.ray-tpu ."
    exit 1
fi

# Push image
echo ""
echo "Pushing image to Artifact Registry..."
docker push ${FULL_IMAGE}

# Tag as latest
echo ""
echo "Tagging as latest..."
docker tag ${FULL_IMAGE} ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:latest
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:latest

echo ""
echo "==========================================================="
echo "✓ Image pushed successfully!"
echo "==========================================================="
echo ""
echo "Image URIs:"
echo "  - ${FULL_IMAGE}"
echo "  - ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:latest"
echo ""
echo "To use in RayService, update the image in gpu-tpu-app.rayservice.yaml:"
echo "  spec.rayClusterConfig.workerGroupSpecs[tpu-group].template.spec.containers[0].image"
echo ""
