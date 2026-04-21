#!/bin/bash
# Simple curl-based test for RepVGG apps

BASE_URL="http://localhost:8000"
IMAGE_PATH="${1:-/home/gech/workspace/oppo/gke-ray-solution/repvgg/gpu/burger.jpeg}"

echo "============================================================"
echo "Testing RepVGG Apps with curl"
echo "============================================================"
echo "Image: $IMAGE_PATH"
echo ""

# Encode image to base64
echo "Encoding image to base64..."
IMAGE_BASE64=$(base64 -w 0 "$IMAGE_PATH")

# Test GPU app health
echo ""
echo ">>> GPU Health Check"
curl -s "${BASE_URL}/gpu/health" | python3 -m json.tool

# Test TPU app health
echo ""
echo ">>> TPU Health Check"
curl -s "${BASE_URL}/tpu/health" | python3 -m json.tool

# Test GPU inference
echo ""
echo ">>> GPU Inference"
curl -s -X POST "${BASE_URL}/gpu/classify" \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"${IMAGE_BASE64}\", \"top_k\": 3}" | python3 -m json.tool

# Test TPU inference
echo ""
echo ">>> TPU Inference"
curl -s -X POST "${BASE_URL}/tpu/classify" \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"${IMAGE_BASE64}\", \"top_k\": 3}" | python3 -m json.tool

echo ""
echo "============================================================"
echo "Testing complete!"
echo "============================================================"
