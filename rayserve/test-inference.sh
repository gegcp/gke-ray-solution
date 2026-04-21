#!/bin/bash
# Test GPU and TPU inference endpoints
# Usage: ./test-inference.sh [--namespace default]

set -e

NAMESPACE="${1:-default}"

echo "==========================================================="
echo "Testing GPU and TPU Inference Endpoints"
echo "==========================================================="
echo "Namespace: ${NAMESPACE}"
echo ""

# Get head pod
HEAD_POD=$(kubectl get pods -n ${NAMESPACE} -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$HEAD_POD" ]; then
    echo "❌ Error: No head pod found in namespace ${NAMESPACE}"
    exit 1
fi

echo "Head pod: ${HEAD_POD}"
echo ""

# Create test script in pod
kubectl exec -n ${NAMESPACE} ${HEAD_POD} -- bash -c 'cat > /tmp/test_inference.py << '\''PYTHON'\''
#!/usr/bin/env python3
import base64
import json
import subprocess
import time
from io import BytesIO
from PIL import Image
import numpy as np

print("=" * 60)
print("GPU and TPU Inference Test")
print("=" * 60)
print()

# Create test image
print("Creating test image (224x224 RGB)...")
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
buffer = BytesIO()
img.save(buffer, format="PNG")
img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
print(f"✓ Image size: {len(img_b64)} bytes (base64)")
print()

# Create payload file
payload = {"image": img_b64, "top_k": 5, "img_size": 224}
with open("/tmp/test_payload.json", "w") as f:
    json.dump(payload, f)

# Test GPU endpoint
print("=" * 60)
print("GPU Inference (/gpu/classify)")
print("=" * 60)
start_time = time.time()
result = subprocess.run([
    "curl", "-s", "-X", "POST",
    "http://localhost:8000/gpu/classify",
    "-H", "Content-Type: application/json",
    "-d", "@/tmp/test_payload.json"
], capture_output=True, text=True)
gpu_time = (time.time() - start_time) * 1000

if result.returncode == 0:
    try:
        gpu_result = json.loads(result.stdout)
        print(f"✅ Status: Success")
        print(f"⏱️  Latency: {gpu_time:.2f}ms")
        print(f"🔧 Device: {gpu_result.get('\''device'\'', '\''unknown'\'')}")
        print(f"🏷️  Top 3 predictions:")
        for i, pred in enumerate(gpu_result.get("predictions", [])[:3], 1):
            print(f"   {i}. Class {pred['\''class_id'\'']}: {pred['\''probability'\'']:.6f}")
    except Exception as e:
        print(f"❌ Error parsing response: {e}")
        print(result.stdout[:500])
else:
    print(f"❌ Request failed: {result.stderr}")

print()

# Test TPU endpoint
print("=" * 60)
print("TPU Inference (/tpu/classify)")
print("=" * 60)
start_time = time.time()
result = subprocess.run([
    "curl", "-s", "-X", "POST",
    "http://localhost:8000/tpu/classify",
    "-H", "Content-Type: application/json",
    "-d", "@/tmp/test_payload.json"
], capture_output=True, text=True)
tpu_time = (time.time() - start_time) * 1000

if result.returncode == 0:
    try:
        tpu_result = json.loads(result.stdout)
        if "error" in tpu_result:
            print(f"❌ Error: {tpu_result['\''error'\'']}")
        else:
            print(f"✅ Status: Success")
            print(f"⏱️  Latency: {tpu_time:.2f}ms")
            print(f"🔧 Device: {tpu_result.get('\''device'\'', '\''unknown'\'')}")
            print(f"🏷️  Top 3 predictions:")
            for i, pred in enumerate(tpu_result.get("predictions", [])[:3], 1):
                print(f"   {i}. Class {pred['\''class_id'\'']}: {pred['\''probability'\'']:.6f}")

            # Performance comparison
            if gpu_time > 0 and tpu_time > 0:
                speedup = gpu_time / tpu_time
                print()
                print("📊 Performance:")
                print(f"   GPU: {gpu_time:.2f}ms")
                print(f"   TPU: {tpu_time:.2f}ms")
                print(f"   Speedup: {speedup:.2f}x ({'\''+'\''TPU faster'\'' if speedup > 1 else '\''GPU faster'\''+'\''})")
    except Exception as e:
        print(f"❌ Error parsing response: {e}")
        print(result.stdout[:500])
else:
    print(f"❌ Request failed: {result.stderr}")

print()
print("=" * 60)
print("✅ Test Complete!")
print("=" * 60)
PYTHON
' 2>/dev/null

# Run the test
echo "Running inference test..."
echo ""
kubectl exec -n ${NAMESPACE} ${HEAD_POD} -- python3 /tmp/test_inference.py

# Cleanup
kubectl exec -n ${NAMESPACE} ${HEAD_POD} -- rm -f /tmp/test_inference.py /tmp/test_payload.json 2>/dev/null || true

echo ""
echo "==========================================================="
echo "Health Check Summary"
echo "==========================================================="
echo ""

# Check health endpoints
echo "GPU Health:"
kubectl exec -n ${NAMESPACE} ${HEAD_POD} -- curl -s http://localhost:8000/gpu/health 2>/dev/null | python3 -m json.tool || echo "❌ Not available"

echo ""
echo "TPU Health:"
kubectl exec -n ${NAMESPACE} ${HEAD_POD} -- curl -s http://localhost:8000/tpu/health 2>/dev/null | python3 -m json.tool || echo "❌ Not available"

echo ""
