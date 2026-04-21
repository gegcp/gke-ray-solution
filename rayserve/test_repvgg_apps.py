#!/usr/bin/env python3
"""
Test script for RepVGG GPU and TPU inference apps
"""

import base64
import json
import requests
import sys
from pathlib import Path


def encode_image_to_base64(image_path):
    """Encode image file to base64 string"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def test_endpoint(endpoint_url, image_path, top_k=5):
    """Test a RepVGG inference endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {endpoint_url}")
    print(f"Image: {image_path}")
    print(f"{'='*60}")

    # Check health endpoint first
    health_url = endpoint_url.rsplit('/', 1)[0] + '/health'
    try:
        health_response = requests.get(health_url, timeout=5)
        print(f"\n✓ Health check: {health_response.json()}")
    except Exception as e:
        print(f"\n✗ Health check failed: {e}")
        return

    # Encode image
    print(f"\nEncoding image...")
    image_base64 = encode_image_to_base64(image_path)

    # Prepare request
    payload = {
        "image": image_base64,
        "top_k": top_k,
        "img_size": 224
    }

    # Send inference request
    print(f"Sending inference request...")
    try:
        response = requests.post(
            endpoint_url,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()

        print(f"\n✓ Inference successful!")
        print(f"Model: {result.get('model', 'N/A')}")
        print(f"Device: {result.get('device', 'N/A')}")
        print(f"\nTop {top_k} Predictions:")
        print(f"{'-'*60}")

        for i, pred in enumerate(result.get('predictions', []), 1):
            class_id = pred.get('class_id', 'N/A')
            probability = pred.get('probability', 0) * 100
            print(f"{i}. Class ID: {class_id:4d}  Probability: {probability:6.2f}%")

        return result

    except requests.exceptions.Timeout:
        print(f"\n✗ Request timeout (30s)")
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Request failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def main():
    # Configuration
    base_url = "http://localhost:8000"
    gpu_endpoint = f"{base_url}/gpu/classify"
    tpu_endpoint = f"{base_url}/tpu/classify"

    # Find test image
    test_image = Path("/home/gech/workspace/oppo/gke-ray-solution/repvgg/gpu/burger.jpeg")

    if not test_image.exists():
        print(f"Error: Test image not found at {test_image}")
        print("Please provide an image path as argument:")
        print(f"  python {sys.argv[0]} /path/to/image.jpg")
        sys.exit(1)

    # Allow custom image path
    if len(sys.argv) > 1:
        test_image = Path(sys.argv[1])
        if not test_image.exists():
            print(f"Error: Image not found at {test_image}")
            sys.exit(1)

    print(f"\n{'#'*60}")
    print(f"# RepVGG Inference Test")
    print(f"{'#'*60}")
    print(f"\nBase URL: {base_url}")
    print(f"Test Image: {test_image}")

    # Test GPU endpoint
    print(f"\n\n{'*'*60}")
    print(f"* GPU INFERENCE TEST")
    print(f"{'*'*60}")
    gpu_result = test_endpoint(gpu_endpoint, test_image)

    # Test TPU endpoint
    print(f"\n\n{'*'*60}")
    print(f"* TPU INFERENCE TEST")
    print(f"{'*'*60}")
    tpu_result = test_endpoint(tpu_endpoint, test_image)

    print(f"\n\n{'='*60}")
    print(f"Testing complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
