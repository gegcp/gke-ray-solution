#!/usr/bin/env python3
"""
Test script for Real-ESRGAN GPU and TPU inference apps
"""

import base64
import json
import requests
import sys
from pathlib import Path
from PIL import Image
import io


def encode_image_to_base64(image_path):
    """Encode image file to base64 string"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def save_base64_image(base64_string, output_path):
    """Save base64 encoded image to file"""
    # Remove data URL prefix if present
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]

    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image.save(output_path)
    print(f"  Saved output image to: {output_path}")
    print(f"  Output size: {image.size[0]}x{image.size[1]}")


def test_endpoint(endpoint_url, image_path, output_dir, save_output=True):
    """Test a Real-ESRGAN inference endpoint"""
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

    # Get input image size
    input_image = Image.open(image_path)
    input_size = input_image.size
    print(f"\nInput image size: {input_size[0]}x{input_size[1]}")

    # Encode image
    print(f"Encoding image...")
    image_base64 = encode_image_to_base64(image_path)

    # Prepare request
    payload = {
        "image": image_base64,
        "return_image": True
    }

    # Send inference request
    print(f"Sending upscaling request...")
    try:
        response = requests.post(
            endpoint_url,
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()

        print(f"\n✓ Upscaling successful!")
        print(f"Model: {result.get('model', 'N/A')}")
        print(f"Device: {result.get('device', 'N/A')}")
        print(f"Scale: {result.get('scale', 'N/A')}x")

        input_info = result.get('input_size', {})
        output_info = result.get('output_size', {})

        print(f"\nInput size:  {input_info.get('width', 'N/A')}x{input_info.get('height', 'N/A')}")
        print(f"Output size: {output_info.get('width', 'N/A')}x{output_info.get('height', 'N/A')}")

        # Save output image if requested
        if save_output and 'image' in result:
            device_type = 'gpu' if 'GPU' in result.get('device', '') else 'tpu'
            input_name = Path(image_path).stem
            output_path = output_dir / f"{input_name}_{device_type}_upscaled.png"
            save_base64_image(result['image'], output_path)

        return result

    except requests.exceptions.Timeout:
        print(f"\n✗ Request timeout (60s)")
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Request failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Configuration
    base_url = "http://localhost:8000"
    gpu_endpoint = f"{base_url}/gpu/upscale"
    tpu_endpoint = f"{base_url}/tpu/upscale"

    # Output directory for upscaled images
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    # Allow custom image path from command line
    if len(sys.argv) > 1:
        test_image = Path(sys.argv[1])
        if not test_image.exists():
            print(f"Error: Image not found at {test_image}")
            sys.exit(1)
    else:
        # Find test image - look in common locations
        test_image_paths = [
            Path("./test_images/test.jpg"),
            Path("./real-esrgan/gpu/eva.jpg"),
            Path("/home/gech/gke-ray-demo/real-esrgan/gpu/eva.jpg"),
        ]

        test_image = None
        for path in test_image_paths:
            if path.exists():
                test_image = path
                break

        if test_image is None:
            print(f"Error: No test image found in default locations")
            print("Please provide an image path as argument:")
            print(f"  python {sys.argv[0]} /path/to/image.jpg")
            sys.exit(1)

    print(f"\n{'#'*60}")
    print(f"# Real-ESRGAN Inference Test")
    print(f"{'#'*60}")
    print(f"\nBase URL: {base_url}")
    print(f"Test Image: {test_image}")
    print(f"Output Directory: {output_dir}")

    # Test GPU endpoint
    print(f"\n\n{'*'*60}")
    print(f"* GPU INFERENCE TEST")
    print(f"{'*'*60}")
    gpu_result = test_endpoint(gpu_endpoint, test_image, output_dir)

    # Test TPU endpoint
    print(f"\n\n{'*'*60}")
    print(f"* TPU INFERENCE TEST")
    print(f"{'*'*60}")
    tpu_result = test_endpoint(tpu_endpoint, test_image, output_dir)

    print(f"\n\n{'='*60}")
    print(f"Testing complete!")
    print(f"Output images saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
