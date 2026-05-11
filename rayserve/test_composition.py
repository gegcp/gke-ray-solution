#!/usr/bin/env python3
"""
Test script for Model Composition: Real-ESRGAN (TPU) → RepVGG (GPU)
"""

import base64
import json
import requests
import sys
from pathlib import Path
from PIL import Image
import io


def load_imagenet_labels():
    """Load ImageNet class labels - returns a list indexed by class ID"""
    # Try to fetch from URL first
    try:
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            labels = response.json()
            if isinstance(labels, list) and len(labels) == 1000:
                return labels
    except:
        pass

    # Fallback: return None, will use class IDs only
    return None


def get_class_name(class_id, labels=None):
    """Get class name for a given class ID"""
    if labels and isinstance(labels, list) and 0 <= class_id < len(labels):
        return labels[class_id]
    else:
        # Fallback to a minimal set of common ImageNet classes
        common_labels = {
            0: "tench", 1: "goldfish", 2: "great white shark", 207: "golden retriever",
            208: "Labrador retriever", 281: "tabby cat", 282: "tiger cat", 283: "Persian cat",
            285: "Egyptian cat", 291: "lion", 292: "tiger", 293: "jaguar", 294: "leopard",
            352: "gibbon", 353: "orangutan", 354: "gorilla", 355: "chimpanzee",
            386: "African elephant", 460: "container ship", 491: "chain saw",
            497: "church", 504: "coffee mug", 509: "computer keyboard", 510: "computer mouse",
            530: "dishwasher", 609: "laptop computer", 627: "milk can", 701: "parachute",
            717: "pickup truck", 779: "school bus", 816: "sports car", 817: "steam locomotive",
            829: "streetcar", 848: "tank", 864: "totem pole", 867: "trailer truck",
            952: "lemon", 954: "banana", 963: "meat loaf"
        }
        return common_labels.get(class_id, f"class_{class_id}")


# Load ImageNet labels at module level
IMAGENET_LABELS = load_imagenet_labels()


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
    print(f"  Saved upscaled image to: {output_path}")
    print(f"  Size: {image.size[0]}x{image.size[1]}")


def test_composition(endpoint_url, image_path, output_dir, save_upscaled=True):
    """Test the model composition pipeline"""
    print(f"\n{'='*70}")
    print(f"Model Composition Test: Real-ESRGAN (TPU) → RepVGG (GPU)")
    print(f"{'='*70}")
    print(f"Endpoint: {endpoint_url}")
    print(f"Image: {image_path}")
    print(f"{'='*70}")

    # Check health endpoint first
    health_url = endpoint_url.rsplit('/', 1)[0] + '/health'
    try:
        health_response = requests.get(health_url, timeout=5)
        print(f"\n✓ Health check: {json.dumps(health_response.json(), indent=2)}")
    except Exception as e:
        print(f"\n✗ Health check failed: {e}")
        return

    # Get input image size
    input_image = Image.open(image_path)
    input_size = input_image.size
    print(f"\nInput image size: {input_size[0]}x{input_size[1]}")

    # Encode image
    print(f"\nEncoding image...")
    image_base64 = encode_image_to_base64(image_path)

    # Prepare request
    payload = {
        "image": image_base64,
        "top_k": 5,
        "img_size": 224,
        "return_upscaled_image": save_upscaled
    }

    # Send inference request
    print(f"\nSending request through pipeline...")
    print(f"  Step 1: Real-ESRGAN upscaling (TPU)")
    print(f"  Step 2: RepVGG classification (GPU)")
    try:
        response = requests.post(
            endpoint_url,
            json=payload,
            timeout=120  # Longer timeout for two-stage pipeline
        )
        response.raise_for_status()

        result = response.json()

        print(f"\n{'='*70}")
        print(f"✓ Pipeline completed successfully!")
        print(f"{'='*70}")

        # Display pipeline info
        print(f"\nPipeline: {result.get('pipeline', 'N/A')}")

        # Display upscaling results
        upscaling_info = result.get('upscaling', {})
        print(f"\n--- Step 1: Upscaling ---")
        print(f"Model: {upscaling_info.get('model', 'N/A')}")
        print(f"Device: {upscaling_info.get('device', 'N/A')}")
        print(f"Scale: {upscaling_info.get('scale', 'N/A')}x")

        original_size = result.get('original_size', {})
        upscaled_size = upscaling_info.get('output_size', {})
        print(f"Input:  {original_size.get('width', 'N/A')}x{original_size.get('height', 'N/A')}")
        print(f"Output: {upscaled_size.get('width', 'N/A')}x{upscaled_size.get('height', 'N/A')}")

        # Display classification results
        classification_info = result.get('classification', {})
        predictions = classification_info.get('predictions', [])

        print(f"\n--- Step 2: Classification ---")
        print(f"Model: {classification_info.get('model', 'N/A')}")
        print(f"Device: {classification_info.get('device', 'N/A')}")
        print(f"\nTop {len(predictions)} predictions:")

        for i, pred in enumerate(predictions, 1):
            class_id = pred.get('class_id', 'N/A')
            prob = pred.get('probability', 0)

            # Get class label
            class_name = get_class_name(class_id, IMAGENET_LABELS)

            print(f"  {i}. {class_name:<30} | ID: {class_id:4d} | Prob: {prob:.4f} ({prob*100:.2f}%)")

        # Save upscaled image if requested and available
        if save_upscaled and 'upscaled_image' in result:
            input_name = Path(image_path).stem
            output_path = output_dir / f"{input_name}_composition_upscaled.png"
            save_base64_image(result['upscaled_image'], output_path)

        print(f"\n{'='*70}")
        return result

    except requests.exceptions.Timeout:
        print(f"\n✗ Request timeout (120s)")
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Request failed: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Configuration
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/classify_upscaled"

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

    print(f"\n{'#'*70}")
    print(f"# Model Composition Test")
    print(f"# Real-ESRGAN (TPU) → RepVGG (GPU)")
    print(f"{'#'*70}")
    print(f"\nBase URL: {base_url}")
    print(f"Test Image: {test_image}")
    print(f"Output Directory: {output_dir}")

    # Test the composition
    result = test_composition(endpoint, test_image, output_dir, save_upscaled=True)

    if result:
        print(f"\n{'='*70}")
        print(f"Test complete!")
        print(f"Output images saved to: {output_dir}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
