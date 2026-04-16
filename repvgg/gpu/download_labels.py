#!/usr/bin/env python3
"""Download ImageNet class labels"""

import json
import urllib.request

url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

print("Downloading ImageNet labels...")
try:
    with urllib.request.urlopen(url) as response:
        labels_list = json.loads(response.read().decode())

    # Convert list to dict with index as key
    labels_dict = {str(i): label for i, label in enumerate(labels_list)}

    with open('imagenet_classes.json', 'w') as f:
        json.dump(labels_dict, f, indent=2)

    print(f"✓ Downloaded {len(labels_dict)} ImageNet class labels")
    print("Saved to: imagenet_classes.json")
except Exception as e:
    print(f"Error downloading labels: {e}")
    print("You can run inference without labels (will show class indices)")
