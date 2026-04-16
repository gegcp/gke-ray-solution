#!/usr/bin/env python3
"""Fixed CPU test - load in train mode then convert to deploy"""

import json
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from repvgg import create_RepVGG_B0


# Load class names
with open('imagenet_classes.json', 'r') as f:
    class_names = json.load(f)

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

image = Image.open('/home/gech/gke-ray-solution/repvgg/gpu/burger.jpeg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# Load model in TRAIN mode first
print("Loading model in training mode...")
model = create_RepVGG_B0(deploy=False)  # ← IMPORTANT: deploy=False
checkpoint = torch.load('weights/RepVGG-B0-train.pth', map_location='cpu', weights_only=False)

if 'model' in checkpoint:
    state_dict = checkpoint['model']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Load state dict
result = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {len(result.missing_keys)}")
print(f"Unexpected keys: {len(result.unexpected_keys)}")

# Convert to deploy mode
print("\nConverting to deploy mode (fusing branches)...")
for module in model.modules():
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()

model.eval()

# Run inference
print("Running inference...")
with torch.no_grad():
    output = model(image_tensor)
    probabilities = F.softmax(output[0], dim=0)

# Get top 10 predictions
top_k = 10
top_probs, top_indices = torch.topk(probabilities, top_k)

print("\nTop 10 predictions:")
for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
    class_name = class_names.get(str(int(idx)), f"Unknown (class {idx})")
    print(f"{i}. {class_name:40s} {prob*100:6.2f}%")
