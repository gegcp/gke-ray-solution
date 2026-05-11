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

"""
Model Composition: Real-ESRGAN (TPU) → RepVGG (GPU)
Chains super-resolution and image classification
"""

import os
import io
import base64
from typing import Dict
from PIL import Image

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request

from ray import serve
from ray.serve.handle import DeploymentHandle


app = FastAPI()


@serve.deployment(
    name="CompositionDeployment",
    ray_actor_options={"num_cpus": 1},
    num_replicas=1,
)
@serve.ingress(app)
class CompositionDeployment:
    """
    Model composition deployment that chains Real-ESRGAN and RepVGG.

    Pipeline:
    1. User input (image) → Real-ESRGAN (TPU) for upscaling
    2. Upscaled image → RepVGG (GPU) for classification
    """

    def __init__(self, esrgan_handle: DeploymentHandle, repvgg_handle: DeploymentHandle):
        """
        Initialize composition with handles to Real-ESRGAN and RepVGG deployments.

        Args:
            esrgan_handle: Handle to Real-ESRGAN TPU deployment
            repvgg_handle: Handle to RepVGG GPU deployment
        """
        self.esrgan = esrgan_handle
        self.repvgg = repvgg_handle
        print("✓ Model composition initialized: Real-ESRGAN (TPU) → RepVGG (GPU)")

    @app.post("/classify_upscaled")
    async def classify_upscaled(self, request: Request) -> JSONResponse:
        """
        Upscale image with Real-ESRGAN (TPU) then classify with RepVGG (GPU).

        Request body:
        - image: base64 encoded image string
        - top_k: (optional) number of top predictions to return (default: 5)
        - img_size: (optional) RepVGG input size (default: 224)
        - return_upscaled_image: (optional) include upscaled image in response (default: false)

        Returns:
        - original_size: Input image dimensions
        - upscaled_size: Real-ESRGAN output dimensions
        - esrgan_device: Device used for upscaling
        - predictions: RepVGG classification results
        - repvgg_device: Device used for classification
        - upscaled_image: (optional) Base64 encoded upscaled image
        """
        request_dict = await request.json()

        # Get parameters
        image_data = request_dict.get("image")
        if not image_data:
            return JSONResponse(
                status_code=400,
                content={"error": "No image provided"}
            )

        top_k = request_dict.get("top_k", 5)
        img_size = request_dict.get("img_size", 224)
        return_upscaled_image = request_dict.get("return_upscaled_image", False)

        try:
            # Decode input image to get original size
            if image_data.startswith('data:image'):
                image_data_clean = image_data.split(',')[1]
            else:
                image_data_clean = image_data

            image_bytes = base64.b64decode(image_data_clean)
            original_image = Image.open(io.BytesIO(image_bytes))
            original_size = original_image.size

            print(f"Pipeline input: {original_size[0]}x{original_size[1]}")

            # Step 1: Upscale with Real-ESRGAN (TPU)
            print("Step 1: Upscaling with Real-ESRGAN (TPU)...")
            esrgan_response = await self.esrgan.upscale.remote({
                "image": image_data,
                "return_image": True
            })

            # Extract upscaled image and metadata
            upscaled_image_data = esrgan_response["image"]
            upscaled_size = esrgan_response["output_size"]
            esrgan_device = esrgan_response["device"]
            esrgan_scale = esrgan_response["scale"]

            print(f"Step 1 complete: {upscaled_size['width']}x{upscaled_size['height']} on {esrgan_device}")

            # Step 2: Classify with RepVGG (GPU)
            print("Step 2: Classifying with RepVGG (GPU)...")
            repvgg_response = await self.repvgg.classify.remote({
                "image": upscaled_image_data,
                "top_k": top_k,
                "img_size": img_size
            })

            predictions = repvgg_response["predictions"]
            repvgg_device = repvgg_response["device"]
            repvgg_model = repvgg_response["model"]

            print(f"Step 2 complete: {len(predictions)} predictions on {repvgg_device}")
            print(f"Top prediction: class_id={predictions[0]['class_id']}, prob={predictions[0]['probability']:.4f}")

            # Prepare response
            response_data = {
                "pipeline": "Real-ESRGAN (TPU) → RepVGG (GPU)",
                "original_size": {
                    "width": original_size[0],
                    "height": original_size[1]
                },
                "upscaling": {
                    "model": "Real-ESRGAN-TPU",
                    "device": esrgan_device,
                    "scale": esrgan_scale,
                    "output_size": upscaled_size
                },
                "classification": {
                    "model": repvgg_model,
                    "device": repvgg_device,
                    "predictions": predictions
                }
            }

            if return_upscaled_image:
                response_data["upscaled_image"] = upscaled_image_data

            return JSONResponse(content=response_data)

        except Exception as e:
            import traceback
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"Pipeline failed: {str(e)}",
                    "traceback": traceback.format_exc()
                }
            )

    @app.get("/health")
    async def health(self) -> Dict[str, str]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "pipeline": "Real-ESRGAN (TPU) → RepVGG (GPU)",
            "description": "Image upscaling followed by classification"
        }


# Build the application graph with model composition
def build_app():
    """Build the model composition application"""
    # Import backend deployment classes (no FastAPI ingress)
    from realesrgan_tpu_backend import RealESRGANBackend
    from repvgg_gpu_backend import RepVGGBackend

    # Create backend deployment instances
    esrgan = RealESRGANBackend.bind()
    repvgg = RepVGGBackend.bind()

    # Create composition deployment with handles
    composition = CompositionDeployment.bind(esrgan, repvgg)

    return composition


model = build_app()
