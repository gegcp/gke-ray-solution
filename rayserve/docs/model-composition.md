# Model Composition: Real-ESRGAN (TPU) → RepVGG (GPU)

This demonstrates Ray Serve model composition by chaining two ML models on different accelerators.

## Architecture

```
User Input Image
    ↓
Real-ESRGAN (TPU)  ← Super-resolution upscaling (4x)
    ↓
RepVGG (GPU)       ← Image classification
    ↓
Classification Results
```

## Components

### 1. **RealESRGANBackend** (TPU)
- **File**: `realesrgan_tpu_backend.py`
- **Device**: TPU v6e with torch-xla2 (JAX/XLA)
- **Purpose**: 4x image super-resolution
- **Resources**: 1 CPU, 1 TPU
- **No FastAPI ingress** (backend only)

### 2. **RepVGGBackend** (GPU)
- **File**: `repvgg_gpu_backend.py`
- **Device**: NVIDIA L4 GPU
- **Purpose**: ImageNet classification
- **Resources**: 2 CPUs, 1 GPU
- **No FastAPI ingress** (backend only)

### 3. **CompositionDeployment** (Head Node)
- **File**: `composition_esrgan_repvgg_app.py`
- **Device**: CPU
- **Purpose**: Orchestrates the pipeline
- **Resources**: 1 CPU
- **Has FastAPI ingress** (single entry point)

## Key Design Patterns

### Backend Deployments Without Ingress
The backend deployments (`RealESRGANBackend` and `RepVGGBackend`) do not use `@serve.ingress(app)`. They only have a `@serve.deployment` decorator and expose methods that can be called via deployment handles.

```python
@serve.deployment(
    name="RealESRGANBackend",
    ray_actor_options={"num_cpus": 1, "resources": {"TPU": 1}},
    num_replicas=1,
)
class RealESRGANBackend:
    def upscale(self, request: Dict) -> Dict:
        # Backend method called by composition
        ...
```

### Composition with DeploymentHandle
The composition deployment receives handles to the backend deployments and calls them sequentially:

```python
@serve.deployment(...)
@serve.ingress(app)  # Only this has FastAPI ingress
class CompositionDeployment:
    def __init__(self, esrgan_handle: DeploymentHandle, repvgg_handle: DeploymentHandle):
        self.esrgan = esrgan_handle
        self.repvgg = repvgg_handle

    @app.post("/classify_upscaled")
    async def classify_upscaled(self, request: Request):
        # Step 1: Call Real-ESRGAN backend
        esrgan_response = await self.esrgan.upscale.remote({
            "image": image_data,
            "return_image": True
        })

        # Step 2: Call RepVGG backend with upscaled image
        repvgg_response = await self.repvgg.classify.remote({
            "image": upscaled_image_data,
            "top_k": top_k
        })
        ...
```

### Application Graph
The application graph binds the deployments together:

```python
def build_app():
    esrgan = RealESRGANBackend.bind()
    repvgg = RepVGGBackend.bind()
    composition = CompositionDeployment.bind(esrgan, repvgg)
    return composition

model = build_app()
```

## Deployment

### Files
- `composition-esrgan-repvgg.rayservice.yaml` - RayService configuration
- `deploy-composition.sh` - Deployment script
- `test_composition.py` - Test script

### Deploy
```bash
cd /home/gech/gke-ray-demo/rayserve
bash deploy-composition.sh
```

### Test
```bash
# Port-forward
kubectl port-forward <head-pod> 8000:8000

# Run test
python test_composition.py /path/to/image.jpg
```

## API Endpoint

### POST `/classify_upscaled`

**Request:**
```json
{
  "image": "base64_encoded_image",
  "top_k": 5,
  "img_size": 224,
  "return_upscaled_image": false
}
```

**Response:**
```json
{
  "pipeline": "Real-ESRGAN (TPU) → RepVGG (GPU)",
  "original_size": {"width": 350, "height": 197},
  "upscaling": {
    "model": "Real-ESRGAN-TPU",
    "device": "TPU (JAX/XLA)",
    "scale": 4,
    "output_size": {"width": 1400, "height": 788}
  },
  "classification": {
    "model": "RepVGG",
    "device": "cuda",
    "predictions": [
      {"class_id": 952, "probability": 0.001},
      ...
    ]
  },
  "upscaled_image": "base64_encoded_image"  // if requested
}
```

## Test Results

```
✓ Pipeline completed successfully!

Pipeline: Real-ESRGAN (TPU) → RepVGG (GPU)

--- Step 1: Upscaling ---
Model: Real-ESRGAN-TPU
Device: TPU (JAX/XLA)
Scale: 4x
Input:  350x197
Output: 1400x788

--- Step 2: Classification ---
Model: RepVGG
Device: cuda
Top 5 predictions: [...]
```

## Resource Allocation

| Component | CPU | GPU | TPU | Memory |
|-----------|-----|-----|-----|--------|
| Head Node | 2 | 0 | 0 | 8Gi |
| GPU Worker | 2 | 1 | 0 | 10Gi |
| TPU Worker | 1 | 0 | 1 | 20Gi |
| **Total** | **5** | **1** | **1** | **38Gi** |

## Important Notes

1. **Single FastAPI Ingress**: Only the composition deployment has `@serve.ingress(app)`. Backend deployments are called via deployment handles, not HTTP.

2. **Head Node CPUs**: The head node must have CPUs available (`num-cpus: "2"`) for the composition deployment to run.

3. **Async Communication**: Backend calls use `await handle.method.remote()` for async communication.

4. **Error Handling**: Each backend returns error dictionaries if processing fails, which the composition can handle gracefully.

5. **Model Weights**: Both models use random initialization in this demo. For production, load pre-trained weights via `WEIGHT_PATH` environment variable.

## References

- [Ray Serve Model Composition](https://docs.ray.io/en/latest/serve/model_composition.html)
- [Ray Serve DeploymentHandle](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.handle.DeploymentHandle.html)
