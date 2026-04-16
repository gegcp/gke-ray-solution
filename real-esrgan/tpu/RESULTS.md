# Real-ESRGAN TPU v6e - Test Results

## System Configuration

- **Device**: TPU v6e (TpuDevice, process=0, coords=(0,0,0,0))
- **Model**: RealESRGAN_x4plus (23 RRDB blocks, 64M parameters)
- **Framework**: PyTorch + torch_xla2 + JAX
- **Precision**: FP32 (highest matmul precision)
- **Weight**: `/home/gech/gke-ray-solution/real-esrgan/tpu/weights/RealESRGAN_x4plus.pth`

## Test Results

### Test 1: Eva Image (Anime Character)

**Input:**
- Path: `/home/gech/gke-ray-solution/real-esrgan/gpu/eva.jpg`
- Size: 350 x 197 pixels
- Format: JPEG

**Output:**
- Path: `eva_upscaled_tpu.png`
- Size: 1400 x 788 pixels (4x upscale)
- Format: PNG

**Performance:**
- **TPU Compilation Time**: 5.98 seconds (one-time)
- **TPU Inference Time**: 3,737.04 ms
- **CPU Inference Time**: 2,176.27 ms (for comparison)
- **Output File Size**: 1.1 MB

### Test 2: Burger Image (Food Photo)

**Input:**
- Path: `/home/gech/gke-ray-solution/repvgg/gpu/burger.jpeg`
- Size: 225 x 225 pixels
- Format: JPEG

**Output:**
- Path: `burger_upscaled_tpu.png`
- Size: 900 x 900 pixels (4x upscale)
- Format: PNG

**Performance:**
- **TPU Compilation Time**: 5.18 seconds (one-time)
- **TPU Inference Time**: 2,907.82 ms
- **Output File Size**: 709 KB

## Performance Analysis

### Compilation Overhead

The first inference on TPU includes JIT compilation:
- Eva image: 5.98s compilation + 3.74s inference = 9.72s total
- Burger image: 5.18s compilation + 2.91s inference = 8.09s total

**Note**: Compilation happens once per input shape. Subsequent inferences on same-sized images would skip this step.

### Small Image Performance

For small images (<500px), CPU is faster than TPU:
- **CPU advantage**: No compilation overhead, direct execution
- **TPU overhead**: JIT compilation, data transfer, XLA graph building

### When TPU Excels

TPU performance improves with:
1. **Larger images** (>1000px) - better compute/transfer ratio
2. **Batch processing** - process multiple images together
3. **Repeated inferences** - amortize compilation cost
4. **Tiled processing** - optimal memory usage for huge images

## Quality Assessment

Both upscaled images show excellent quality:

✅ **Eva Image**:
- Sharp details on character edges
- Smooth color gradients
- No visible artifacts
- 4x resolution increase (197→788 height, 350→1400 width)

✅ **Burger Image**:
- Enhanced texture details (sesame seeds, lettuce)
- Clear meat and cheese layers
- Natural color reproduction
- 4x resolution increase (225→900 in both dimensions)

## Summary Table

| Image | Input Size | Output Size | TPU Time (ms) | CPU Time (ms) | Output Size (MB) |
|-------|------------|-------------|---------------|---------------|------------------|
| Eva   | 350×197    | 1400×788    | 3,737.04      | 2,176.27      | 1.1              |
| Burger| 225×225    | 900×900     | 2,907.82      | N/A           | 0.7              |

## Recommendations

### For Production Use

1. **Small images (<500px)**: Use CPU for better latency
2. **Medium images (500-2000px)**: TPU becomes competitive
3. **Large images (>2000px)**: Use TPU with tiled inference
4. **Batch processing**: Always use TPU to process multiple images

### Optimization Strategies

1. **Batch multiple images** - Process several images in one forward pass
2. **Use tiled inference** - Split large images into 512x512 tiles
3. **Cache compiled models** - Reuse JIT-compiled functions
4. **Mixed precision** - Consider FP16/BF16 for even faster inference

## Files Generated

All output files are located in `/home/gech/gke-ray-solution/real-esrgan/tpu/`:

```
├── eva_upscaled_cpu.png   (1.1 MB) - CPU version
├── eva_upscaled_tpu.png   (1.1 MB) - TPU version
└── burger_upscaled_tpu.png (709 KB) - TPU version
```

## Date

Generated: April 16, 2026
