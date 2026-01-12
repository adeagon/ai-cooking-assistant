# GPU Acceleration Setup (RTX 5090)

## Current Status
- **Hardware**: RTX 5090 (32GB VRAM) ✅
- **PyTorch**: CPU-only version ❌
- **Performance**: ~19s per 500 recipes on CPU

## To Enable GPU Acceleration

### 1. Install CUDA Toolkit
Download and install CUDA 12.x from NVIDIA:
https://developer.nvidia.com/cuda-downloads

### 2. Install PyTorch with CUDA Support
Replace CPU-only PyTorch with CUDA version:

```bash
# Uninstall CPU version
pip uninstall torch torchvision torchaudio

# Install CUDA version (for CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Verify GPU Access
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Should output:
```
CUDA: True
GPU: NVIDIA GeForce RTX 5090
```

### 4. Rebuild Vector Store
Once GPU is enabled, sentence-transformers will automatically use it:

```bash
python -m src.app.cli ingest embed
```

## Expected Performance Improvements

### Embedding Generation
- **CPU (current)**: ~19s per 500 recipes (88K recipes in ~56 minutes)
- **RTX 5090 (expected)**: ~0.5-2s per 500 recipes (88K recipes in ~2-7 minutes)
- **Speedup**: **10-40x faster**

### Search/Query Performance
- **CPU (current)**: 4-10ms per query ✅ (already very fast)
- **GPU**: 1-3ms per query (marginal improvement)
- **Note**: Search is already fast enough on CPU; GPU mainly helps with batch embedding

## Benefits of GPU Acceleration

1. **Faster Reindexing**: Rebuild entire vector store in ~5 minutes vs ~1 hour
2. **Experiment Faster**: Try different embedding models quickly
3. **Larger Models**: Can use bge-large-en-v1.5 (1.34GB) efficiently
4. **Simultaneous Workloads**: Run embeddings + Llama 3.3 70B at same time

## Memory Usage

With 32GB VRAM on RTX 5090:
- Llama 3.3 70B quantized: ~20-25GB
- all-mpnet-base-v2 embeddings: ~1-2GB
- bge-large-en-v1.5 embeddings: ~2-3GB

**Plenty of room to run both LLM and embeddings simultaneously!**

## Trade-offs

**Pros**:
- 10-40x faster embedding generation
- Can experiment with larger, better models
- Future-proof for scaling

**Cons**:
- Additional setup (CUDA toolkit, PyTorch reinstall)
- ~5-10GB disk space for CUDA
- Only benefits batch operations (not real-time search)

## Recommendation

**For now**: Let CPU rebuild finish (search is already fast)
**For future**: Install GPU support before Phase 3+ for faster experimentation

## One-Time Setup Script

```bash
# Download CUDA from NVIDIA website first, then:

# Reinstall PyTorch with CUDA
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✓ GPU Ready: {torch.cuda.get_device_name(0)}')"

# Test with embeddings
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-mpnet-base-v2'); print(f'Device: {m.device}')"
```

Expected output:
```
✓ GPU Ready: NVIDIA GeForce RTX 5090
Device: cuda
```
