# AI Cooking Assistant

Local recipe assistant using RAG (Retrieval-Augmented Generation) with Llama 3.3 70B via Ollama.

## Current Status: Phase 3 Complete ✅

- ✅ Phase 1: Data ingestion (88,399 recipes indexed)
- ✅ Phase 2: Embeddings + vector store with GPU acceleration
- ✅ Phase 3: Cross-encoder reranking + recipe cards
- 🚧 Phase 4: LLM integration (next)

## Features

- Recommends real recipes from Food.com dataset (180K+ recipes, 88K indexed)
- GPU-accelerated semantic search with ChromaDB
- High-quality embeddings (all-mpnet-base-v2, 768-dim)
- Cross-encoder reranking for improved relevance (ms-marco-MiniLM-L-6-v2)
- Compact recipe cards for LLM context (Phase 3)
- Learns user preferences and dietary restrictions (Phase 5)
- Asks clarifying questions for better recommendations (Phase 4)
- Supports "ingredients on hand" queries (Phase 4)
- Fully local (no cloud dependencies)

## Tech Stack

- **LLM**: Llama 3.3 70B Instruct via Ollama
- **Vector Store**: ChromaDB with 88K recipes
- **Embeddings**: sentence-transformers (all-mpnet-base-v2)
- **GPU**: PyTorch 2.11 nightly with native RTX 5090 support
- **Reranker**: cross-encoder (ms-marco-MiniLM-L-6-v2, GPU-accelerated)
- **Framework**: LangChain (LCEL chains)
- **Database**: SQLite for recipes and user state
- **CLI**: Typer

## Requirements

- **Python**: 3.11, 3.12, or 3.13 (Python 3.14 not yet supported due to onnxruntime compatibility)
- **GPU**: NVIDIA GPU with CUDA 12.8+ (recommended for fast embedding generation)
- **Ollama**: For local LLM inference (Phase 4+)

## Setup

### 1. Clone and Install Dependencies

```bash
git clone <repo-url>
cd ai-cooking-assistant
pip install -e ".[dev,ml]"
```

### 2. GPU Setup (Recommended)

For fast embedding generation with NVIDIA GPUs:

```bash
# Install PyTorch with CUDA support (for RTX 5090 and other modern GPUs)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify GPU access
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

See `docs/GPU_SETUP.md` for detailed GPU setup instructions.

### 3. Data Ingestion

Download and process the Food.com dataset:

```bash
# Download dataset from Kaggle (requires Kaggle API credentials)
python -m src.app.cli ingest download

# Process recipes: normalize, filter, and save to SQLite
python -m src.app.cli ingest process

# Build embeddings and vector store (uses GPU if available)
python -m src.app.cli ingest embed

# View statistics
python -m src.app.cli ingest stats
```

**Time estimates**:
- Download: ~5 minutes
- Process: ~2 minutes
- Embed (GPU): ~2-3 minutes for 88K recipes
- Embed (CPU): ~10-15 minutes

### 4. Install Ollama (Phase 4+)

```bash
ollama serve
ollama pull llama3.3:70b
```

## Usage

### Search Recipes

```bash
# Basic vector search
python -m src.app.cli search "chicken tomato spicy"

# Search with cross-encoder reranking (improved relevance)
python -m src.app.cli search "chicken tomato spicy" --rerank

# Search with recipe cards (detailed display, implies --rerank)
python -m src.app.cli search "chicken tomato spicy" --cards

# Search with specific number of results
python -m src.app.cli search "quick pasta dinner" -k 5

# View configuration
python -m src.app.cli config
```

**New in Phase 3:**
- `--rerank` / `-r`: Enable cross-encoder reranking (100 candidates → 20 → display top k)
- `--cards` / `-c`: Display detailed recipe cards with summaries and match explanations

### Interactive Chat (Phase 4+)

```bash
python -m src.app.cli chat
```

## Development

### Run Tests

```bash
# Run all tests (82 tests total)
pytest

# Run retrieval tests only
pytest tests/test_retrieval*.py -v

# Run Phase 3 tests (reranking + recipe cards)
pytest tests/test_rerank.py tests/test_recipe_cards.py -v

# Run regression tests (verify no functionality broke)
pytest tests/test_regression.py -v

# Run with coverage
pytest --cov=src
```

**Test Suite** (Phase 3):
- 82 total tests (all passing)
- Unit tests: Reranking (6), Recipe cards (14), Ingestion (13), CLI (4)
- Integration tests: Retrieval (23), Reranking (4), Recipe cards (4)
- Regression tests: 14 tests verifying Phase 1-3 compatibility

### Available Commands

```bash
# Data ingestion
python -m src.app.cli ingest download    # Download Food.com dataset
python -m src.app.cli ingest process     # Process and filter recipes
python -m src.app.cli ingest embed       # Build vector store
python -m src.app.cli ingest stats       # Show dataset statistics
python -m src.app.cli ingest sample <id> # View a recipe by ID

# Search and testing
python -m src.app.cli search <query>     # Search for recipes
python -m src.app.cli config             # View settings
python -m src.app.cli version            # Show version

# Chat (Phase 4+)
python -m src.app.cli chat               # Interactive assistant
```

## Documentation

- `PROJECT_PLAN.md` - Detailed phased development plan
- `CLAUDE.md` - Architecture and development guidance
- `PHASE_2_TEST_RESULTS.md` - Phase 2 test results and benchmarks
- `PHASE_2_UPGRADE_SUMMARY.md` - GPU acceleration and quality improvements
- `docs/GPU_SETUP.md` - GPU setup instructions

## Project Structure

```
ai-cooking-assistant/
├── src/
│   ├── app/          # CLI and settings
│   ├── domain/       # Pydantic models
│   ├── ingest/       # Data ingestion pipeline
│   ├── retrieval/    # Vector search and retrieval
│   ├── llm/          # LLM client (Phase 4+)
│   ├── memory/       # User preferences (Phase 5+)
│   └── chains/       # LangChain LCEL (Phase 4+)
├── tests/            # Unit and integration tests
├── data/             # Data directory (not in git)
│   ├── raw/          # Downloaded datasets
│   ├── processed/    # Normalized recipes
│   ├── chroma/       # Vector store
│   └── sqlite/       # SQLite database
└── docs/             # Additional documentation
```
