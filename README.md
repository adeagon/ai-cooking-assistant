# AI Cooking Assistant

Local recipe assistant using RAG (Retrieval-Augmented Generation) with Qwen 2.5 14B via Ollama.

## Current Status: Phase 5 Complete ✅

- ✅ Phase 1: Data ingestion (88,399 recipes indexed)
- ✅ Phase 2: Embeddings + vector store with GPU acceleration
- ✅ Phase 3: Cross-encoder reranking + recipe cards
- ✅ Phase 4: LLM integration with Ollama (LangChain LCEL)
- ✅ Phase 5: Memory & personalization with feedback system

## Features

- **Conversational Interface**: Natural language chat powered by Qwen 2.5 14B (Phase 4)
- **Smart Recommendations**: Recommends real recipes from Food.com dataset (88K+ indexed)
- **Recipe Box**: Save and bookmark recipes for later reference
- **Feedback System**: Like/dislike/rate recipes to improve recommendations (Phase 5)
- **Cooking History**: Track what you've cooked and when (Phase 5)
- **Smart Filtering**: Excludes liked/cooked/disliked recipes from future recommendations (Phase 5)
- **Full Recipe Display**: View complete recipes with ingredients and instructions (Phase 5)
- **GPU-Accelerated Search**: Semantic search with ChromaDB + high-quality embeddings (all-mpnet-base-v2)
- **Cross-Encoder Reranking**: Improved relevance with ms-marco-MiniLM-L-6-v2 (Phase 3)
- **Intelligent Clarification**: Asks questions when constraints are insufficient (Phase 4)
- **Constraint Extraction**: Rule-based NLP for ingredients, time, diet, cuisine, goals (Phase 4)
- **Session Memory**: Rolling summaries and user preferences (Phase 4)
- **Recipe Cards**: Compact LLM-ready representations (120-250 tokens) (Phase 3)
- **Fully Local**: No cloud dependencies - runs entirely on your machine

## Tech Stack

- **LLM**: Qwen 2.5 14B via Ollama
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
ollama pull qwen2.5:14b
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

### Interactive Chat (Phase 4)

Start a conversational session with the recipe assistant:

```bash
python -m src.app.cli chat
```

**Features:**
- Natural language conversation powered by Qwen 2.5 14B
- **Natural language commands** - say "I loved that one" instead of `/like 1`
- Automatic constraint extraction (ingredients, time, diet, cuisine, goals)
- Intelligent clarification when constraints are vague
- Real-time recipe recommendations from 88K+ indexed recipes
- Session memory with rolling summaries

**Commands:**
- `/new` - Start a new session
- `/prefs` - Show your preferences
- `/like <ref>` - Like a recipe (by number or name) (Phase 5)
- `/dislike <ref>` - Dislike a recipe (Phase 5)
- `/rate <1-5> <ref>` - Rate a recipe (Phase 5)
- `/show <ref>` - Show full recipe with ingredients and instructions (Phase 5)
- `/cooked <ref>` - Mark recipe as cooked (Phase 5)
- `/history` - Show cooking history (Phase 5)
- `/save <ref>` - Save recipe to Recipe Box
- `/unsave <ref>` - Remove recipe from Recipe Box
- `/box` - List all saved recipes
- `quit` or `exit` - End the chat

**Example:**
```
You: I have chicken and tomatoes, something quick and healthy
Assistant: [Recommends 3 recipes]
You: /like 1
You: /show 1
[Full recipe with ingredients and instructions displayed]
You: /cooked 1

## Development

### Run Tests

```bash
# Run all tests (206 tests total: 82 Phase 1-3, 76 Phase 4, 48 Phase 5)
pytest

# Run retrieval tests only
pytest tests/test_retrieval*.py -v

# Run Phase 3 tests (reranking + recipe cards)
pytest tests/test_rerank.py tests/test_recipe_cards.py -v

# Run Phase 4 tests (memory + chains + integration + scenarios)
pytest tests/test_memory.py tests/test_chains.py tests/test_chat_integration.py tests/test_chat_scenarios.py -v

# Run Phase 5 tests (feedback + cooking history + integration)
pytest tests/test_feedback.py tests/test_history.py tests/test_feedback_integration.py -v

# Run LLM integration tests (requires Ollama running)
pytest tests/test_llm_chat_phase5.py -v -s -m llm

# Run chat scenario tests only
pytest tests/test_chat_scenarios.py -v

# Run regression tests (verify no functionality broke)
pytest tests/test_regression.py -v

# Run with coverage
pytest --cov=src
```

**Test Suite** (Phase 5):
- 206 total tests (all passing)
- **Phase 5 New Tests (48)**:
  - FeedbackStore (12): Like/dislike/rate, cuisine learning
  - HistoryStore (10): Cooking tracking, date filtering
  - Integration (23): Recipe reference resolver, exclusion filtering, full workflows
  - **LLM Integration (3)**: Full conversation tests with Ollama (requires `ollama serve`)
- **Total: 82 (Phase 1-3) + 76 (Phase 4) + 48 (Phase 5) = 206 tests**
- **Phase 4 Tests (76)**:
  - Memory system (20): ProfileStore, SessionStore, RollingSummarizer
  - Chains (27): ConstraintExtractor, prompt formatters, gate logic
  - Integration (7): LCEL chain integration, end-to-end flow
  - Chat scenarios (22): Real-world conversation flows, constraint extraction, clarification gates
- **Phase 1-3 Tests (82)**: All regression tests passing

See `PHASE_4_TEST_RESULTS.md` for Phase 4 test results and `PHASE_5_SUMMARY.md` for Phase 5 implementation details.

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
- `PHASE_2_UPGRADE_SUMMARY.md` - GPU acceleration and quality improvements (Phase 2)
- `PHASE_4_SUMMARY.md` - LLM integration with LangChain LCEL (Phase 4)
- `PHASE_4_TEST_RESULTS.md` - Comprehensive Phase 4 test results (10 scenarios, GPU metrics)
- `PHASE_5_SUMMARY.md` - Memory & personalization with feedback system (Phase 5)
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
