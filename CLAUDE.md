# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local Recipe Assistant: A fully-local, interactive dinner-planning assistant using RAG (Retrieval-Augmented Generation) with Llama 3.3 70B via Ollama. Recommends real recipes from an indexed Food.com dataset, learns user preferences, asks clarifying questions, and supports "ingredients on hand" queries.

## Tech Stack

- **Python**: 3.11-3.13
- **LLM Runtime**: Ollama (local HTTP API) with Llama 3.3 70B Instruct
- **Vector Store**: ChromaDB with 88K recipes
- **Embeddings**: sentence-transformers (`all-mpnet-base-v2`, 768-dim)
- **GPU**: PyTorch 2.11 nightly with native RTX 5090 support (CUDA 12.8)
- **Reranker**: cross-encoder (`ms-marco-MiniLM-L-6-v2` or `BAAI/bge-reranker-base`)
- **Framework**: LangChain (LCEL chains)
- **Database**: SQLite for recipes and user state
- **CLI**: typer
- **Config**: pydantic-settings
- **Testing**: pytest

## Current Status

- ✅ **Phase 1 Complete**: Data ingestion (88,399 recipes indexed)
- ✅ **Phase 2 Complete**: Embeddings + vector store with GPU acceleration
  - all-mpnet-base-v2 embeddings (768-dim)
  - PyTorch 2.11 nightly with native RTX 5090 support
  - GPU-accelerated embedding generation (3.8x faster)
  - Increased retrieval parameters (k=100, k_rerank=20, k_context=6)
  - All 27 tests passing
- 🚧 **Phase 3 Next**: Reranking + recipe cards

## Development Workflow

- **Make sure to create a feature branch for each Phase** (e.g., `phase-2-embeddings`, `phase-3-reranking`)
- Merge to main/master only after phase completion and testing
- Keep commits atomic and descriptive
- **IMPORTANT: Before committing at the end of each phase, ensure all documentation is up-to-date**:
  - Update `README.md` with new features, setup steps, and usage examples
  - Update `CLAUDE.md` with architecture changes, tech stack updates, and current status
  - Update `PROJECT_PLAN.md` if scope or timeline changes
  - Create phase summary documents (e.g., `PHASE_2_UPGRADE_SUMMARY.md`)
  - Verify all code examples and commands are accurate

## Build Commands

```bash
# Install dependencies with ML packages
pip install -e ".[ml]"

# Install PyTorch with GPU support (recommended)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Data ingestion
python -m src.app.cli ingest download  # Download dataset
python -m src.app.cli ingest process   # Process and filter
python -m src.app.cli ingest embed     # Build vector store (uses GPU)
python -m src.app.cli ingest stats     # Show statistics

# Search recipes
python -m src.app.cli search "chicken tomato spicy"

# Run CLI chat (Phase 4+)
python -m src.app.cli chat

# Run tests
pytest

# Run retrieval tests
pytest tests/test_retrieval*.py -v

# Run with verbose output
pytest -v
```

## Architecture

### High-Level Components

1. **Ingestion Pipeline** (`src/ingest/`): Offline processing of Food.com dataset - loads, normalizes, embeds, and persists to vector store and SQLite.

2. **Retrieval System** (`src/retrieval/`): RAG pipeline with retrieve (k=100) → rerank (k=20) → context (k=6) architecture. Phase 2 complete with GPU-accelerated vector search. Phase 3 will add cross-encoder reranking (deterministic code, not LLM).

3. **LLM Layer** (`src/llm/`): Abstracted client interface (`LLMClient`) with Ollama implementation. Enables runtime swapping.

4. **Memory System** (`src/memory/`): Three-layer memory model:
   - Pinned preferences (persistent, structured in SQLite)
   - Session constraints (per dinner-planning session)
   - Rolling summary (1-3 sentences updated each turn)

5. **CLI App** (`src/app/`): Typer-based conversational interface.

### Data Flow

**Current (Phase 2)**:
User query → GPU-accelerated embedding → Vector retrieval (100 candidates) → Return top results

**Future (Phase 3+)**:
User query → Constraint extraction → Vector retrieval (100 candidates) → Cross-encoder rerank (20) → Build RecipeCards (6) → LLM generates response

### Key Design Constraints

- **Tight context**: RecipeCards are compact (120-250 tokens each) - never dump full recipes into LLM prompt during recommendation
- **Memory efficiency**: Rolling summary + pinned prefs, not full chat history
- **Context window**: Target 8192 tokens
- **Clarify vs Recommend**: Gate function checks if constraints are sufficient before retrieval

## Data Stores

- `data/raw/`: Downloaded datasets (not in git)
- `data/processed/`: Normalized recipes (jsonl/parquet)
- `data/chroma/`: Persistent vector DB
- `data/sqlite/`: SQLite database (app.db)

## Core Domain Models (Pydantic)

- `Recipe`: Canonical recipe with id, title, ingredients, instructions, tags, ratings
- `RecipeCard`: Compact prompt representation with title, tags, key_ingredients, one_sentence_summary, why_match
- `PreferenceProfile`: Persistent user preferences (spice level, diet, avoid ingredients, cuisines)
- `SessionState`: Current session constraints (ingredients on hand, time limit, goals)

## LangChain Chains (LCEL)

1. `constraint_extractor_chain`: Parse user input for constraints
2. `retrieval_chain`: Vector search with metadata filters
3. `rerank_chain`: Cross-encoder scoring (deterministic)
4. `response_chain`: LLM generates clarifying questions OR recipe recommendations

## Debugging

Log at each turn:
- Parsed constraints
- Retrieval query string
- Top candidates with scores
- Reranked list
- Final cards passed to LLM

## Development Best Practices

### Using Context7 MCP
- Always use Context7 MCP when you need library/API documentation
- Especially useful for PyTorch, ChromaDB, sentence-transformers, LangChain
- Example queries:
  - "Latest PyTorch installation with CUDA support"
  - "ChromaDB persistent client setup"
  - "sentence-transformers model selection"

### GPU Acceleration
- Verify GPU access: `python -c "import torch; print(torch.cuda.is_available())"`
- sentence-transformers automatically uses GPU when available
- For RTX 5090 (Blackwell), use PyTorch nightly with CUDA 12.8+
- See `docs/GPU_SETUP.md` for detailed setup
