# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local Recipe Assistant: A fully-local, interactive dinner-planning assistant using RAG (Retrieval-Augmented Generation) with Llama 3.3 70B via Ollama. Recommends real recipes from an indexed Food.com dataset, learns user preferences, asks clarifying questions, and supports "ingredients on hand" queries.

## Tech Stack

- **Python**: 3.11+
- **LLM Runtime**: Ollama (local HTTP API) with Llama 3.3 70B Instruct
- **Vector Store**: ChromaDB (or FAISS)
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2` baseline)
- **Reranker**: cross-encoder (`ms-marco-MiniLM-L-6-v2` or `BAAI/bge-reranker-base`)
- **Framework**: LangChain (LCEL chains)
- **Database**: SQLite for user state
- **CLI**: typer
- **Config**: pydantic-settings
- **Testing**: pytest

## Development Workflow

- **Make sure to create a feature branch for each Phase** (e.g., `phase-2-embeddings`, `phase-3-reranking`)
- Merge to master only after phase completion and testing
- Keep commits atomic and descriptive

## Build Commands

```bash
# Install dependencies (once pyproject.toml exists)
pip install -e .

# Run CLI chat
python -m app.cli chat

# Run tests
pytest

# Run single test file
pytest tests/test_retrieval.py

# Run with verbose output
pytest -v
```

## Architecture

### High-Level Components

1. **Ingestion Pipeline** (`src/ingest/`): Offline processing of Food.com dataset - loads, normalizes, embeds, and persists to vector store and SQLite.

2. **Retrieval System** (`src/retrieval/`): RAG pipeline with retrieve (k=30) → rerank (k=10) → context (k=4) architecture. Uses cross-encoder reranking (deterministic code, not LLM).

3. **LLM Layer** (`src/llm/`): Abstracted client interface (`LLMClient`) with Ollama implementation. Enables runtime swapping.

4. **Memory System** (`src/memory/`): Three-layer memory model:
   - Pinned preferences (persistent, structured in SQLite)
   - Session constraints (per dinner-planning session)
   - Rolling summary (1-3 sentences updated each turn)

5. **CLI App** (`src/app/`): Typer-based conversational interface.

### Data Flow

User query → Constraint extraction → Vector retrieval (30 candidates) → Cross-encoder rerank (10) → Build RecipeCards (4) → LLM generates response

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
