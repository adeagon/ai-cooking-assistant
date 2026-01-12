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
- ✅ **Phase 3 Complete**: Cross-encoder reranking + recipe cards
  - ms-marco-MiniLM-L-6-v2 reranker (GPU-accelerated)
  - RecipeCard builder with template-based summaries
  - Compact cards (120-250 tokens) with why_match explanations
  - All 82 tests passing (42 new tests for Phase 3, including 14 regression tests)
- ✅ **Phase 4 Complete**: LLM integration with Ollama
  - LangChain LCEL chains for chat orchestration
  - Memory system: ProfileStore, SessionStore, RollingSummarizer
  - Rule-based constraint extraction (ingredients, time, diet, cuisine, goals)
  - Conversational chat command with clarification vs recommendation branching
  - All 158 tests passing (76 new tests for Phase 4)
- ✅ **Phase 5 Complete**: Memory & personalization with feedback system
  - Feedback commands: `/like`, `/dislike`, `/rate` for user preferences
  - Cooking history: `/cooked` to track recipes, `/history` to view
  - Smart filtering: Excludes liked/disliked/recently cooked from recommendations
  - Full recipe display: `/show` command with ingredients and instructions
  - FeedbackStore and HistoryStore for persistent user data
  - **Natural language intents**: LLM-based classification for conversational commands
  - All 218 tests passing (48 Phase 5 + 12 Recipe Box + 15 intent classification tests)

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
- **IMPORTANT: Review test coverage at the end of each phase**:
  - Run full test suite and verify all tests pass
  - Review existing test files for potential improvements or additions
  - Add regression tests to verify new functionality doesn't break existing features
  - Add integration tests for end-to-end workflows
  - Update test documentation in `README.md` with new test counts and examples

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

# Phase 5 CLI commands (in chat mode) - use slash commands OR natural language:
# /like <ref>     - Like a recipe (or "I loved that one")
# /dislike <ref>  - Dislike a recipe (or "didn't like it")
# /rate <1-5> <ref> - Rate a recipe (or "give it 4 stars")
# /show <ref>     - Show full recipe (or "show me the recipe")
# /cooked <ref>   - Mark recipe as cooked (or "I made that")
# /history        - View cooking history (or "what have I cooked")
# /save <ref>     - Save recipe to Recipe Box (or "save that recipe")
# /unsave <ref>   - Remove recipe from Recipe Box (or "unsave it")
# /box            - List all saved recipes (or "my saved recipes")

# Run tests
pytest

# Run Phase 5 integration tests
pytest tests/test_feedback.py tests/test_history.py tests/test_feedback_integration.py -v

# Run LLM integration tests (requires Ollama running)
pytest tests/test_llm_chat_phase5.py -v -s -m llm

# Run retrieval tests
pytest tests/test_retrieval*.py -v

# Run with verbose output
pytest -v
```

## Architecture

### High-Level Components

1. **Ingestion Pipeline** (`src/ingest/`): Offline processing of Food.com dataset - loads, normalizes, embeds, and persists to vector store and SQLite.

2. **Retrieval System** (`src/retrieval/`): RAG pipeline with retrieve (k=100) → rerank (k=20) → context (k=6) architecture. GPU-accelerated vector search (Phase 2) and cross-encoder reranking (Phase 3) complete. RecipeCard builder creates compact representations for LLM prompts.

3. **LLM Layer** (`src/llm/`): Abstracted client interface (`LLMClient`) with Ollama implementation. Enables runtime swapping.

4. **Memory System** (`src/memory/`): Multi-layer memory and personalization:
   - **ProfileStore**: Pinned preferences (persistent, structured in SQLite)
   - **SessionStore**: Session constraints (per dinner-planning session)
   - **RollingSummarizer**: Rolling summary (1-3 sentences updated each turn)
   - **FeedbackStore**: Recipe feedback (likes, dislikes, ratings) and cuisine learning (Phase 5)
   - **HistoryStore**: Cooking history with date-based filtering (Phase 5)
   - **RecipeBoxStore**: Saved/bookmarked recipes for later reference (no exclusion from recommendations)

5. **CLI App** (`src/app/`): Typer-based conversational interface.

### Data Flow

**Current (Phase 5 with Natural Language Intents)**:
```
User Input
    |
    v
Check for quit/exit
    |
    v
[NEW] Natural Language Intent Classification (if not starting with /)
    |  - LLM-based intent detection (like, save, show, rate, etc.)
    |  - Quick pattern cache for stateless commands
    |  - Conservative classification (defaults to conversation)
    |
    +-- intent detected? → Execute command → Continue
    |
    v
Slash Command Detection (/like, /save, /show, etc.) [Fallback]
    |
    v (if not a command)
Compute exclusion set (liked + disliked + recently cooked recipes)
    |
    v
Constraint extraction from user query
    |
    v
GPU-accelerated embedding generation
    |
    v
Vector retrieval (100 candidates)
    |
    v
Exclusion filtering (remove liked/disliked/cooked)
    |
    v
Cross-encoder rerank (20 candidates)
    |
    v
Build RecipeCards (6 for LLM context)
    |
    v
LLM generates response (clarification or recommendations)
    |
    v
Display response + capture recipe cards for commands
```

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
- `RecipeFeedback`: User feedback on recipes (like, dislike, rate) with timestamps (Phase 5)
- `CookingHistoryEntry`: Record of cooked recipes with dates and optional notes (Phase 5)
- `IntentClassification`: LLM-based intent classification result (intent, confidence, recipe_reference, rating_value, reasoning)
- `SavedRecipe`: Bookmarked recipe in Recipe Box with title, saved date, and notes

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
