# AI Cooking Assistant

Local recipe assistant using RAG (Retrieval-Augmented Generation) with Qwen 3 14B via Ollama.

## Current Status: Phase 5+ Enhanced ✅

- ✅ Phase 1: Data ingestion (88,399 recipes indexed)
- ✅ Phase 2: Embeddings + vector store with GPU acceleration
- ✅ Phase 3: Cross-encoder reranking + recipe cards
- ✅ Phase 4: LLM integration with Ollama (LangChain LCEL)
- ✅ Phase 5: Memory & personalization with feedback system
- ✅ **Enhanced**: Data-driven constraint extraction + taste classification
- ✅ **Metadata Filtering**: ChromaDB-level dietary/cuisine/time filtering
- ✅ **Chat Enhancements**: Bug fixes for reliability (negative constraints, intent classification, empty response handling)
- ✅ **Hybrid LLM**: Selective thinking mode - thoughtful clarification, fast recommendations

## Features

- **Conversational Interface**: Natural language chat powered by Qwen 3 14B (Phase 4)
- **Smart Recommendations**: Recommends real recipes from Food.com dataset (88K+ indexed)
- **Recipe Box**: Save and bookmark recipes for later reference
- **Feedback System**: Like/dislike/rate recipes to improve recommendations (Phase 5)
- **Cooking History**: Track what you've cooked and when (Phase 5)
- **Smart Filtering**: Excludes liked/cooked/disliked recipes from future recommendations (Phase 5)
- **Full Recipe Display**: View complete recipes with ingredients and instructions (Phase 5)
- **GPU-Accelerated Search**: Semantic search with ChromaDB + high-quality embeddings (all-mpnet-base-v2)
- **Cross-Encoder Reranking**: Improved relevance with ms-marco-MiniLM-L-6-v2 (Phase 3)
- **Intelligent Clarification**: Asks questions when constraints are insufficient (Phase 4)
- **Constraint Extraction**: Data-driven NLP for ingredients, time, diet, cuisine, goals (Enhanced)
- **Negative Constraints**: "no casseroles", "without cheese", "but not soups" now respected
- **Precise Dietary Filtering**: Vegetarian/vegan constraints enforced at database level (no false positives)
- **32 Cuisines Supported**: Asian, Korean, Greek, Middle-Eastern, and more loaded from recipe data
- **Taste Tags**: Light, hearty, mild, rich classifications via LLM (parallel processing)
- **Profile Preferences in Search**: User's preferred cuisines boost retrieval relevance
- **Session Memory**: Rolling summaries and user preferences (Phase 4)
- **Recipe Cards**: Compact LLM-ready representations (120-250 tokens) (Phase 3)
- **Hybrid LLM Mode**: Thoughtful clarification questions (reasoning enabled), fast recommendations (reasoning disabled)
- **Fully Local**: No cloud dependencies - runs entirely on your machine

## Tech Stack

- **LLM**: Qwen 3 14B via Ollama (50% faster than Qwen 2.5, with reasoning mode)
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
ollama pull qwen3:14b
```

### 5. Create Custom Modelfiles (Recommended)

Custom Modelfiles bake behavioral guidelines into the model, improving consistency and reducing prompt overhead:

```bash
# Create the cooking-assistant model (main chat)
ollama create cooking-assistant -f config/models/Modelfile.cooking-assistant

# Create the intent-classifier model (command detection)
ollama create intent-classifier -f config/models/Modelfile.intent-classifier

# Verify they work
ollama run cooking-assistant "What can I make with chicken?"
```

The Modelfiles configure:
- **cooking-assistant**: Core behaviors (English-only, dietary respect, numbered recommendations)
- **intent-classifier**: Command classification (lower temperature for deterministic output)

**To use the custom models**, set environment variables or create a `.env` file:
```bash
# .env file
OLLAMA_MODEL=cooking-assistant
OLLAMA_INTENT_MODEL=intent-classifier
```

If you skip this step, the app will use `qwen3:14b` directly (works but with less optimized behavior).

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
- Natural language conversation powered by Qwen 3 14B
- **Natural language commands** - say "I loved that one" instead of `/like 1`
- Automatic constraint extraction (ingredients, time, diet, cuisine, goals)
- Intelligent clarification when constraints are vague
- Real-time recipe recommendations from 88K+ indexed recipes
- Session memory with rolling summaries

**Commands** (use slash commands OR natural language):

| Slash Command | Natural Language Examples |
|---------------|--------------------------|
| `/new` | "start over", "new session", "reset" |
| `/prefs` | "my preferences", "show settings" |
| `/like <ref>` | "I loved the first one", "thumbs up", "that was great" |
| `/dislike <ref>` | "didn't like it", "not for me", "thumbs down" |
| `/rate <1-5> <ref>` | "give it 4 stars", "rate it a 3", "5 out of 5" |
| `/show <ref>` | "show me the recipe", "what's in that", "full details" |
| `/cooked <ref>` | "I made that", "cooked it last night", "tried the pasta" |
| `/history` | "what have I cooked", "cooking history", "show history" |
| `/save <ref>` | "save that recipe", "bookmark it", "add to my box" |
| `/unsave <ref>` | "remove from saved", "unsave it" |
| `/box` | "my saved recipes", "show bookmarks", "recipe box" |
| `quit` / `exit` | "quit", "exit" |

**Examples:**
```
# Recipe discovery and feedback with natural language
You: I have chicken and tomatoes, something quick and healthy
Assistant: [Recommends 3 recipes]
You: I loved the first one
[Liked: Chicken Tacos]
You: save that
[Saved to Recipe Box: Chicken Tacos]
You: show me the full recipe
[Full recipe with ingredients and instructions displayed]

# Using slash commands (still supported)
You: quick pasta dinner
Assistant: [Recommends recipes]
You: /rate 4 2
[Rated Pasta Carbonara: 4/5]
You: /cooked 2
[Marked as cooked: Pasta Carbonara]

## Development

### Run Tests

```bash
# Run all tests (289 tests total)
pytest

# Run retrieval tests (including metadata filtering)
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

**Test Suite**:
- **289 total tests** (all passing)
- **Chat Enhancement Tests (14 new)**:
  - Negative constraint extraction ("no casseroles", "without cheese")
  - Article stripping for recipe name matching
  - Word-subset matching for fuzzy lookups
  - English-only prompt rules
  - Empty response validation
- **Metadata Filtering Tests (11)**:
  - Vegetarian/vegan filter verification
  - Cuisine filter (Italian, etc.)
  - Time constraint filtering
  - Combined filters (vegetarian + Italian + under 30 min)
  - Metadata schema validation
- **Enhanced Tests (12)**: Data-driven cuisine/goal extraction
- **Phase 5 Tests (48)**: FeedbackStore, HistoryStore, integration workflows
- **Phase 4 Tests (76)**: Memory system, chains, integration, chat scenarios
- **Hybrid LLM Tests (1 new)**: Dish name + cuisine clarification logic
- **Phase 1-3 Tests (82)**: All regression tests passing
- **Conversation Tests (48)**: Comprehensive chatbot scenarios (run via `scripts/conversation_test_session.py`)

See `PHASE_4_TEST_RESULTS.md` for Phase 4 test results and `PHASE_5_SUMMARY.md` for Phase 5 implementation details.

### Recipe Classification (Complete)

All 88,399 recipes have been classified with dietary, taste, and cuisine tags:

**Current Coverage:**
- **Dietary**: 51K vegetarian, 15K vegan (ingredient-based rules)
- **Taste**: 100% coverage (88,362 recipes)
  - savory: 56,879 | rich: 33,965 | sweet: 30,824 | light: 19,368 | mild: 9,247 | spicy: 8,747
- **Cuisine**: 93.6% coverage (82,754 recipes)
  - american: 54,901 | italian: 5,664 | southern-united-states: 5,164 | mexican: 3,564 | french: 2,739

**Classification Categories:**
- **Dietary** (ingredient rules): vegetarian, vegan (only plant-based broths allowed)
- **Taste** (LLM): sweet, savory, spicy, mild, rich, light
- **Occasion** (LLM): weeknight, comfort-food, kid-friendly, dinner-party, holiday-event, inexpensive, etc.
- **Cuisine** (LLM): american, italian, mexican, chinese, indian, thai, greek, french, etc. (30+ cuisines)

**To re-run classification** (if needed):
```bash
# Step 1: Apply ingredient-based rules for vegetarian/vegan tags (~2 min)
python scripts/apply_ingredient_rules.py

# Step 2: Run LLM classification for taste, occasion, and cuisine tags (~8-9 hours)
python scripts/classify_comprehensive_tags.py --workers 4

# Test on samples first (recommended)
python scripts/classify_comprehensive_tags.py --test 100
```

This enables search queries like "something light", "Italian food", or "quick weeknight meals".

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

- `CLAUDE.md` - Architecture and development guidance
- `PROJECT_NOTES.md` - Consolidated development history and phase summaries
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
│   ├── chains/       # LangChain LCEL (Phase 4+)
│   └── utils/        # Shared utilities (tag_loader, etc.)
├── config/
│   └── models/       # Ollama Modelfiles for behavioral steering
│       ├── Modelfile.cooking-assistant   # Main chat model
│       └── Modelfile.intent-classifier   # Intent classification model
├── scripts/          # Classification and utility scripts
│   ├── apply_ingredient_rules.py        # Deterministic vegetarian/vegan tagging
│   ├── classify_comprehensive_tags.py   # LLM taste/occasion/cuisine classification
│   ├── classify_taste_tags_parallel.py  # Legacy taste classification (deprecated)
│   ├── spot_check_classifications.py    # Validation script
│   ├── benchmark_accuracy.py            # Classification accuracy testing
│   ├── benchmark_comprehensive.py       # Model comparison benchmarks
│   ├── conversation_test_session.py     # Comprehensive chatbot tests (48 scenarios)
│   ├── test_clarification_quality.py    # Clarification response quality tests
│   └── compare_clarification_modes.py   # Compare reasoning=True vs False
├── tests/            # Unit and integration tests
├── data/             # Data directory (not in git)
│   ├── raw/          # Downloaded datasets
│   ├── processed/    # Normalized recipes
│   ├── chroma/       # Vector store
│   └── sqlite/       # SQLite database
└── docs/             # Additional documentation
```
