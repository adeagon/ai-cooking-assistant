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
- ✅ **Meal Planning**: Plan meals for the week with ingredient overlap optimization
- ✅ **Multi-User Support**: Complete user isolation with `/login`, `/logout`, `/whoami` commands
- ✅ **Web App**: Local web interface accessible on home network (FastAPI + modern chat UI)

## Features

- **Web Interface**: Modern chat UI accessible from any device on your home network
- **Conversational Interface**: Natural language chat powered by Qwen 3 14B (Phase 4)
- **Smart Recommendations**: Recommends real recipes from Food.com dataset (88K+ indexed)
- **Multi-User Support**: Multiple users with complete data isolation (`/login`, `/logout`, `/whoami`)
- **Meal Planning**: Plan up to a week of meals with ingredient overlap optimization
- **Grocery List**: Auto-generated shopping list from your meal plan
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
- **Web**: FastAPI with SSE streaming, Jinja2 templates
- **Database**: SQLite for recipes, user state, and web sessions
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

### Web App (Recommended)

The web app provides a modern chat interface accessible from any device on your network:

```bash
# Install web dependencies
pip install -e ".[web,ml]"

# Start the web server
uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://localhost:8000` in your browser (or use your computer's IP address from other devices).

**Web App Features:**
- Modern chat interface with SSE streaming
- Conversation persistence (survives server restarts)
- Multi-user support with login/logout
- Recipe cards with ratings, time, and ingredients
- Conversation history sidebar
- Mobile-responsive design
- Works on any device on your home network

**Default Users:** alex, jordan, taylor, casey

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

# Start as a specific user
python -m src.app.cli chat --user alex
```

**Features:**
- Natural language conversation powered by Qwen 3 14B
- **Natural language commands** - say "I loved that one" instead of `/like 1`
- **Multi-user support** - each user has isolated preferences, history, and saved recipes
- Automatic constraint extraction (ingredients, time, diet, cuisine, goals)
- Intelligent clarification when constraints are vague
- Real-time recipe recommendations from 88K+ indexed recipes
- Session memory with rolling summaries

**Available Users:** guest (default), alex, caitlyn, family, test

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
| `/mealplan` | "plan my meals", "help me plan dinners", "meal plan" |
| `/plan` | "show my plan", "view meal plan", "current plan" |
| `/grocery` | "grocery list", "shopping list", "what do I need to buy" |
| `/login <user>` | "login as alex", "switch to caitlyn" |
| `/logout` | "logout", "sign out" |
| `/whoami` | "who am I", "current user" |
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

# Meal planning
You: plan 5 vegetarian dinners for the week
[Generates meal plan with ingredient overlap optimization]
You: show my plan
[Displays the meal plan]
You: grocery list
[Generates aggregated shopping list]

# Multi-user support
You: /login alex
[Logged in as: alex]
You: /whoami
[Logged in as: alex]
You: /logout
[Logged out. Now logged in as: guest]
```

## Development

### Run Tests

```bash
# Run all tests (727+ tests total)
pytest

# Run web app tests (79 tests)
pytest tests/web/ -v

# Run retrieval tests (including metadata filtering)
pytest tests/test_retrieval*.py -v

# Run Phase 3 tests (reranking + recipe cards)
pytest tests/test_rerank.py tests/test_recipe_cards.py -v

# Run Phase 4 tests (memory + chains + integration + scenarios)
pytest tests/test_memory.py tests/test_chains.py tests/test_chat_integration.py tests/test_chat_scenarios.py -v

# Run Phase 5 tests (feedback + cooking history + integration)
pytest tests/test_feedback.py tests/test_history.py tests/test_feedback_integration.py -v

# Run meal planning tests
pytest tests/test_meal_plan*.py tests/test_ingredient*.py tests/test_grocery*.py -v

# Run multi-user isolation tests (95 tests)
pytest tests/test_multi_user_isolation.py tests/test_store_factory.py tests/test_cli_login_flow.py tests/test_user_context.py -v

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
- **727+ total tests** (all passing)
- **Web App Tests (79)**: Services (user, session, conversation), API endpoints (auth, conversations), Playwright E2E structure
- **Multi-User Tests (95 unit + 34 conversation)**:
  - User identity tracking (`/login`, `/logout`, `/whoami`)
  - Data isolation across all 6 stores (profile, feedback, history, recipe_box, session, meal_plan)
  - StoreFactory caching and user switching
  - BaseUserBoundStore inheritance verification
  - Rapid user switching without data leakage
  - End-to-end conversation tests with Ollama
- **Chat Enhancement Tests (14)**:
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
- **Hybrid LLM Tests (1)**: Dish name + cuisine clarification logic
- **Meal Planning Tests (240+)**:
  - Ingredient normalizer (phrase preservation, stop tokens)
  - Ingredient categories classifier (dairy, meat, seafood, nuts, gluten)
  - Meal plan store (CRUD, constraints persistence)
  - Meal planner algorithm (beam search, determinism, diversity)
  - Grocery list generator (aggregation, pantry exclusion)
  - Constraint extractor (days, dietary, time, exclusions)
  - Integration tests (full flow, audit trail, Recipe Box integration)
- **Phase 1-3 Tests (82)**: All regression tests passing
- **Conversation Tests (48)**: Comprehensive chatbot scenarios (run via `scripts/conversation_test_session.py`)
- **Meal Planning Conversation Tests (8)**: Full flow tests (run via `scripts/test_meal_planning_conversation.py`)
- **Multi-User Conversation Tests (34)**: User isolation tests (run via `scripts/test_multi_user_conversation.py`)

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
│   ├── app/          # CLI, settings, and user context
│   │   ├── cli.py           # Typer CLI with multi-user support
│   │   ├── settings.py      # Pydantic settings
│   │   └── user_context.py  # UserContext and UserRegistry
│   ├── web/          # Web application (FastAPI)
│   │   ├── app.py           # FastAPI application factory
│   │   ├── config.py        # Web-specific settings
│   │   ├── db.py            # SQLite schema and connection
│   │   ├── dependencies.py  # FastAPI dependencies
│   │   ├── models.py        # Pydantic API models
│   │   ├── routers/         # API endpoints (auth, chat, conversations)
│   │   ├── services/        # Business logic (user, session, chat)
│   │   ├── static/          # CSS and JavaScript
│   │   └── templates/       # Jinja2 HTML templates
│   ├── domain/       # Pydantic models
│   ├── ingest/       # Data ingestion pipeline
│   ├── retrieval/    # Vector search and retrieval
│   ├── llm/          # LLM client (Phase 4+)
│   ├── memory/       # User preferences with multi-user support
│   │   ├── base_store.py       # BaseUserBoundStore abstract class
│   │   ├── store_factory.py    # StoreFactory for user-scoped stores
│   │   ├── profile_store.py    # User preferences
│   │   ├── feedback_store.py   # Likes/dislikes/ratings
│   │   ├── history_store.py    # Cooking history
│   │   ├── recipe_box_store.py # Saved recipes
│   │   ├── session_store.py    # Session state
│   │   └── meal_plan_store.py  # Meal plans
│   ├── chains/       # LangChain LCEL (Phase 4+)
│   ├── planning/     # Meal planning system
│   └── utils/        # Shared utilities (tag_loader, etc.)
├── config/
│   └── models/       # Ollama Modelfiles for behavioral steering
│       ├── Modelfile.cooking-assistant   # Main chat model
│       └── Modelfile.intent-classifier   # Intent classification model
├── scripts/          # Classification and utility scripts
│   ├── apply_ingredient_rules.py        # Deterministic vegetarian/vegan tagging
│   ├── classify_comprehensive_tags.py   # LLM taste/occasion/cuisine classification
│   ├── spot_check_classifications.py    # Validation script
│   ├── benchmark_accuracy.py            # Classification accuracy testing
│   ├── conversation_test_session.py     # Comprehensive chatbot tests (48 scenarios)
│   ├── test_clarification_quality.py    # Clarification response quality tests
│   ├── compare_clarification_modes.py   # Compare reasoning=True vs False
│   ├── test_meal_planning_conversation.py  # Meal planning flow tests (8 scenarios)
│   └── test_multi_user_conversation.py  # Multi-user isolation tests (34 scenarios)
├── tests/            # Unit and integration tests (648 tests)
│   ├── test_multi_user_isolation.py  # Multi-user data isolation tests
│   ├── test_store_factory.py         # StoreFactory tests
│   ├── test_cli_login_flow.py        # Login/logout flow tests
│   ├── test_user_context.py          # UserContext/UserRegistry tests
│   └── ...                           # Other test files
├── data/             # Data directory (not in git)
│   ├── raw/          # Downloaded datasets
│   ├── processed/    # Normalized recipes
│   ├── chroma/       # Vector store
│   └── sqlite/       # SQLite database
└── docs/             # Additional documentation
```
