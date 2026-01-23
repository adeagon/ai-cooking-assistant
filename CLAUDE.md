# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local Recipe Assistant: A fully-local, interactive dinner-planning assistant using RAG (Retrieval-Augmented Generation) with Qwen 3 14B via Ollama. Recommends real recipes from an indexed Food.com dataset, learns user preferences, asks clarifying questions, and supports "ingredients on hand" queries.

## Tech Stack

- **Python**: 3.11-3.13
- **LLM Runtime**: Ollama (local HTTP API) with Qwen 3 14B (hybrid reasoning mode)
- **LLM Strategy**: Hybrid approach - thinking mode for clarification, fast mode for recommendations
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
- ✅ **Enhanced Constraint Extraction**: Data-driven cuisine/goal recognition
  - **32 cuisines** loaded from recipe database (asian, korean, greek, etc.)
  - **Goal fallbacks**: light→low-calorie, cheap→inexpensive, hearty→comfort-food
  - **Profile preferences**: User's preferred cuisines boost retrieval relevance
  - **Taste classification**: LLM-based scripts for light/hearty/mild/rich tags
  - All 267 tests passing (12 new tests for enhanced extraction)
- ✅ **Comprehensive Recipe Classification**: Hybrid approach for metadata enrichment
  - **Ingredient-based rules**: Deterministic vegetarian/vegan tagging (plant-based broths only)
  - **LLM classification**: Taste (sweet/savory/spicy/mild/rich/light), occasion, cuisine
  - **30+ cuisines**: american, italian, mexican, chinese, indian, thai, greek, french, etc.
  - **Smart skip logic**: Only classifies recipes missing tags
  - **Quality controls**: Category limits in prompts, mutual exclusion rules, HIGH+MEDIUM confidence
- ✅ **ChromaDB Metadata Filtering**: Precise constraint filtering at database level
  - **Structured metadata**: `is_vegetarian`, `is_vegan`, `cuisine` fields in ChromaDB
  - **Database-level filtering**: Dietary/cuisine/time constraints applied via ChromaDB `where` clauses
  - **Guaranteed constraint satisfaction**: Vegan filter returns ONLY vegan recipes (no false positives)
  - **Fixed broth classification**: Animal-based broths (chicken, beef) now correctly excluded from vegetarian
  - **51K vegetarian, 15K vegan** recipes after corrected tagging
- ✅ **Chat Enhancements**: Critical bug fixes for chat reliability
  - **Negative constraints**: "no casseroles", "without cheese", "but not soups" now respected
  - **Intent classifier fixes**: Fixed "save to recipe box" saving wrong recipe, "show me some X" misclassification
  - **Recipe name matching**: Article stripping ("the", "a", "an") and word-subset matching for fuzzy lookups
  - **English-only responses**: Prompts now enforce English language output
  - **Empty response handling**: Fallback message if LLM returns empty response
  - **Context pollution fix**: Rolling summary no longer pollutes vector search queries
  - All 288 tests passing (14 new tests for chat enhancements)
- ✅ **Behavioral Steering**: Ollama Modelfiles for consistent LLM behavior
  - **Custom Modelfiles**: Baked-in behavioral guidelines reduce per-request prompt overhead
  - **cooking-assistant**: Main chat model with core behaviors (English-only, dietary respect, numbered recommendations)
  - **intent-classifier**: Command classification with low temperature for deterministic output
  - **Token savings**: ~770 tokens saved per conversation by moving rules to Modelfile
  - **Setup**: `ollama create cooking-assistant -f config/models/Modelfile.cooking-assistant`
- ✅ **Hybrid LLM Optimization**: Selective thinking mode for quality vs speed
  - **Three LLM instances**: Fast (recommendations), Thoughtful (clarification), Intent (classification)
  - **Clarification LLM**: `reasoning=True` with 2x token budget for crafting better questions
  - **Recommendation LLM**: `reasoning=False` for fast, reliable recipe presentations
  - **Intent LLM**: `reasoning=False` with low temperature for deterministic classification
  - **Quality improvement**: Thoughtful mode produces more specific recipe suggestions and diverse cuisine options
  - **Reliability fix**: Fast mode prevents empty responses from thinking consuming all tokens
  - **Comprehensive test suite**: 48 conversation tests covering all scenarios (100% pass rate)
- ✅ **Meal Planning Feature**: Plan meals for the week with ingredient optimization
  - **Deterministic algorithm**: Beam search (width 5) with stable tie-breaking for reproducible plans
  - **Ingredient overlap optimization**: Minimizes grocery shopping by sharing ingredients across meals
  - **Diversity constraints**: Max 2 same protein, max 2 same cuisine for variety
  - **Recipe Box integration**: Prioritizes saved recipes in plans
  - **Grocery list generation**: Aggregates ingredients, excludes pantry staples (salt, oil, etc.)
  - **Natural language entry**: "plan my meals for the week", "help me plan 5 dinners"
  - **Constraint extraction**: Days, dietary restrictions, time limits, exclusions with audit trail
  - **Category exclusions**: Exclude dairy, meat, seafood, nuts, gluten categories
- ✅ **Multi-User Support**: Complete user isolation with user-bound stores
  - **User identity tracking**: `/login <username>` command to switch users (Phase 1)
  - **Data isolation**: All stores filter by `username` column - users cannot see each other's data (Phase 2)
  - **User-bound store instances**: Stores bind to username at instantiation (Phase 3)
  - **StoreFactory pattern**: Centralized factory creates and caches user-scoped stores
  - **Immutable user binding**: `store.user` is read-only to prevent mid-session mutation
  - **BaseUserBoundStore**: Abstract base class for consistent store initialization
  - **UserStores dataclass**: Container for all 6 store types (profile, feedback, history, recipe_box, session, meal_plan)
  - **CLI flow enhancements** (Phase 4): Multi-user startup banner, redundant login detection, user context in all logs
  - All 639 tests passing (53 new multi-user tests including 4 for CLI flow)

## Pending Tasks

None - all planned features complete.

## Completed Classifications

- ✅ **LLM Classification Complete** (January 2026)
  - **84,024 recipes** classified with taste and cuisine tags
  - **Taste coverage**: 100% (88,362 recipes)
    - savory: 56,879 | rich: 33,965 | sweet: 30,824
    - light: 19,368 | mild: 9,247 | spicy: 8,747
  - **Cuisine coverage**: 93.6% (82,754 recipes)
    - Top: american (54,901), italian (5,664), southern-united-states (5,164), mexican (3,564)
  - Duration: 8.8 hours with 4 workers at 2.6 recipes/sec
  - Quality validated via spot checks - classifications are accurate

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
python -m src.app.cli chat --user alex  # Start as specific user

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

# Meal Planning commands (in chat mode):
# /mealplan       - Start meal planning (or "plan my meals for the week")
# /plan           - Show current meal plan (or "show my plan")
# /grocery        - Generate grocery list from meal plan (or "grocery list")

# Multi-User commands (in chat mode):
# /login <user>   - Switch to a different user (or "login as alex")
# /whoami         - Show current user

# Run tests
pytest

# Run Phase 5 integration tests
pytest tests/test_feedback.py tests/test_history.py tests/test_feedback_integration.py -v

# Run meal planning tests
pytest tests/test_meal_plan*.py tests/test_ingredient*.py tests/test_grocery*.py -v

# Run multi-user isolation tests
pytest tests/test_multi_user_isolation.py tests/test_store_factory.py tests/test_cli_login_flow.py tests/test_user_context.py -v

# Run LLM integration tests (requires Ollama running)
pytest tests/test_llm_chat_phase5.py -v -s -m llm

# Run retrieval tests
pytest tests/test_retrieval*.py -v

# Run with verbose output
pytest -v

# Recipe classification (optional, enhances search for taste/occasion/cuisine)
# Step 1: Apply deterministic ingredient rules for vegetarian/vegan (~2 min)
python scripts/apply_ingredient_rules.py
python scripts/apply_ingredient_rules.py --test  # Test on 50 samples first

# Step 2: Run LLM classification for taste, occasion, cuisine (~4-5 hours)
python scripts/classify_comprehensive_tags.py --workers 4
python scripts/classify_comprehensive_tags.py --test 100  # Test on samples first

# Validation scripts
python scripts/spot_check_classifications.py    # Validate classification accuracy
python scripts/benchmark_accuracy.py            # Test classification accuracy

# Conversation testing (requires Ollama running)
python scripts/conversation_test_session.py     # Run 48 comprehensive chat tests
python scripts/test_clarification_quality.py    # Test clarification responses
python scripts/compare_clarification_modes.py   # Compare reasoning modes

# Meal planning conversation tests
python scripts/test_meal_planning_conversation.py  # Test meal planning flow (8 tests)
```

## Architecture

### High-Level Components

1. **Ingestion Pipeline** (`src/ingest/`): Offline processing of Food.com dataset - loads, normalizes, embeds, and persists to vector store and SQLite.

2. **Retrieval System** (`src/retrieval/`): RAG pipeline with retrieve (k=100) → rerank (k=20) → context (k=6) architecture. GPU-accelerated vector search (Phase 2) and cross-encoder reranking (Phase 3) complete. **ChromaDB metadata filtering** for dietary (vegetarian/vegan), cuisine, and time constraints. RecipeCard builder creates compact representations for LLM prompts.

3. **LLM Layer** (`src/llm/`): Abstracted client interface (`LLMClient`) with Ollama implementation. Enables runtime swapping.

4. **Memory System** (`src/memory/`): Multi-layer memory and personalization with multi-user support:
   - **StoreFactory**: Creates and caches user-scoped store instances
   - **BaseUserBoundStore**: Abstract base class - all stores bind to username at init
   - **UserStores**: Dataclass container for all 6 store types per user
   - **ProfileStore**: Pinned preferences (persistent, structured in SQLite)
   - **SessionStore**: Session constraints (per dinner-planning session)
   - **RollingSummarizer**: Rolling summary (1-3 sentences updated each turn)
   - **FeedbackStore**: Recipe feedback (likes, dislikes, ratings) and cuisine learning
   - **HistoryStore**: Cooking history with date-based filtering
   - **RecipeBoxStore**: Saved/bookmarked recipes for later reference
   - **MealPlanStore**: Meal plan persistence with planned meals and grocery lists

5. **Planning System** (`src/planning/`): Deterministic meal planning:
   - **MealPlanner**: Beam search algorithm with ingredient overlap optimization
   - **IngredientNormalizer**: Token-based normalization with phrase preservation
   - **IngredientCategoryClassifier**: Best-effort category classification (dairy, meat, nuts, etc.)
   - **GroceryListGenerator**: Aggregates ingredients across meal plan
   - **ConstraintExtractor**: Extracts days, dietary, time, exclusions from natural language

6. **CLI App** (`src/app/`): Typer-based conversational interface.

7. **Utilities** (`src/utils/`): Shared utilities for the application:
   - **tag_loader**: Loads valid cuisines and goals from recipe database with LRU caching
   - Provides fallback mappings for user-friendly terms (light→low-calorie, etc.)

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
Constraint extraction from user query (dietary, cuisine, time, ingredients, avoid)
    |
    v
GPU-accelerated embedding generation
    |
    v
Vector retrieval with metadata filters (100 candidates)
    |  - ChromaDB where: is_vegetarian, is_vegan, cuisine, minutes
    |  - Guarantees constraint satisfaction at DB level
    |
    v
Exclusion filtering (remove liked/disliked/cooked)
    |
    v
Avoid filtering (remove recipes matching "no X" constraints)
    |
    v
Cross-encoder rerank (20 candidates)
    |
    v
Build RecipeCards (6 for LLM context)
    |
    v
[NEW] Hybrid LLM Branching
    |  - Clarification path: llm_clarification (reasoning=True, 2x tokens)
    |  - Recommendation path: llm (reasoning=False, fast mode)
    |
    v
LLM generates response (clarification or recommendations)
    |
    v
Validate response (fallback if empty)
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
- `Constraints`: Extracted constraints from user query (ingredients, time_limit, dietary, cuisine, goals, dish_name, avoid)
- `PreferenceProfile`: Persistent user preferences (spice level, diet, avoid ingredients, cuisines)
- `SessionState`: Current session constraints (ingredients on hand, time limit, goals)
- `RecipeFeedback`: User feedback on recipes (like, dislike, rate) with timestamps (Phase 5)
- `CookingHistoryEntry`: Record of cooked recipes with dates and optional notes (Phase 5)
- `IntentClassification`: LLM-based intent classification result (intent, confidence, recipe_reference, rating_value, reasoning)
- `SavedRecipe`: Bookmarked recipe in Recipe Box with title, saved date, and notes
- `MealPlan`: Complete meal plan with start/end dates, status, constraints, metrics
- `PlannedMeal`: Individual meal entry with day, meal_type (breakfast/lunch/dinner), recipe_id, source (box/discovery)
- `MealPlanConstraints`: Planning constraints including days, dietary, max_prep_time, exclusions, diversity limits
- `GroceryList`: Aggregated shopping list from meal plan with categorized items
- `GroceryItem`: Individual grocery item with normalized name, recipes using it, category

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

### Multi-User Store Architecture
All store classes are **user-bound** - they bind to a specific username at instantiation and cannot change users mid-session.

**Key conventions**:
1. Use `StoreFactory.get_stores(username)` to get all stores for a user
2. Store instances are cached - switching users reuses existing instances
3. `store.user` is read-only to prevent accidental user-switching bugs
4. All stores extend `BaseUserBoundStore` for consistent initialization
5. On user switch, get new stores from factory and reset session state

**API pattern**:
```python
# Create factory once
factory = StoreFactory(db_path)

# Get stores for a user (cached)
stores = factory.get_stores("alex")
stores.profile.save(profile)      # No user_id parameter needed
stores.feedback.add_feedback(fb)  # Automatically uses bound user

# Switch users
stores = factory.get_stores("caitlyn")  # Gets/creates caitlyn's stores
```

**Thread safety**: Each store opens its own SQLite connection. For high-concurrency scenarios, consider connection pooling.
