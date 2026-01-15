# Project Notes - AI Cooking Assistant

This document consolidates development history, decisions, and summaries from the phased implementation of the AI Cooking Assistant.

---

## Project Overview

**Goal**: Build a fully-local, interactive dinner-planning assistant that recommends real recipes from an indexed dataset, learns preferences from feedback, asks clarifying questions, and supports "ingredients on hand" queries.

**Environment**:
- Windows 11
- RTX 5090 FE (32GB VRAM)
- Ollama with Qwen 3 14B (upgraded from Llama 3.3 70B)
- Food.com dataset (88,399 recipes)

---

## Architecture Summary

### High-Level Components

1. **Ingestion Pipeline** (`src/ingest/`): Offline processing - load, normalize, embed, persist
2. **Retrieval System** (`src/retrieval/`): RAG pipeline (k=100 → rerank k=20 → context k=6)
3. **Memory System** (`src/memory/`): ProfileStore, SessionStore, RollingSummarizer, FeedbackStore, HistoryStore
4. **LLM Layer** (`src/llm/`): Abstracted client with Ollama implementation
5. **CLI App** (`src/app/`): Typer-based conversational interface
6. **LCEL Chains** (`src/chains/`): LangChain orchestration

### Data Flow

```
User Input → Intent Classification → Command or Conversation
    │
    ├─ Command → Execute (like, save, show, etc.)
    │
    └─ Conversation → Constraint Extraction → should_clarify()?
                          │
                          ├─ TRUE  → Clarification Prompt
                          │
                          └─ FALSE → Vector Retrieval (k=100)
                                       → Exclusion Filtering
                                       → Avoid Filtering
                                       → Cross-Encoder Rerank (k=20)
                                       → Recipe Cards (k=6)
                                       → LLM Recommendation
                                       → Response Validation
                                       → Display
```

### Key Design Decisions

- **Tight context**: RecipeCards compact (120-250 tokens) - never dump full recipes
- **Memory efficiency**: Rolling summary + pinned prefs, not full chat history
- **Context window**: Target 8192 tokens
- **Deterministic reranking**: Cross-encoder code, not LLM
- **Rule-based extraction**: Fast, deterministic constraint parsing

---

## Phase Summaries

### Phase 1: Data Ingestion ✅

**Completed**: Initial project setup and Food.com dataset processing

- Downloaded and processed 88,399 recipes
- Normalized ingredients (lowercase, stripped units)
- Created SQLite storage for recipes
- Basic CLI skeleton with Typer

**Exit Criteria**: Can print a recipe by ID; dataset stats logged.

---

### Phase 2: Embeddings + Vector Store ✅

**Key Upgrades**:
- **Embedding Model**: `all-MiniLM-L6-v2` (384-dim) → `all-mpnet-base-v2` (768-dim)
- **GPU Acceleration**: PyTorch 2.11 nightly with native RTX 5090 support
- **Retrieval Parameters**: k=30 → k=100, k_rerank=10 → k=20, k_context=4 → k=6

**Performance Results**:
| Metric | Before (CPU) | After (GPU) |
|--------|-------------|-------------|
| Embedding Time (88K) | 578.8s | 151.4s |
| Recipes/second | ~153 | ~584 |
| Improvement | - | **3.8x faster** |

**Search Quality**: Scores improved from 0.85 to 0.89+ on test queries.

**Trade-offs**:
- ✅ Better quality, faster embedding
- ⚠️ Larger model (420MB vs 90MB)
- ⚠️ PyTorch nightly dependency

---

### Phase 3: Cross-Encoder Reranking + Recipe Cards ✅

**New Components**:
- `RecipeReranker`: ms-marco-MiniLM-L-6-v2 (GPU-accelerated, ~37ms for 100 candidates)
- `RecipeCardBuilder`: Template-based summaries, why_match explanations

**CLI Enhancements**:
- `--rerank` / `-r`: Enable cross-encoder reranking
- `--cards` / `-c`: Display detailed recipe cards

**Architecture**: retrieve (k=100) → rerank (k=20) → cards (k=6)

**Performance**: Total pipeline ~342ms (well under 1s target)

**Tests**: 42 new tests (10 reranker, 18 cards, 14 regression)

---

### Phase 4: LLM Integration ✅

**New Components**:

1. **Memory System**:
   - ProfileStore: Persistent preferences in SQLite
   - SessionStore: Per-session dinner planning state
   - RollingSummarizer: Template-based context (1-3 key points)

2. **LCEL Chains**:
   - ConstraintExtractor: Rule-based NLP (ingredients, time, diet, cuisine, goals)
   - RetrievalRunnable: Wraps RAG pipeline
   - Chat Prompts: Clarification and Recommendation templates
   - Main Chain: RunnableBranch for conditional logic

3. **CLI Chat**: `python -m src.app.cli chat`

**Gate Logic**: `should_clarify()` checks for actionable constraints before retrieval.

**Test Results**: 10/10 comprehensive scenarios passed
- Vague requests → clarification
- Specific constraints → recommendations
- Multi-turn memory → refined results
- Dietary/cuisine → correct filtering

**Performance**:
| Component | Time |
|-----------|------|
| Constraint Extraction | <10ms |
| Vector Retrieval | ~500ms |
| Reranking | ~200ms |
| LLM Generation | 15-60s |

---

### Phase 5: Memory & Personalization ✅

**New Features**:

1. **Feedback System**:
   - `/like`, `/dislike`, `/rate` commands
   - Cuisine learning from likes
   - Persistent SQLite storage

2. **Cooking History**:
   - `/cooked` to record, `/history` to view
   - Smart exclusion (7-day window)

3. **Smart Filtering**:
   - Liked recipes: Last 20 excluded
   - Disliked recipes: Permanently excluded
   - Recently cooked: 7-day exclusion

4. **Full Recipe Display**: `/show` with ingredients and instructions

5. **Recipe Box**: `/save`, `/unsave`, `/box` for bookmarking

**New Tables**:
- `recipe_feedback`: Likes, dislikes, ratings
- `cooking_history`: Cooked dates and notes

**Tests**: 45 new tests (12 feedback, 10 history, 23 integration)

---

### Phase 5+: Enhancements ✅

**Data-Driven Constraint Extraction**:
- 32 cuisines loaded from recipe database
- Goal fallbacks: light→low-calorie, cheap→inexpensive, hearty→comfort-food
- Profile preferences boost retrieval relevance

**Recipe Classification** (Hybrid Approach):
- Ingredient-based rules: Deterministic vegetarian/vegan tagging
- LLM classification: Taste (sweet/savory/spicy/mild/rich/light), occasion, cuisine
- 30+ cuisines supported

**ChromaDB Metadata Filtering**:
- Structured metadata: `is_vegetarian`, `is_vegan`, `cuisine` in ChromaDB
- Database-level filtering via `where` clauses
- Guaranteed constraint satisfaction (no false positives)
- 51K vegetarian, 15K vegan recipes after corrected tagging

**Natural Language Intents**:
- LLM-based intent classification for conversational commands
- Quick pattern cache for stateless commands
- Conservative classification (defaults to conversation)

**Chat Enhancements**:
- Negative constraints: "no casseroles", "without cheese" now respected
- Intent classifier fixes: Save commands, "show me some X" disambiguation
- Recipe name matching: Article stripping, word-subset matching
- English-only responses enforced
- Empty response handling with fallback
- Context pollution fix: Rolling summary removed from retrieval query

**Behavioral Steering** (Latest):
- Custom Ollama Modelfiles bake behavioral guidelines into model configuration
- `cooking-assistant`: Main chat model with core behaviors (English-only, dietary respect, numbered recommendations)
- `intent-classifier`: Command classification with low temperature (0.2) for deterministic output
- ~770 tokens saved per conversation by moving rules from prompts to Modelfile
- Setup: `ollama create cooking-assistant -f config/models/Modelfile.cooking-assistant`

---

## Test Summary

**Total Tests**: 288 (all passing)

| Category | Count |
|----------|-------|
| Phase 1-3 (Ingestion, Retrieval, Cards) | 82 |
| Phase 4 (Memory, Chains, Integration) | 76 |
| Phase 5 (Feedback, History) | 48 |
| Enhanced Extraction | 12 |
| Metadata Filtering | 11 |
| Chat Enhancements | 14 |
| Regression | 14 |
| Other | 31 |

---

## Key Commands

```bash
# Data ingestion
python -m src.app.cli ingest download
python -m src.app.cli ingest process
python -m src.app.cli ingest embed
python -m src.app.cli ingest stats

# Search
python -m src.app.cli search "chicken tomato spicy"
python -m src.app.cli search "quick pasta" --rerank --cards

# Chat
python -m src.app.cli chat

# In-chat commands
/like <ref>     /dislike <ref>    /rate <1-5> <ref>
/show <ref>     /cooked <ref>     /history
/save <ref>     /unsave <ref>     /box
/new            /prefs            quit

# Tests
pytest                           # All tests
pytest tests/test_retrieval*.py  # Retrieval tests
pytest -m llm                    # LLM integration tests
```

---

## Configuration

**Settings** (`src/app/settings.py`):
```python
# Retrieval
k_retrieve = 100
k_rerank = 20
k_context = 6

# LLM
ollama_model = "qwen3:14b"  # or "cooking-assistant" with Modelfile
ollama_intent_model = "qwen3:14b"  # or "intent-classifier" with Modelfile
llm_temperature = 0.3
llm_max_tokens = 1024
ollama_timeout = 300.0

# Embeddings
embedding_model = "all-mpnet-base-v2"
```

**Modelfile Setup** (recommended for improved behavior):
```bash
ollama create cooking-assistant -f config/models/Modelfile.cooking-assistant
ollama create intent-classifier -f config/models/Modelfile.intent-classifier
```

Then set in `.env`:
```
OLLAMA_MODEL=cooking-assistant
OLLAMA_INTENT_MODEL=intent-classifier
```

---

## Future Considerations

1. **Hybrid BM25+Semantic Search**: Better keyword matching
2. **Query Preprocessing**: Extract filters automatically
3. **Streaming Responses**: Reduce perceived latency
4. **Shopping List Generation**: From liked/planned recipes
5. **Meal Planning**: Multi-day variety optimization
6. **Web UI**: Streamlit/Gradio frontend

---

## References

- [PyTorch RTX 5090 Support](https://x.com/PyTorch/status/1887977473578844448)
- [sentence-transformers: all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- [Food.com Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
