# Phase 3: Cross-Encoder Reranking + Recipe Cards - Summary

**Date**: 2026-01-12
**Branch**: `phase-3-reranking`
**Status**: Complete ✅

---

## Overview

Phase 3 adds cross-encoder reranking and recipe card generation to improve search relevance and prepare compact recipe representations for LLM context (Phase 4).

**Architecture**: retrieve (k=100) → rerank (k=20) → build cards (k=6)

---

## New Features

### 1. Cross-Encoder Reranking

**File**: `src/retrieval/rerank.py`

- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (GPU-accelerated)
- **Purpose**: Rerank vector search candidates for improved relevance
- **Performance**: ~37ms to rerank 100 candidates on RTX 5090
- **API**: `RecipeReranker.rerank(query, candidates, top_k)`

### 2. Recipe Card Builder

**File**: `src/retrieval/recipe_cards.py`

- **Model**: `RecipeCard` (Pydantic, already defined in Phase 1)
- **Purpose**: Create compact recipe representations for LLM prompts
- **Target**: 120-250 tokens per card
- **Features**:
  - Template-based one-sentence summaries (heuristic, not LLM)
  - `why_match` explanations (ingredient/tag/time matching)
  - Key ingredient selection (deprioritizes salt/pepper/oil)
  - Cuisine/dish type detection from tags

### 3. CLI Enhancements

**File**: `src/app/cli.py` (updated `search` command)

New flags:
- `--rerank` / `-r`: Enable cross-encoder reranking
- `--cards` / `-c`: Display detailed recipe cards (implies `--rerank`)

Example usage:
```bash
# Basic vector search
python -m src.app.cli search "chicken tomato spicy"

# With reranking
python -m src.app.cli search "chicken tomato spicy" --rerank

# With recipe cards (detailed display)
python -m src.app.cli search "chicken tomato spicy" --cards
```

---

## Implementation Details

### RecipeReranker

**Class**: `RecipeReranker`

```python
def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2")
def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int = 20) -> list[RetrievalResult]
```

- Uses `sentence_transformers.CrossEncoder`
- GPU-accelerated (automatically uses CUDA if available)
- Builds query-title pairs for batch prediction
- Updates `RetrievalResult.score` with cross-encoder scores

### RecipeCardBuilder

**Class**: `RecipeCardBuilder`

```python
def __init__(self, db_path: Path)
def build_cards(self, results: list[RetrievalResult], query: str, max_cards: int = 6) -> list[RecipeCard]
def build_card(self, recipe: Recipe, query: str, score: float) -> RecipeCard
def generate_summary(self, recipe: Recipe) -> str
def compute_why_match(self, recipe: Recipe, query: str) -> str
def select_key_ingredients(self, ingredients: list[str], max_count: int = 12) -> list[str]
```

**Summary Generation**:
- Identifies cuisine (mexican, italian, thai, etc.)
- Identifies dish type (soup, salad, pasta, etc.)
- Identifies cooking method (baked, grilled, fried, etc.)
- Includes top 3 main ingredients
- Notes time (quick, 30-minute, 2+ hour)

**Why Match Computation**:
- Matches query terms against ingredients
- Matches query terms against tags
- Notes quick prep for time-related queries
- Notes dietary matches (vegetarian, healthy, etc.)
- Highlights highly-rated recipes (≥4.5 stars, ≥10 reviews)

---

## Test Coverage

### New Test Files

1. **`tests/test_rerank.py`** (10 tests)
   - Unit tests: empty candidates, metadata preservation, score updates, top_k limits
   - Integration tests: relevance improvement, performance, diverse queries

2. **`tests/test_recipe_cards.py`** (18 tests)
   - Unit tests: summary generation, why_match computation, ingredient selection
   - Integration tests: card building from results, token budget validation

3. **`tests/test_regression.py`** (14 tests) - **Added after initial implementation**
   - Phase 1 regression: Recipe models, ingredient normalization, quality filters, database access
   - Phase 2 regression: Embedding text, vector retrieval, filters, performance
   - Phase 3 backwards compatibility: Import paths, result compatibility
   - End-to-end regression: Complete pipeline, diverse queries

### Test Results

**Final**: `======================== 82 passed in 86.18s ========================`
- **42 new tests for Phase 3** (10 reranker + 18 recipe cards + 14 regression)
- 40 existing tests (all still passing)
- All integration tests now running (database path fix enabled recipe card integration tests)

### Bug Fixes During Testing

1. **Database path mismatch**: Fixed `settings.py` to use `data/sqlite/recipes.db` instead of `app.db`
   - Enabled all recipe card integration tests to run
   - Previously skipped tests now pass

---

## Performance

| Operation | Time (RTX 5090) | Notes |
|-----------|-----------------|-------|
| Vector retrieval (k=100) | ~300ms | Phase 2 baseline |
| Reranking (100→20) | ~37ms | Cross-encoder scoring |
| Card building (6 cards) | ~5ms | Template-based, no LLM |
| **Total (with rerank)** | **~342ms** | Well under 1s target |

---

## Files Modified

### Created

- `src/retrieval/rerank.py` - Cross-encoder reranker
- `src/retrieval/recipe_cards.py` - Recipe card builder
- `tests/test_rerank.py` - Reranker tests
- `tests/test_recipe_cards.py` - Card builder tests

### Modified

- `src/retrieval/__init__.py` - Added exports for new classes
- `src/app/cli.py` - Added `--rerank` and `--cards` flags
- `README.md` - Updated status to Phase 3 complete, added usage examples
- `CLAUDE.md` - Updated architecture and data flow

---

## Next Steps (Phase 4)

1. Implement `OllamaLLMClient` in `src/llm/`
2. Create LCEL chains for conversational flow
3. Integrate RecipeCards into LLM prompts
4. Add clarifying questions and multi-turn dialogue
5. Test end-to-end with Llama 3.3 70B

---

## Known Issues

None. All tests passing, functionality complete.

**Note**: Recipe cards require SQLite database to be populated. Run `ingest process` before using `--cards` flag.

---

## Architecture Diagram

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       v
┌────────────────────────┐
│ RecipeRetriever        │
│ Vector search (k=100)  │  Phase 2
└──────┬─────────────────┘
       │
       v
┌────────────────────────┐
│ RecipeReranker         │
│ Cross-encoder (k=20)   │  Phase 3
└──────┬─────────────────┘
       │
       v
┌────────────────────────┐
│ RecipeCardBuilder      │
│ Build cards (k=6)      │  Phase 3
└──────┬─────────────────┘
       │
       v
┌─────────────┐
│   Display   │ (or pass to LLM in Phase 4+)
└─────────────┘
```

---

## Commit Message

```
Phase 3: Add cross-encoder reranking and recipe card generation

- Implement RecipeReranker with ms-marco-MiniLM-L-6-v2 (GPU-accelerated)
- Implement RecipeCardBuilder with template-based summaries
- Add --rerank and --cards flags to search command
- Add 42 new tests (10 reranker, 18 card builder, 14 regression)
- Fix database path mismatch (settings now use recipes.db)
- Add regression test suite to verify Phase 1-3 compatibility
- Update documentation (README.md, CLAUDE.md, PHASE_3_SUMMARY.md)

All 82 tests passing (100% pass rate). Ready for Phase 4 (LLM integration).

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```
