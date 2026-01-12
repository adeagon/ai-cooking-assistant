# Phase 2: Potential Improvements Based on Test Results

## Performance Analysis

### Current Performance
- **Search Speed**: 4-10ms (target was <200ms)
- **Headroom**: **20-50x faster than required**
- **Bottleneck**: Not search speed, but likely semantic quality

### Key Observation
We optimized for speed with `all-MiniLM-L6-v2` (small, fast model), but we have massive performance budget remaining.

---

## Recommended Improvements

### 1. Upgrade Embedding Model ⭐ **HIGH IMPACT**

**Current**: `all-MiniLM-L6-v2`
- Embedding dimension: 384
- Model size: 90MB
- Speed: Fast ✅
- Quality: Good, but not state-of-the-art

**Recommended Options**:

#### Option A: `all-mpnet-base-v2` (Recommended for Phase 2)
- Embedding dimension: 768 (2x larger)
- Model size: 420MB
- Speed: Still fast on CPU (~15-20ms expected)
- Quality: **Significantly better** semantic understanding
- **Trade-off**: 2x slower, but still well under 200ms target

#### Option B: `BAAI/bge-large-en-v1.5` (Best quality, for GPU)
- Embedding dimension: 1024
- Model size: 1.34GB
- Speed: Slower on CPU (~40-60ms), but **very fast on GPU** with RTX 5090
- Quality: **State-of-the-art** semantic search
- **Trade-off**: Requires GPU setup, larger index

#### Option C: `sentence-transformers/multi-qa-mpnet-base-dot-v1`
- Embedding dimension: 768
- Optimized specifically for Q&A and retrieval tasks
- Similar performance to all-mpnet-base-v2

**Recommendation**: Start with **all-mpnet-base-v2** - better quality with minimal performance impact.

### 2. Increase k_retrieve ⭐ **RECOMMENDED**

**Current**: `k_retrieve = 30`

**Observation from tests**:
- Retrieving k=100 takes only 6ms (vs 4ms for k=30)
- More candidates = better reranking results in Phase 3

**Recommendation**:
```python
k_retrieve = 100  # or even 200 - only adds ~2-4ms
k_rerank = 20     # instead of 10
k_context = 5-6   # instead of 4
```

**Rationale**:
- Cross-encoder reranking (Phase 3) will benefit from more candidates
- Marginal cost is trivial (2ms)
- Better final results to LLM

### 3. Add Hybrid Search (BM25 + Semantic) ⭐ **HIGH VALUE**

**Current**: Pure semantic search only

**Problem**:
- Semantic search is great for concepts but can miss exact keyword matches
- Example: "basil pesto pasta" might rank lower than "italian herb pasta" semantically

**Solution**: Hybrid retrieval
```python
# Retrieve 50 from semantic + 50 from BM25, then merge and rerank
semantic_results = vector_search(query, k=50)
keyword_results = bm25_search(query, k=50)
combined = merge_and_dedupe(semantic_results, keyword_results)
final = rerank(combined, k=20)
```

**Benefits**:
- Better handling of specific ingredients ("chicken thighs" vs "chicken")
- Exact phrase matching ("gluten free")
- Fallback when semantic embeddings struggle
- rank-bm25 is already in dependencies!

**Performance Impact**: +5-10ms (still well under budget)

### 4. Enrich Embedding Text ⭐ **MODERATE IMPACT**

**Current embedding text**:
```
{title}. Tags: {tags}. Ingredients: {ingredients_normalized}
```

**Proposed**:
```
{title}. Tags: {tags}. Ingredients: {ingredients_normalized}.
Cooking method: {derived_from_tags}. Time: {minutes} minutes.
Difficulty: {easy/medium/hard}.
```

**Additional fields to consider**:
- First 1-2 instruction steps (cooking technique indicators)
- Derived flags: `quick`, `one-pot`, `no-cook`, `budget-friendly`
- Nutrition indicators: `high-protein`, `low-calorie`

**Trade-off**: Larger embeddings, but with 20-50x headroom, we can afford it

### 5. Add More Metadata Filters

**Current metadata**:
- `recipe_id`, `title`, `rating_avg`, `rating_count`, `minutes`, `tags`

**Suggested additions**:
- `n_ingredients` - for "simple recipes"
- `n_steps` - for complexity filtering
- `difficulty` - derived heuristic (could use steps + time)
- Nutrition flags - derived from ingredients or tags
  - `high_protein`, `low_carb`, `vegetarian`, `vegan`

**Performance impact**: Minimal (just more metadata fields)

### 6. Query Preprocessing & Expansion

**Current**: Raw query directly to embedding

**Improvements**:
1. **Synonym expansion**: "hot" → "hot OR spicy OR fiery"
2. **Ingredient normalization**: "chicken breasts" → "chicken breast" OR "chicken"
3. **Time extraction**: "30 minutes" → add time filter automatically
4. **Dietary extraction**: "vegan" → add to metadata filter

**Example**:
```python
query = "quick vegan dinner under 30 minutes"
# Extract filters:
# - max_minutes = 30
# - dietary = "vegan"
# Clean query: "quick dinner"
```

**Benefits**: Better results through structured filters vs semantic-only

---

## Implementation Priority

### Immediate (Before Phase 3)
1. ✅ **Increase k_retrieve to 50-100** - Zero risk, high value for reranking
2. ⭐ **Upgrade to all-mpnet-base-v2** - Significant quality improvement

### Phase 2.5 (Optional enhancement sprint)
3. 🔧 **Add hybrid BM25 search** - Better keyword matching
4. 📝 **Enrich embedding text** - Include cooking method, difficulty

### Phase 3+ (After reranking is working)
5. 🎯 **Query preprocessing** - Extract filters, normalize ingredients
6. 📊 **Additional metadata** - Difficulty, nutrition flags

---

## Concrete Next Steps

### Option A: Conservative (Recommended)
Just adjust settings before Phase 3:
```python
# src/app/settings.py
k_retrieve: int = Field(default=100)  # was 30
k_rerank: int = Field(default=20)     # was 10
k_context: int = Field(default=6)     # was 4
```

### Option B: Quality Upgrade (Recommended if time permits)
1. Change embedding model to `all-mpnet-base-v2`
2. Rebuild vector store (run `ingest embed` again, ~15 minutes)
3. Increase k_retrieve to 100
4. Run tests to verify quality improvement

### Option C: Hybrid Search (Advanced)
1. Implement BM25 indexing in parallel to ChromaDB
2. Create hybrid retriever that merges results
3. More complex but significantly better results

---

## Testing Plan for Improvements

### If upgrading embedding model:
```bash
# Rebuild with new model
python -m src.app.cli ingest embed

# Run comparative tests
pytest tests/test_retrieval_comprehensive.py -v

# Manual quality checks
python -m src.app.cli search "chicken with tomato and basil"
python -m src.app.cli search "quick healthy dinner under 20 minutes"
```

### Success Metrics:
- Search speed still <50ms (have 150ms budget)
- Top result scores increase (currently 0.8-0.9, target >0.9)
- Better handling of multi-ingredient queries
- More diverse top results

---

## Risk Assessment

### Low Risk (Do Now)
- ✅ Increase k_retrieve/k_rerank/k_context
- ✅ Adjust settings only, no code changes

### Medium Risk (Test First)
- ⚠️ Upgrade embedding model - requires rebuild
- ⚠️ Enrich embedding text - requires rebuild

### Higher Risk (Phase 3+)
- ⚠️⚠️ Hybrid search - architectural change
- ⚠️⚠️ Query preprocessing - complex logic

---

## Resource Utilization

### Current State
- **CPU**: Minimal usage (sentence-transformers on CPU)
- **GPU (RTX 5090)**: **Unused** ⚠️
- **Memory**: Small model (90MB)
- **Disk**: 88K recipes indexed

### Opportunity
With RTX 5090 (32GB VRAM), you could:
1. Use GPU-accelerated embeddings (10-100x faster)
2. Use larger, better models (bge-large-en-v1.5)
3. Process embeddings in parallel
4. Run Llama 3.3 70B + embedding model simultaneously

**Note**: sentence-transformers automatically uses GPU if available. Just need to rebuild index.

---

## Recommendation Summary

**Before Phase 3 starts**:
1. ✅ Increase `k_retrieve` to 100 (edit settings.py only)
2. ✅ Increase `k_rerank` to 20
3. ✅ Increase `k_context` to 6

**Optional quality upgrade** (if 15 minutes available):
1. ⭐ Change `embedding_model` to `"all-mpnet-base-v2"`
2. ⭐ Rebuild vector store: `python -m src.app.cli ingest embed`
3. ⭐ Verify quality improvement

**Phase 3+**:
1. 🔧 Implement hybrid BM25+semantic search
2. 🎯 Add query preprocessing and filter extraction

This balances quality improvement with minimal risk and time investment.
