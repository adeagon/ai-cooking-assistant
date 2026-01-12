# PROJECT_PLAN.md — Local Recipe Assistant (RAG + Agentic Dialogue)

**Goal:** Build a fully-local, interactive dinner-planning assistant that recommends *real* recipes from an indexed dataset, learns preferences from feedback, asks clarifying questions, and supports “ingredients on hand” queries.

**Primary dev environment:** Windows 11  
**GPU:** RTX 5090 FE (32GB VRAM)  
**LLM:** Llama 3.3 70B Instruct (local)  
**Runtime (recommended):** Ollama (local HTTP API)
**Dataset**: Food.com Recipes and Interactions. This dataset consists of 180K+ recipes and 700K+ recipe reviews covering 18 years of user interactions and uploads on Food.com. https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

---

## 1) Decisions / Constraints (Lock These In)

### 1.1 Model + runtime
- **Model:** Llama 3.3 70B Instruct
- **Runtime:** **Ollama** (start here for lowest friction + stable API)
- **Key inference constraint:** 70B + 32GB VRAM ⇒ expect **quantization + some CPU RAM offload**. Optimize UX with:
  - **Tight retrieved context** (compact “recipe cards”)
  - **Small prompt top_k (3–6)**
  - **Summarized memory** (rolling summary + pinned preferences)

### 1.2 Local-first
- Everything runs on the Windows machine:
  - Local LLM runtime (Ollama)
  - Local vector store (Chroma or FAISS)
  - Local SQLite for user state (prefs, pantry, history, ratings)
  - No cloud dependencies in MVP

### 1.3 UI
- MVP: CLI conversational app
- Later: optional web UI (Streamlit/Gradio/FastAPI), but not required for v1

---

## 2) Architecture (Specific)

### 2.1 High-level components
1. **Ingestion pipeline (offline/periodic)**
   - Load recipe dataset (initially Food.com Recipes + Interactions)
   - Clean/normalize text + ingredients
   - Compute embeddings
   - Persist:
     - Vector index (Chroma/FAISS)
     - Canonical recipe store (SQLite tables or JSONL + SQLite pointers)
2. **Runtime app (CLI)**
   - Maintains dialogue loop
   - Extracts constraints (ingredients, time, dietary, dislikes)
   - Runs retrieval + rerank
   - Builds compact “recipe cards” context
   - Calls LLM (Ollama) for conversational output
   - Saves feedback and updates user profile + rolling summary

### 2.2 Data stores (local)
**SQLite (structured state):**
- `users` (just one user profile; still model it cleanly)
- `pinned_preferences` (long-lived defaults)
- `session_state` (per dinner planning session)
- `pantry_items` (current inventory; optional)
- `recipe_feedback` (liked/disliked, rating, notes, cooked_date)
- `cooking_history` (what you cooked and when)
- `recipe_cache` (optional: precomputed summary fields)

**Vector store (semantic retrieval):**
- Chroma persistent directory (`./data/chroma/`) OR FAISS index (`./data/faiss/`)
- Stores embeddings + metadata pointers (recipe_id, rating, tags, key ingredients)

### 2.3 Retrieval architecture (RAG)
**Principle:** retrieve wide → rerank narrow → prompt compact

- Retrieve candidates from vector DB: `k_retrieve = 30`
- Rerank candidates with local cross-encoder: `k_rerank = 10`
- Provide LLM prompt with top `k_context = 4` recipe cards

**Reranker (local) recommendations:**
- Fast baseline: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Stronger (slower): `BAAI/bge-reranker-base` or `...-large`

### 2.4 “Tight context” recipe cards (MUST)
Do **NOT** dump full recipes into the prompt during recommendation/ranking.

**RecipeCard fields (target 120–250 tokens per recipe):**
- `title`
- `rating_avg`, `rating_count`
- `tags` (cuisine, course, technique, derived flags: spicy/healthy/quick)
- `time_total` (if available)
- `key_ingredients` (8–15 normalized ingredients)
- `one_sentence_summary` (precomputed at ingestion)
- `why_match` (computed at query time: e.g., “matches chicken+tomato; avoids fish”)

**Full recipe display rule:**
- If user asks: “show recipe / steps / ingredients list” ⇒ fetch from DB and print verbatim
- Only use LLM to:
  - format into checklist,
  - generate timeline,
  - generate shopping list,
  - propose substitutions

### 2.5 Memory model (3-layer; MUST)
Avoid dumping the full chat log into context.

1) **Pinned preferences (persistent; structured)**
- spice level, cuisine prefs, dislikes, diet, typical time limit, etc.
- Stored in SQLite; injected into prompt as compact JSON-like block

2) **Session constraints (tonight; structured)**
- ingredients on hand, avoid_tonight, time_limit, “healthy” goal, etc.
- Stored in SQLite or in-memory for session

3) **Rolling summary (short text)**
- updated each turn: 1–3 sentences
- include only actionable constraints and decisions

**Optional:** include last 4–8 messages verbatim for conversational continuity.

### 2.6 LLM integration (Ollama)
- Use Ollama as a local HTTP inference server
- Configure:
  - context length target: start at **8192** (tune upward only if needed)
  - temperature: low-moderate (e.g., 0.2–0.5) for reliable suggestions

**LLM abstraction layer (required):**
Implement a thin interface so you can swap runtimes later (LM Studio, llama.cpp, vLLM):
- `LLMClient.chat(messages, options) -> assistant_text`
- `OllamaLLMClient` implements this via HTTP

### 2.7 LangChain approach (LCEL)
Use LCEL to keep chains explicit and testable.

Recommended chain graph:
1) `constraint_extractor_chain` (LLM or rules)
2) `retrieval_chain` (vector search + metadata filters)
3) `rerank_chain` (cross-encoder rerank; deterministic code)
4) `response_chain` (LLM: ask clarifying questions OR propose recipes)

---

## 3) Tech Stack (Pinned Versions Later)

### 3.1 Python libraries
- Core: `python>=3.11`
- LangChain: `langchain`, `langchain-community`, `langchain-core`
- Vector store: `chromadb` (or `faiss-cpu` / `faiss-gpu` if needed)
- Embeddings: `sentence-transformers`
- Rerankers: `sentence-transformers` cross-encoder OR `transformers`
- Storage: `sqlite3` (builtin) + `sqlalchemy` (optional)
- CLI: `typer` (recommended) or `argparse`
- Config: `pydantic-settings` or simple `yaml`
- Testing: `pytest`

### 3.2 Dataset
- Start: **Food.com Recipes + Interactions**
- Later: add personal recipe imports (JSON/YAML) and (if licensed) additional sources

---

## 4) Repository Layout

```
recipe-assistant/
  PROJECT_PLAN.md
  README.md
  pyproject.toml
  .env.example
  data/
    raw/                 # downloaded datasets (NOT in git)
    processed/           # normalized recipes (jsonl/parquet)
    chroma/              # persistent vector db
    sqlite/              # app.db
  src/
    app/
      cli.py
      settings.py
      logging.py
    llm/
      base.py
      ollama_client.py
      prompts.py
    ingest/
      load_foodcom.py
      normalize.py
      build_embeddings.py
      build_vectorstore.py
    retrieval/
      retriever.py
      filters.py
      rerank.py
      recipe_cards.py
    memory/
      profile_store.py
      session_store.py
      summarizer.py
    domain/
      models.py           # Pydantic models: Recipe, RecipeCard, PreferenceProfile
    eval/
      golden_queries.yaml
      run_eval.py
  tests/
    test_ingest.py
    test_retrieval.py
    test_memory.py
    test_cli_smoke.py
```

---

## 5) Core Data Models (Pydantic)

### 5.1 Recipe (canonical)
- `recipe_id: str`
- `title: str`
- `ingredients: list[str]` (normalized)
- `instructions: list[str]`
- `tags: dict[str, str|bool|list[str]]` (cuisine, course, technique, derived flags)
- `rating_avg: float|None`
- `rating_count: int|None`
- `source: str` (foodcom, user, etc.)

### 5.2 PreferenceProfile (pinned)
- `spice_level: Literal["none","mild","medium","hot"]`
- `diet: Literal["none","vegetarian","vegan","pescatarian","keto","gluten_free", ...]`
- `avoid_ingredients: list[str]`
- `preferred_cuisines: list[str]`
- `time_limit_default_minutes: int|None`

### 5.3 SessionState (tonight)
- `ingredients_on_hand: list[str]`
- `avoid_tonight: list[str]`
- `goal: list[str]` (e.g., ["healthy","quick"])
- `time_limit_minutes: int|None`
- `servings: int|None`

### 5.4 RecipeCard (prompt)
(see section 2.4)

---

## 6) Prompting Strategy (Concrete)

### 6.1 System prompt (behavior)
Requirements:
- Ask 1–3 clarifying questions **if constraints are insufficient**
- Prefer suggestions grounded in retrieved RecipeCards
- Offer 2–4 options, each with:
  - short justification
  - which ingredients match
  - what’s missing (if pantry-based)
- Never invent recipe names; only recommend from provided cards
- If user wants full recipe: instruct app to fetch from DB

### 6.2 “Clarify vs Recommend” policy
Implement a simple gating function before calling retrieval:
- If user input is vague AND session constraints empty → ask clarifying questions
- Else → retrieve + rerank + recommend

Example “vague”:
- “What should we cook tonight?”
Example “not vague”:
- “We have chicken thighs and tomatoes; something healthy and quick.”

---

## 7) Development Plan (Phased)
- CREATE TO-DOs FOR ALL
### Phase 0 — Project scaffolding (Day 0–1)
- [ ] Create repo structure, venv, dependency management
- [ ] Add `.env.example` and `settings.py`
- [ ] Add basic logging + CLI skeleton (`typer`)
**Exit criteria:** `python -m app.cli chat` starts a loop and prints placeholder responses.

### Phase 1 — Data + ingestion (Day 1–3)
- [ ] Download Food.com dataset into `data/raw/`
- [ ] Implement loader: parse recipes + interactions
- [ ] Normalize ingredients (lowercase, strip units, canonicalize common variants)
- [ ] Produce `data/processed/recipes.jsonl`
- [ ] Write SQLite `recipes` table (or store JSONL + pointer)
**Exit criteria:** can print a recipe by `recipe_id`; dataset stats logged.

### Phase 2 — Embeddings + vector store (Day 3–5)
- [ ] Choose embedding model (start: `all-MiniLM-L6-v2`)
- [ ] Build embeddings in batches
- [ ] Persist Chroma index to `data/chroma/`
- [ ] Implement retrieval query; return top 10 titles
**Exit criteria:** “chicken tomato spicy” returns plausible titles quickly (<200ms retrieval).

### Phase 3 — Rerank + recipe cards (Day 5–7)
- [ ] Implement reranker (cross-encoder) for top 30 → top 10
- [ ] Implement `RecipeCard` builder:
  - precompute `one_sentence_summary`
  - derive tags (spicy/healthy/quick) heuristically where needed
- [ ] Verify prompt card size (token estimation optional)
**Exit criteria:** given a query + prefs, returns 4 compact cards + reasons.

### Phase 4 — LLM runtime + LCEL chat chain (Day 7–10)
- [ ] Install Ollama; verify `llama3.3` runs
- [ ] Implement `OllamaLLMClient` and `LLMClient` base
- [ ] Build LCEL chain:
  - clarify gate
  - retrieve + rerank + cards
  - response generation
- [ ] CLI interactive dinner session
**Exit criteria:** end-to-end: user asks, assistant asks clarifying Qs, then suggests 2–4 recipes from DB.

### Phase 5 — Memory + personalization (Day 10–14)
- [ ] Implement pinned preferences store in SQLite
- [ ] Implement session constraints store
- [ ] Implement rolling summary updater
- [ ] Implement feedback commands:
  - `like <recipe_id>` / `dislike <recipe_id>` / `rate <recipe_id> 1-5`
- [ ] Exclude already-liked recipes by default (configurable)
**Exit criteria:** assistant improves after a few likes/dislikes; avoids disliked ingredients.

### Phase 6 — Evaluation + hardening (Day 14–18)
- [ ] Create `eval/golden_queries.yaml` (10–30 typical queries)
- [ ] Build eval runner: measures relevance + constraint satisfaction
- [ ] Add smoke tests for ingestion/retrieval
- [ ] Add “safe fallbacks”:
  - if retrieval empty → broaden search or ask question
  - if model slow → reduce context or k_context
**Exit criteria:** stable CLI; predictable results; no hallucinated recipe names.

### Phase 7 — Optional UI (Later)
- [ ] Streamlit/Gradio chat frontend
- [ ] Pantry management UI (scan/checkbox list)
- [ ] Meal plan + grocery list export

---

## 8) Roadmap (Milestones)

### Milestone A — “Search works”
- Recipe dataset loaded + normalized
- Vector search returns relevant recipes
- No LLM yet
**Output:** `search "chicken tomato quick"` prints top results

### Milestone B — “Chat recommends real recipes”
- LLM integrated via Ollama
- Retrieval + rerank + recipe cards prompt
- Conversational suggestions + clarifying questions
**Output:** interactive CLI dinner assistant

### Milestone C — “Personalized assistant”
- Persistent profile + pantry + feedback
- Avoid repeats, learns likes/dislikes
**Output:** better suggestions week over week

### Milestone D — “Polish + UX”
- Web UI optional
- Exports: shopping list, grocery items, timeline
- Better ingredient normalization and tag derivation

---

## 9) Implementation Notes (Do This, Not That)

### 9.1 Avoid prompt bloat
- Keep `k_context <= 6`
- Keep cards compact
- Keep memory summarized + structured

### 9.2 Deterministic reranking
- Reranking should be **code**, not LLM
- Save reranker scores for debugging

### 9.3 Debuggability
Log at each turn:
- parsed constraints
- retrieval query string
- top candidates + scores
- reranked top list
- final cards passed to LLM

### 9.4 Swappable LLM runtime
All app code should depend only on `LLMClient` interface.

---

## 10) Immediate Next Steps (Actionable)

1) Install Ollama + confirm:
   - `ollama run llama3.3`
2) Scaffold repo + CLI loop
3) Acquire dataset (Food.com) and implement ingestion
4) Build Chroma index + retrieval command
5) Add reranker + recipe cards
6) Integrate LCEL + memory layers
7) Add feedback loop and persistence

---

## 11) “Claude Code” Implementation Guidance (How to Use This File)

When working in Claude Code:
- Treat each Phase as a separate implementation prompt.
- Include:
  - file paths to create/modify
  - acceptance criteria (Exit criteria)
  - tests to add
- Keep PRs small: one module per change set.

Suggested first Claude prompt:
- “Implement Phase 0 scaffolding: repo layout, settings, CLI skeleton with `typer`, basic logging. Output code with file paths.”
