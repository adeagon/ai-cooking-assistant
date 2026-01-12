# AI Cooking Assistant

Local recipe assistant using RAG (Retrieval-Augmented Generation) with Llama 3.3 70B via Ollama.

## Features

- Recommends real recipes from Food.com dataset (180K+ recipes)
- Learns user preferences and dietary restrictions
- Asks clarifying questions for better recommendations
- Supports "ingredients on hand" queries
- Fully local (no cloud dependencies)

## Tech Stack

- **LLM**: Llama 3.3 70B Instruct via Ollama
- **Vector Store**: ChromaDB
- **Embeddings**: sentence-transformers
- **Framework**: LangChain (LCEL chains)
- **Database**: SQLite
- **CLI**: Typer

## Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and configure
3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Install and run Ollama:
   ```bash
   ollama serve
   ollama pull llama3.3:70b
   ```

## Usage

Start the interactive chat:
```bash
python -m src.app.cli chat
```

Or use the installed command:
```bash
recipe-assistant chat
```

## Development

Run tests:
```bash
pytest
```

See `PROJECT_PLAN.md` and `CLAUDE.md` for detailed architecture and development guidance.
