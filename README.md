# MemLayer – The Plug-and-play persistent memory for your LLMs

**The memory layer for LLMs - add persistent, intelligent memory to any LLM in minutes.**

MemLayer transforms stateless LLMs into memory-enabled AI assistants that remember context across conversations, extract structured knowledge, and proactively surface relevant information when needed.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<p align="center">
  <img src="./memlayer.png" alt="MemLayer Overview" width="600">
</p>

## Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Key Concepts](#key-concepts)
- [Memory Modes](#memory-modes)
- [Search Tiers](#search-tiers)
- [Providers](#providers)
- [Advanced Features](#advanced-features)
- [Examples](#examples)
- [Performance](#performance)
- [Documentation](#documentation)
- [Contributing](#contributing)

##  Features

- **Universal LLM Support**: Works with OpenAI, Claude, Gemini, Ollama models
- **Plug-and-play**: Install with `pip install memlayer` and get started in minutes — minimal setup required.
- **Intelligent Memory Filtering**: Three operation modes (LOCAL/ONLINE/LIGHTWEIGHT) automatically filter important information
- **Hybrid Search**: Combines vector similarity + knowledge graph traversal for accurate retrieval
- **Three Search Tiers**: Fast (<100ms), Balanced (<500ms), Deep (<2s) optimized for different use cases
- **Knowledge Graph**: Automatically extracts entities, relationships, and facts from conversations
- **Proactive Reminders**: Schedule tasks and get automatic reminders when they're due
- **Built-in Observability**: Trace every search operation with detailed performance metrics
- **Flexible Storage**: ChromaDB (vector) + NetworkX (graph) or graph-only mode
- **Production Ready**: Serverless-friendly with fast cold starts using online mode

##  Quick Start

### Installation

```bash
pip install memlayer
```

### Basic Usage

```python
from memlayer.wrappers.openai import OpenAI

# Initialize with memory capabilities
client = OpenAI(
    model="gpt-4.1-mini",
    storage_path="./memories",
    user_id="user_123"
)

# Store information automatically
client.chat([
    {"role": "user", "content": "My name is Alice and I work at TechCorp"}
])

# Retrieve information automatically (no manual prompting needed!)
response = client.chat([
    {"role": "user", "content": "Where do I work?"}
])
# Response: "You work at TechCorp."
```

That's it! MemLayer automatically:
1. ✅ Filters salient information using ML-based classification
2. ✅ Extracts structured facts, entities, and relationships
3. ✅ Stores memories in hybrid vector + graph storage
4. ✅ Retrieves relevant context for each query
5. ✅ Injects memories seamlessly into LLM context

##  Key Concepts

### Salience Filtering
Not all conversation content is worth storing. MemLayer uses **salience gates** to intelligently filter:
- ✅ **Save**: Facts, preferences, user info, decisions, relationships
- ❌ **Skip**: Greetings, acknowledgments, filler words, meta-conversation

### Hybrid Storage
Memories are stored in two complementary systems:
- **Vector Store (ChromaDB)**: Semantic similarity search for facts
- **Knowledge Graph (NetworkX)**: Entity relationships and structured knowledge

### Automatic Consolidation
After each conversation, background threads:
1. Extract facts, entities, and relationships using LLM
2. Store facts in vector database with embeddings
3. Build knowledge graph with entities and relationships
4. Index everything for fast retrieval

##  Memory Modes

MemLayer offers three modes that control both **memory filtering (salience)** and **storage**:

### 1. LOCAL Mode (Default)
```python
client = OpenAI(salience_mode="local")
```
- **Filtering**: Sentence-transformers ML model (high accuracy)
- **Storage**: ChromaDB (vector) + NetworkX (graph)
- **Startup**: ~10s (model loading)
- **Best for**: High-volume production, offline apps
- **Cost**: Free (no API calls)

### 2. ONLINE Mode
```python
client = OpenAI(salience_mode="online")
```
- **Filtering**: OpenAI embeddings API (high accuracy)
- **Storage**: ChromaDB (vector) + NetworkX (graph)
- **Startup**: ~2s (no model loading!)
- **Best for**: Serverless, cloud functions, fast cold starts
- **Cost**: ~$0.0001 per operation

### 3. LIGHTWEIGHT Mode
```python
client = OpenAI(salience_mode="lightweight")
```
- **Filtering**: Keyword-based (medium accuracy)
- **Storage**: NetworkX only (no vector storage!)
- **Startup**: <1s (instant)
- **Best for**: Prototyping, testing, low-resource environments
- **Cost**: Free (no embeddings at all)

**Performance Comparison:**
```
Mode          Startup Time    Accuracy    API Cost    Storage
──────────────────────────────────────────────────────────────
LOCAL         ~10s            High        Free        Vector+Graph
ONLINE        ~2s             High        $0.0001/op  Vector+Graph  
LIGHTWEIGHT   <1s             Medium      Free        Graph-only
```

##  Search Tiers

MemLayer provides three search tiers optimized for different latency requirements:

### Fast Tier (<100ms)
```python
# Automatic - LLM chooses based on query complexity
response = client.chat([{"role": "user", "content": "What's my name?"}])
```
- 2 vector search results
- No graph traversal
- Perfect for: Real-time chat, simple factual recall

### Balanced Tier (<500ms)  DEFAULT
```python
# Automatic - handles most queries well
response = client.chat([{"role": "user", "content": "Tell me about my projects"}])
```
- 5 vector search results
- No graph traversal
- Perfect for: General conversation, most use cases

### Deep Tier (<2s)
```python
# Explicit request or auto-detected for complex queries
response = client.chat([{
    "role": "user",
    "content": "Use deep search: Tell me everything about Alice and her relationships"
}])
```
- 10 vector search results
- Graph traversal enabled (entity extraction + 1-hop relationships)
- Perfect for: Research, "tell me everything", multi-hop reasoning

## 🔌 Providers

MemLayer works with all major LLM providers:

### OpenAI
```python
from memlayer.wrappers.openai import OpenAI

client = OpenAI(
    model="gpt-4.1-mini",  # or gpt-4.1, gpt-5, etc.
    storage_path="./memories",
    user_id="user_123"
)
```

### Claude (Anthropic)
```python
from memlayer.wrappers.claude import Claude

client = Claude(
    model="claude-4-sonnet",
    storage_path="./memories",
    user_id="user_123"
)
```

### Google Gemini
```python
from memlayer.wrappers.gemini import Gemini

client = Gemini(
    model="gemini-2.5-flash",
    storage_path="./memories",
    user_id="user_123"
)
```

### Ollama (Local)
```python
from memlayer.wrappers.ollama import Ollama

client = Ollama(
    host="http://localhost:11434",
    model="qwen3:1.7b",  # or llama3.2, mistral, etc.
    storage_path="./memories",
    user_id="user_123",
    salience_mode="local"  # Run 100% offline!
)
```

**All providers share the same API** - switch between them seamlessly!

##  Advanced Features

### Proactive Task Reminders

```python
# User schedules a task
client.chat([{
    "role": "user",
    "content": "Remind me to submit the report next Friday at 9am"
}])

# Later, when the task is due, MemLayer automatically injects it
response = client.chat([{"role": "user", "content": "What should I do today?"}])
# Response includes: "Don't forget to submit the report - it's due today at 9am!"
```

### Observability & Tracing

```python
response = client.chat(messages)

# Inspect search performance
if client.last_trace:
    print(f"Search tier: {client.last_trace.events[0].metadata.get('tier')}")
    print(f"Total time: {client.last_trace.total_duration_ms}ms")
    
    for event in client.last_trace.events:
        print(f"  {event.event_type}: {event.duration_ms}ms")
```

### Custom Salience Threshold

```python
# Control memory filtering strictness
client = OpenAI(
    salience_threshold=-0.1  # Permissive (saves more)
    # salience_threshold=0.0   # Balanced (default)
    # salience_threshold=0.1   # Strict (saves less)
)
```

### Knowledge Graph Extraction

```python
# Manually extract structured knowledge
kg = client.analyze_and_extract_knowledge(
    "Alice leads Project Phoenix in the London office. The project uses Python and React."
)

print(kg["facts"])         # ["Alice leads Project Phoenix", ...]
print(kg["entities"])      # [{"name": "Alice", "type": "Person"}, ...]
print(kg["relationships"]) # [{"subject": "Alice", "predicate": "leads", "object": "Project Phoenix"}]
```

##  Examples

Explore the `examples/` directory for comprehensive examples:

### Basics
```bash
# Getting started
python examples/01_basics/getting_started.py
```

### Search Tiers
```bash
# Try all three search tiers
python examples/02_search_tiers/fast_tier_example.py
python examples/02_search_tiers/balanced_tier_example.py
python examples/02_search_tiers/deep_tier_example.py

# Compare them side-by-side
python examples/02_search_tiers/tier_comparison.py
```

### Advanced Features
```bash
# Proactive task reminders
python examples/03_features/task_reminders.py

# Knowledge graph visualization
python examples/03_features/test_knowledge_graph.py
```

### Benchmarks
```bash
# Compare salience modes
python examples/04_benchmarks/compare_salience_modes.py
```

### Providers
```bash
# Try different LLM providers
python examples/05_providers/openai_example.py
python examples/05_providers/claude_example.py
python examples/05_providers/gemini_example.py
python examples/05_providers/ollama_example.py
```

See [examples/README.md](examples/README.md) for full documentation.

##  Performance

### Salience Mode Comparison
Real-world startup times from benchmarks:

```
Mode          First Use    Memory Savings    Trade-off
─────────────────────────────────────────────────────────
LIGHTWEIGHT   ~5s          No embeddings     No semantic search
ONLINE        ~5s          5s faster         Small API cost
LOCAL         ~10s         No API cost       11s model loading
```

### Search Tier Latency
Typical query latencies:

```
Tier        Latency    Vector Results    Graph    Use Case
────────────────────────────────────────────────────────────
Fast        50-150ms   2                 No       Real-time chat
Balanced    200-600ms  5                 No       General use
Deep        800-2500ms 10                Yes      Research queries
```

### Memory Consolidation
Background processing (non-blocking):

```
Step                        Time      Async
──────────────────────────────────────────────
Salience filtering         ~10ms      Yes
Knowledge extraction       ~1-2s      Yes (background thread)
Vector storage             ~50ms      Yes
Graph storage              ~20ms      Yes
Total (non-blocking)       ~0ms       User doesn't wait!
```

##  Documentation

- **Examples**: [examples/README.md](examples/README.md) - Comprehensive examples
- **Providers**: [examples/05_providers/README.md](examples/05_providers/README.md) - Provider comparison
- **Salience Modes**: [docs/SALIENCE_MODES.md](docs/SALIENCE_MODES.md) - Memory filtering details
- **Tests**: [tests/README.md](tests/README.md) - Running tests and benchmarks

##  Tunable features (quick index)

The project exposes several runtime/configuration knobs you can tune to match latency, cost, and accuracy trade-offs. Detailed docs for each area live in the `docs/` folder:

- `docs/tuning/operation_mode.md` — How to choose between `online`, `local`, and `lightweight` modes and the effect on storage and embeddings.
- `docs/tuning/intervals.md` — Scheduler and curation interval configuration (`scheduler_interval_seconds`, `curation_interval_seconds`) and practical guidance.
- `docs/tuning/salience_threshold.md` — How to adjust `salience_threshold` and expected behavior.
- `docs/services/consolidation.md` — Consolidation pipeline internals and how to call it programmatically (including `update_from_text`).
- `docs/services/curation.md` — How memory curation works, archiving rules, and how to run/stop the curation service.
- `docs/storage/chroma.md` — ChromaDB notes: metadata types, connection handling, and Windows file-lock guidance.
- `docs/storage/networkx.md` — Knowledge graph persistence, expected node schemas, and backup/restore tips.
- `docs/providers/` — Provider-specific notes for OpenAI, Claude, Gemini, and Ollama (`docs/providers/README.md` links to individual provider pages).
- `docs/examples/README.md` — Index of important examples and how to run them (includes `examples/06_api/direct_knowledge_ingestion.py`).

Use the docs when tuning for production. The following `docs/` files were added to this repository and provide detailed, practical guidance.

##  Development

### Setup

```bash
# Clone repository
git clone https://github.com/divagr18/memlayer.git
cd memlayer

# Install dependencies
pip install -e .

# Run tests
python -m pytest tests/

# Run examples
python examples/01_basics/getting_started.py
```

### Project Structure

```
memlayer/
├── memlayer/           # Core library
│   ├── wrappers/          # LLM provider wrappers
│   ├── storage/           # Storage backends (ChromaDB, NetworkX)
│   ├── services.py        # Search & consolidation services
│   ├── ml_gate.py         # Salience filtering
│   └── embedding_models.py # Embedding model implementations
├── examples/              # Organized examples by category
│   ├── 01_basics/
│   ├── 02_search_tiers/
│   ├── 03_features/
│   ├── 04_benchmarks/
│   └── 05_providers/
├── tests/                 # Tests and benchmarks
├── docs/                  # Documentation
└── README.md              # This file
```

##  Contributing

Contributions are welcome! Here's how you can help:

1. **Report bugs** - Open an issue with reproduction steps
2. **Suggest features** - Share your use case and requirements
3. **Submit PRs** - Fix bugs, add features, improve docs
4. **Share examples** - Show us what you've built!

Please keep PRs focused and include tests for new features.

##  Contact & Support

- **Author/Maintainer**: Divyansh Agrawal
- **Email**: keshav.r.1925@gmail.com
- **GitHub**: [divagr18](https://github.com/divagr18)
- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/divagr18/memlayer/issues)

For security vulnerabilities, please email directly with `SECURITY` in the subject line instead of opening a public issue.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [ChromaDB](https://www.trychroma.com/) for vector storage
- Uses [NetworkX](https://networkx.org/) for knowledge graph operations
- Powered by [sentence-transformers](https://www.sbert.net/) for local embeddings
- Supports [OpenAI](https://openai.com/), [Anthropic](https://www.anthropic.com/), [Google Gemini](https://ai.google.dev/), and [Ollama](https://ollama.ai/)

---

**Made with ❤️ for the AI community**

Give your LLMs memory. Try MemLayer today! 
