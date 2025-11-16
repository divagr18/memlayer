# MemLayer Examples

Welcome to the MemLayer examples! This directory contains comprehensive examples organized by topic.

## 📁 Structure

```
examples/
├── 01_basics/           # Getting started with MemLayer
├── 02_search_tiers/     # Fast, Balanced, and Deep search modes
├── 03_features/         # Advanced features (tasks, knowledge graph)
├── 04_benchmarks/       # Performance comparisons
├── 05_providers/        # Provider-specific examples (OpenAI, Claude, etc.)
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Install MemLayer
```bash
pip install memlayer
```

### 2. Set your API key
```bash
# For OpenAI
export OPENAI_API_KEY='sk-...'

# For Claude
export ANTHROPIC_API_KEY='sk-ant-...'

# For Gemini
export GOOGLE_API_KEY='...'

# For Ollama (local, no key needed)
ollama pull qwen3:1.7b
```

### 3. Run your first example
```bash
python examples/01_basics/getting_started.py
```

## 📚 Examples by Category

### 🎓 Basics (01_basics/)
Start here if you're new to MemLayer!

**`getting_started.py`** - Simple introduction to MemLayer
- Store and retrieve memories
- Automatic knowledge consolidation
- Basic conversation patterns

```bash
python examples/01_basics/getting_started.py
```

---

### 🔍 Search Tiers (02_search_tiers/)
Learn about the three search modes optimized for different use cases.

MemLayer provides three search tiers optimized for different use cases.

**`fast_tier_example.py`** - Quick lookups (<100ms)
```bash
python examples/02_search_tiers/fast_tier_example.py
```
- 2 vector search results
- No graph traversal
- Real-time chat applications

**`balanced_tier_example.py`** - Standard search (<500ms) [DEFAULT]
```bash
python examples/02_search_tiers/balanced_tier_example.py
```
- 5 vector search results
- No graph traversal
- General conversation

**`deep_tier_example.py`** - Comprehensive search (<2s)
```bash
python examples/02_search_tiers/deep_tier_example.py
```
- 10 vector search results
- Graph traversal enabled
- Complex queries, relationship discovery

**`search_tiers_demo.py`** - Complete demonstration of all tiers
```bash
python examples/02_search_tiers/search_tiers_demo.py
```

**`tier_comparison.py`** - Side-by-side performance comparison
```bash
python examples/02_search_tiers/tier_comparison.py
```

| Tier | Latency | Results | Graph | Use Case |
|------|---------|---------|-------|----------|
| Fast | <100ms | 2 | ❌ | Chatbots, real-time |
| Balanced | <500ms | 5 | ❌ | General conversation |
| Deep | <2s | 10 | ✅ | Research, multi-hop reasoning |

---

### ⚡ Features (03_features/)
Advanced capabilities and integrations.

**`task_reminders.py`** - Proactive task management
```bash
python examples/03_features/task_reminders.py
```
- Schedule future tasks
- Automatic reminders when due
- Natural language date parsing

**`test_knowledge_graph.py`** - Knowledge graph demonstration
```bash
python examples/03_features/test_knowledge_graph.py
```
- Entity and relationship extraction
- Graph-based memory storage
- Visual inspection of knowledge graph

---

### 📊 Benchmarks (04_benchmarks/)
Performance comparisons and measurements.

**`compare_salience_modes.py`** - Compare memory filtering modes
```bash
python examples/04_benchmarks/compare_salience_modes.py
```
Compares three salience (memory filtering) modes:
- **LOCAL**: Sentence-transformers (slow startup, high accuracy)
- **ONLINE**: OpenAI embeddings API (fast startup, API cost)
- **LIGHTWEIGHT**: Keyword-based (instant startup, no embeddings)

Results:
```
LIGHTWEIGHT: ~5s startup   | No API cost | Graph-only storage
ONLINE:      ~5s startup   | Small API cost | Full vector + graph
LOCAL:       ~10s startup  | No API cost | Full vector + graph
```

---

### 🔌 Providers (05_providers/)
Provider-specific examples for each supported LLM.

**`openai_example.py`** - OpenAI/GPT integration
```bash
export OPENAI_API_KEY='sk-...'
python examples/05_providers/openai_example.py
```

**`claude_example.py`** - Anthropic Claude integration
```bash
export ANTHROPIC_API_KEY='sk-ant-...'
python examples/05_providers/claude_example.py
```

**`gemini_example.py`** - Google Gemini integration
```bash
export GOOGLE_API_KEY='...'
python examples/05_providers/gemini_example.py
```

**`ollama_example.py`** - Ollama (local) integration
```bash
ollama pull qwen3:1.7b
python examples/05_providers/ollama_example.py
```

See [05_providers/README.md](05_providers/README.md) for detailed provider comparisons.

---

## 🎯 Use Case Guide

### "I want to build a chatbot with memory"
→ Start with `01_basics/getting_started.py`
→ Use default settings (balanced tier, local mode)

### "I need fast responses (<100ms)"
→ Use `02_search_tiers/fast_tier_example.py`
→ Set `salience_mode="online"` for fastest startup

### "I want to find relationships between entities"
→ Use `02_search_tiers/deep_tier_example.py`
→ Wait 3-5s after conversations for graph consolidation

### "I want proactive reminders"
→ Use `03_features/task_reminders.py`
→ Schedule tasks with natural language dates

### "I want to run entirely offline"
→ Use `05_providers/ollama_example.py`
→ Set `salience_mode="local"` (no API calls)

### "I need instant cold starts (serverless)"
→ Use any provider with `salience_mode="online"`
→ Or use `salience_mode="lightweight"` for graph-only
```python
from memlayer.wrappers.openai import OpenAI

client = OpenAI(
    api_key="your-key",
    model="gpt-4.1-mini",
    storage_path="./my_memories",
    user_id="user_123"
)

# Store information
client.chat([
    {"role": "user", "content": "My favorite color is blue"}
])

# Retrieve information (automatic)
response = client.chat([
    {"role": "user", "content": "What's my favorite color?"}
])
```

### 2. Explicit Search Tier Control
```python
# Fast search
response = client.chat([
    {"role": "user", "content": "Quick question: What's my name?"}
])

# Deep search with graph traversal
response = client.chat([
    {"role": "user", "content": "Tell me everything about my work. Use deep search."}
])
```

### 3. Multiple Providers
```python
# OpenAI
from memlayer.wrappers.openai import OpenAI
client = OpenAI(api_key="...", model="gpt-4.1-mini")

# Claude
from memlayer.wrappers.claude import Claude
client = Claude(api_key="...", model="claude-3-5-sonnet-20241022")

# Gemini
from memlayer.wrappers.gemini import Gemini
client = Gemini(api_key="...", model="gemini-2.0-flash-exp")

# Ollama (local)
from memlayer.wrappers.ollama import Ollama
client = Ollama(host="http://localhost:11434", model="qwen3:1.7b")
```

## 🔍 Search Tier Selection Guide

| Scenario | Recommended Tier | Reason |
|----------|-----------------|---------|
| Chatbot responses | Fast | Low latency required |
| Simple factual recall | Fast | Few memories needed |
| General conversation | Balanced | Good accuracy/speed balance |
| Research queries | Deep | Need comprehensive results |
| Finding connections | Deep | Graph traversal required |
| "Tell me everything about X" | Deep | Multi-source synthesis |

## 📊 Performance Characteristics

Based on typical queries:

```
Fast:     ~50-150ms   (2 vector results)
Balanced: ~200-600ms  (5 vector results)
Deep:     ~800-2500ms (10 vector results + graph traversal)
```

## 🧠 How Deep Search Works

1. **Vector Search**: Retrieves top 10 semantically similar memories
2. **Entity Extraction**: LLM extracts key entities from the query
   - Example: "Tell me about Alice" → ["Alice"]
3. **Graph Traversal**: For each entity, traverse 1 hop in the knowledge graph
   - Finds relationships: "Alice --[works on]--> Project Phoenix"
4. **Combination**: Merges vector results with graph relationships
5. **Synthesis**: LLM creates comprehensive answer from all sources

## 🛠️ Common Patterns

### Pattern 1: Progressive Memory Building
```python
# Day 1: Store basic info
client.chat([{"role": "user", "content": "I'm working on Project X"}])

# Day 2: Add details
client.chat([{"role": "user", "content": "Project X uses Python and React"}])

# Day 3: Query everything
client.chat([{"role": "user", "content": "What do you know about my projects?"}])
```

### Pattern 2: Entity-Centric Queries
```python
# Store interconnected data
client.chat([{"role": "user", "content": "Alice leads Project Phoenix"}])
client.chat([{"role": "user", "content": "Project Phoenix is in London"}])

# Query with deep search for relationships
client.chat([{
    "role": "user", 
    "content": "Tell me about Alice (use deep search)"
}])
# Response includes: Alice's role, project, location via graph
```

### Pattern 3: Observability
```python
response = client.chat(messages)

# Inspect search performance
if client.last_trace:
    for event in client.last_trace.events:
        print(f"{event.event_type}: {event.duration_ms}ms")
        print(f"Metadata: {event.metadata}")
```

## 📝 Notes

- **Background Consolidation**: Knowledge graph building happens in a background thread. Wait a few seconds after conversations for graph to populate.
- **First Run**: Initial runs may not show graph relationships. Run examples twice to see full deep search capabilities.
- **Storage**: Each example creates its own memory directory to avoid conflicts.
- **LLM Auto-Selection**: The LLM often chooses the appropriate search tier automatically based on query complexity.

## 🔗 Related Documentation

- [Hybrid Search Implementation](../HYBRID_SEARCH_IMPLEMENTATION.md) - Technical details
- [Main README](../README.md) - Project overview

## 💡 Tips

1. **Use Fast tier** for high-traffic applications where latency matters
2. **Use Balanced tier** as your default (already is the default)
3. **Use Deep tier** when you need comprehensive answers with relationship reasoning
4. **Explicit tier requests** work: "use deep search" in your query
5. **Check traces** to understand what search operations were performed
6. **Wait for consolidation** before querying stored information (2-5 seconds)

## 🐛 Troubleshooting

**Q: Deep search doesn't show graph relationships**
- A: Wait longer for background consolidation (try 5 seconds)
- A: Run the example twice - first run builds the graph

**Q: Import is slow**
- A: First import loads models (~0.7s). Subsequent imports are cached.

**Q: No memories found**
- A: Ensure you waited for consolidation after storing information
- A: Check that `storage_path` directory was created

**Q: API errors**
- A: Verify your API key is set correctly
- A: Check you have API credits/quota remaining
