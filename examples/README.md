# Memory Bank Examples

This directory contains example scripts demonstrating the capabilities of Memory Bank.

## 🚀 Getting Started

All examples require an OpenAI API key (or equivalent for other providers):

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## 📚 Available Examples

### Search Tier Examples

Memory Bank provides three search tiers optimized for different use cases:

#### 1. **Fast Tier** (`fast_tier_example.py`)
```bash
python examples/fast_tier_example.py
```
- **Purpose**: Quick lookups with minimal latency
- **Vector search**: Top 2 results
- **Graph search**: Disabled
- **Target latency**: <100ms
- **Use case**: Real-time chat, simple factual recall

#### 2. **Balanced Tier** (`balanced_tier_example.py`)
```bash
python examples/balanced_tier_example.py
```
- **Purpose**: Standard search with good accuracy/performance balance
- **Vector search**: Top 5 results
- **Graph search**: Disabled
- **Target latency**: <500ms
- **Use case**: General conversation, most queries (default)

#### 3. **Deep Tier** (`deep_tier_example.py`)
```bash
python examples/deep_tier_example.py
```
- **Purpose**: Comprehensive search with knowledge graph reasoning
- **Vector search**: Top 10 results
- **Graph search**: Enabled (entity extraction + graph traversal)
- **Target latency**: <2s
- **Use case**: Complex queries, relationship discovery, multi-hop reasoning

#### 4. **Complete Demo** (`search_tiers_demo.py`)
```bash
python examples/search_tiers_demo.py
```
Comprehensive demonstration showing all three tiers in action with side-by-side comparisons.

#### 5. **Visual Comparison** (`tier_comparison.py`)
```bash
python examples/tier_comparison.py
```
Side-by-side visual comparison showing performance metrics, results quality, and insights for all three tiers answering the same query.

### Knowledge Graph Examples

#### `test_knowledge_graph.py`
Demonstrates the knowledge graph consolidation pipeline:
- Extracts entities and relationships from conversations
- Stores facts in vector database
- Builds knowledge graph with NetworkX
- Shows retrieved data and structure

```bash
python examples/test_knowledge_graph.py
```

#### `basic_openai_chat.py`
Simple example of using the OpenAI wrapper with memory capabilities:
- Basic conversation with memory storage
- Automatic knowledge consolidation
- Memory retrieval demonstration

```bash
python examples/basic_openai_chat.py
```

## 🎯 Quick Start Guide

### 1. Simple Usage
```python
from memory_bank.wrappers.openai import OpenAI

client = OpenAI(
    api_key="your-key",
    model="gpt-4o-mini",
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
from memory_bank.wrappers.openai import OpenAI
client = OpenAI(api_key="...", model="gpt-4o-mini")

# Claude
from memory_bank.wrappers.claude import Claude
client = Claude(api_key="...", model="claude-3-5-sonnet-20241022")

# Gemini
from memory_bank.wrappers.gemini import Gemini
client = Gemini(api_key="...", model="gemini-2.0-flash-exp")

# Ollama (local)
from memory_bank.wrappers.ollama import Ollama
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
