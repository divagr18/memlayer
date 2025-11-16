# MemLayer Modes

MemLayer supports three operating modes, each optimized for different use cases.

**Key Difference:** These modes control **both salience filtering AND storage architecture**.

## 🖥️ LOCAL Mode (Default)

**Best for:** High-volume applications, offline usage, no ongoing costs

Uses local sentence-transformers models for both salience filtering and vector embeddings.

```python
from memlayer.wrappers.openai import OpenAI

client = OpenAI(
    storage_path="./memories",
    user_id="user123",
    salience_mode="local"  # Default
)
```

**Characteristics:**
- ✅ High accuracy with semantic understanding
- ✅ No API costs after initial setup
- ✅ Works completely offline
- ✅ Shared model across components (optimized)
- ✅ Full semantic vector search
- ❌ Slow startup (~7-8s model loading)
- ❌ Requires ~500MB disk space for model

**Storage:** Vector (ChromaDB) + Graph (NetworkX)
**Startup Time:** ~8 seconds (first use)
**Per-Check Cost:** $0 (free)
**Search Quality:** High (semantic similarity)

---

## ☁️ ONLINE Mode

**Best for:** Production apps, serverless functions, fast cold starts

Uses OpenAI's embeddings API for both salience filtering and vector embeddings.

```python
import os

client = OpenAI(
    storage_path="./memories",
    user_id="user123",
    salience_mode="online",
    api_key=os.getenv("OPENAI_API_KEY")  # Required
)
```

**Characteristics:**
- ✅ Fast startup (~2-3s, no model loading)
- ✅ No local model storage needed
- ✅ Always up-to-date embeddings
- ✅ Scales to serverless/edge environments
- ✅ Full semantic vector search
- ❌ API cost per operation (~$0.0001-0.0002)
- ❌ Requires internet connection
- ❌ Depends on OpenAI API availability

**Storage:** Vector (ChromaDB) + Graph (NetworkX)
**Startup Time:** ~2 seconds
**Per-Check Cost:** ~$0.0001 salience + ~$0.0001 storage (0.02¢ total)
**Search Quality:** High (semantic similarity)

**Cost Estimate:**
- 10,000 operations/month = ~$2.00
- 100,000 operations/month = ~$20.00

---

## 🚀 LIGHTWEIGHT Mode

**Best for:** Prototyping, resource-constrained environments, maximum speed

Uses keyword matching for salience and **graph-only storage** (no embeddings at all).

```python
client = OpenAI(
    storage_path="./memories",
    user_id="user123",
    salience_mode="lightweight"
)
```

**Characteristics:**
- ✅ Instant startup (< 1s)
- ✅ No dependencies (no ML models)
- ✅ No API costs
- ✅ Minimal memory footprint
- ✅ Perfect for rapid prototyping
- ✅ Graph-based memory retrieval
- ❌ No semantic search (keyword/graph only)
- ❌ Lower accuracy (rule-based salience)
- ❌ May miss nuanced content

**Storage:** Graph-only (NetworkX) - **no vector storage**
**Startup Time:** < 1 second
**Per-Check Cost:** $0 (free)
**Search Quality:** Medium (graph traversal + keywords)

---

## Comparison Table

| Feature | LOCAL | ONLINE | LIGHTWEIGHT |
|---------|-------|--------|-------------|
| **Startup Time** | ~8s | ~2s | <1s |
| **Per-Operation Cost** | $0 | ~$0.0002 | $0 |
| **Salience Method** | Semantic (local) | Semantic (API) | Keywords |
| **Storage Type** | Vector + Graph | Vector + Graph | **Graph only** |
| **Search Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Offline Support** | ✅ Yes | ❌ No | ✅ Yes |
| **Disk Space** | ~500MB | ~0MB | ~0MB |
| **Dependencies** | sentence-transformers | openai | None |
| **Best For** | High-volume | Production | Prototyping |

---

## When to Use Each Mode

### Use **LOCAL** when:
- Running long-lived applications (servers, desktop apps)
- Processing high volumes (>100k checks/month)
- Need offline operation
- Startup time doesn't matter
- Want zero ongoing costs

### Use **ONLINE** when:
- Deploying to serverless (Lambda, Cloud Functions)
- Need fast cold starts
- Running on edge/mobile environments
- Volume is moderate (<100k checks/month)
- API cost is acceptable

### Use **LIGHTWEIGHT** when:
- Rapid prototyping and testing
- Extremely resource-constrained environments
- Maximum speed is critical
- Accuracy requirements are relaxed
- No internet connectivity

---

## Benchmarking

Run the comparison script to see performance on your hardware:

```bash
python examples/compare_salience_modes.py
```

Example output:
```
Mode             Init Time       First Chat      Total First Use
----------------------------------------------------------------------
LIGHTWEIGHT        0.234s         2.156s           2.390s
ONLINE             1.892s         2.301s           4.193s
LOCAL             11.234s         2.189s          13.423s
```

---

## Advanced Configuration

### Combining with Custom Thresholds

```python
# Strict LIGHTWEIGHT (only obvious facts)
client = OpenAI(
    salience_mode="lightweight",
    salience_threshold=0.2  # Higher = stricter
)

# Permissive ONLINE (save most content)
client = OpenAI(
    salience_mode="online",
    salience_threshold=-0.05  # Lower = more permissive
)
```

### Mode-Specific Tips

**LOCAL Mode:**
- Share `embedding_model` between clients for faster multi-client init
- Model caching saves ~11s when creating multiple clients in same process

**ONLINE Mode:**
- Prototype embeddings are cached at init time (~2s one-time cost)
- Each salience check makes 1 API call (~$0.0001)

**LIGHTWEIGHT Mode:**
- Customize keywords by editing `SALIENT_KEYWORDS` and `NON_SALIENT_KEYWORDS` in `ml_gate.py`
- Adjust threshold to control sensitivity

---

## Implementation Details

All three modes share the same two-stage filtering:

1. **Fast Heuristic Filter** (< 1ms)
   - Regex pattern matching
   - Catches obvious salient/non-salient content
   - Same across all modes

2. **Semantic/Keyword Check** (mode-specific)
   - **LOCAL:** Sentence-transformer embeddings + cosine similarity
   - **ONLINE:** OpenAI embeddings + cosine similarity  
   - **LIGHTWEIGHT:** TF-IDF keyword matching

---

## Migration Guide

### From LOCAL to ONLINE

```python
# Before
client = OpenAI(salience_mode="local")

# After
client = OpenAI(
    salience_mode="online",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

**Benefit:** 10s faster startup, scales to serverless
**Cost:** ~$0.0001 per salience check

### From LOCAL to LIGHTWEIGHT

```python
# Before
client = OpenAI(salience_mode="local")

# After
client = OpenAI(salience_mode="lightweight")
```

**Benefit:** 11s faster startup, no dependencies
**Trade-off:** ~5-10% lower accuracy on edge cases

---

## FAQ

**Q: Can I switch modes after initialization?**
A: No, mode is set during `__init__()`. Create a new client to change modes.

**Q: Which mode is most cost-effective?**
A: LOCAL for >100k checks/month, ONLINE for <100k, LIGHTWEIGHT for prototyping.

**Q: Does ONLINE mode require OpenAI API key?**
A: Yes, it uses OpenAI's embeddings API. Set `OPENAI_API_KEY` environment variable.

**Q: Can I use ONLINE mode with other LLM providers?**
A: Currently only OpenAI embeddings are supported for ONLINE mode. Use LOCAL or LIGHTWEIGHT with other providers.

**Q: How accurate is LIGHTWEIGHT mode?**
A: ~80-90% of LOCAL/ONLINE accuracy on typical conversations. Lower on nuanced content.

---

## Next Steps

- Try all three modes with `examples/compare_salience_modes.py`
- Read the [Performance Guide](PERFORMANCE.md) for optimization tips
- Check [Examples](../examples/) for usage patterns
