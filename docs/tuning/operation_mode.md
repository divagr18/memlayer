# Operation Mode (online | local | lightweight)

Overview
--------
`operation_mode` controls embedding usage, storage composition, and startup behavior. Choose the right mode based on latency, cost, and environment constraints.

Modes
-----
- `online` (recommended for serverless): Uses a managed embeddings API (OpenAI). Fast startup, good semantic search, small API cost.
- `local`: Uses a local sentence-transformer model for embeddings. No API cost, but slower startup and larger memory footprint.
- `lightweight`: No embeddings. Graph-only mode using keyword matching. Fastest startup, lowest resource use, lower recall accuracy.

How to set
----------
```python
from memlayer.wrappers.openai import OpenAI
client = OpenAI(operation_mode="online")
```

Practical tips
--------------
- Serverless: prefer `online` to avoid heavy model loads.
- Offline deployments: use `local` and pin a small transformer model.
- Rapid prototyping / demos: use `lightweight` so examples are instant.

Impact on other components
--------------------------
- Vector storage (`ChromaStorage`) is disabled in `lightweight` mode.
- `salience_gate` may select different embedding strategies depending on the mode.

Examples
--------
- See `examples/06_api/direct_knowledge_ingestion.py` for an `online` example.

» See also: `docs/tuning/salience_threshold.md`, `docs/storage/chroma.md`