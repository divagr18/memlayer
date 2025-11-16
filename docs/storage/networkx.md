# NetworkX Knowledge Graph Storage

Overview
--------
- NetworkX is used as an in-memory graph with on-disk persistence (pickle) for the knowledge graph.
- The graph stores nodes for entities and facts, with attributes like `created_timestamp`, `last_accessed_timestamp`, `importance_score`, and `status`.

File format & durability
------------------------
- The graph is persisted to `storage_path/knowledge_graph.pkl` after modifications.
- On load, the code will attempt to detect corruption and back up problematic files to `*.pkl.corrupted`.

Node schema
-----------
- Entity node: `{"name": ..., "type": ..., ...}`
- Fact node: `{"fact": ..., "importance_score": ..., "expiration_date": ...}`

Recovery & backups
------------------
- Regularly back up `knowledge_graph.pkl` for production systems.
- If the graph fails to load, the runtime will create a fresh graph and optionally back up the corrupted file.

» See also: `docs/storage/chroma.md`