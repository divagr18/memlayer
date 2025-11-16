# Consolidation Service

The ConsolidationService performs the heavy lifting of turning raw text into stored memories.

Responsibilities
----------------
- Run salience filtering to decide which sentences to keep
- Extract facts, entities, and relationships via LLM calls (`analyze_and_extract_knowledge`)
- Store facts as vector records and entities/relationships in the graph
- Run asynchronously in background worker threads

Programmatic usage
------------------
- `client.consolidation_service.consolidate(text, user_id)` — directly call the consolidation pipeline
- `client.update_from_text(text)` — convenience wrapper that calls the consolidation service

Notes
-----
- Consolidation is non-blocking: it runs in a background thread and returns quickly.
- Ensure `client.close()` is called on shutdown so background threads stop gracefully.

Troubleshooting
---------------
- If entities appear as strings in the graph, ensure `analyze_and_extract_knowledge` returns entities as `[{"name": ..., "type": ...}]`.
- If LLM extraction fails often, check provider credentials and model availability.

» See also: `docs/services/curation.md` and `docs/tuning/salience_threshold.md`