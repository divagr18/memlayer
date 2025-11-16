# Ollama (Local LLM) Provider Notes

Overview
--------
- Ollama runs local models via a REST endpoint. Set the `host` when creating the `Ollama` client.

Notes
-----
- Ollama is ideal for fully offline deployments. Use `operation_mode='local'` if you want to keep embeddings offline as well.
- Check that the Ollama server is reachable at `host` before running examples.

» See also: `docs/tuning/operation_mode.md`