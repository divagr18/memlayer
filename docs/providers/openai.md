# OpenAI Provider Notes

Credentials
-----------
- Set `OPENAI_API_KEY` in your environment or pass `api_key` to `OpenAI(...)`.

Embeddings
----------
- Default online embedding model: `text-embedding-3-small`.
- For lower latency on large volumes, consider batching embedding calls in your own pipeline.

Model selection
---------------
- Synthesized answers use `model` configured on the `OpenAI` client. Use smaller models for cheaper, faster results when appropriate.

» See also: `docs/tuning/operation_mode.md`