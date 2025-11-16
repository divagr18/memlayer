# Claude (Anthropic) Provider Notes

Credentials
-----------
- Set `ANTHROPIC_API_KEY` in your environment or pass `api_key` to `Claude(...)`.
- If you don't have a key, the client can still be used for local-only features (consolidation via other provider clients).

Notes
-----
- Claude's SDK requires `api_key` or `auth_token`. If neither is set, calls to `client.messages.create` will fail.
- You can use OpenAI embeddings in `operation_mode=online` even when using Claude as the chat provider.

» See also: `docs/tuning/operation_mode.md`