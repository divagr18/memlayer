# ChromaDB (Vector Storage)

Notes and gotchas when using ChromaDB as the vector store:

Metadata types
--------------
- ChromaDB accepts only simple types in metadata: `str`, `int`, `float`, `bool`.
- The code filters out `None` values before saving to Chroma. If you need `null`, store a sentinel string like `"<null>"`.

Windows file locks
------------------
- On Windows, SQLite-based backends can keep file handles open. To avoid "PermissionError" during cleanup:
  - Call `client.close()` to stop background services and close the vector client before deleting files.
  - If you see `PermissionError`, add a short `time.sleep(2)` and `gc.collect()` before retrying.

Performance tuning
------------------
- Set `operation_mode=online` to use managed embeddings for faster startup.
- Tune `n_results` and `search_tier` to control how many vectors are retrieved.

Backup and restore
------------------
- Chroma stores its DB files in the `storage_path/chroma` directory. Back up this folder to preserve vectors.

» See also: `docs/storage/networkx.md`