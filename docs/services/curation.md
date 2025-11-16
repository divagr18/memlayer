# Curation Service

The `CurationService` periodically evaluates stored memories and performs lifecycle actions:

- Archive low-relevance memories (moves them to an archived status)
- Delete expired memories (per the `expiration_date` on facts)

How it works
------------
- Runs every `curation_interval_seconds` (configurable)
- Uses hybrid relevance scoring (vector similarity, access recency, importance score)
- Archives when relevance < `archive_threshold` (default ~0.3)
- Deletes when current time > `expiration_date`

Programmatic control
--------------------
- Access via `client.curation_service` (property will start it if not running)
- `client.curation_service.start()` — explicitly start
- `client.curation_service.stop()` — stop cleanly

Practical tips
--------------
- For tests, use `curation_interval_seconds=10` to speed cycles up.
- Call `client.close()` in your application exit path to ensure curation stops before storage is closed (prevents file locks).

» See also: `docs/tuning/intervals.md`