# Scheduler and Curation Intervals

MemLayer exposes two interval knobs to control background automation:

- `scheduler_interval_seconds` (default: 60)
  - Controls how often the scheduler wakes up to check for due tasks/reminders.
  - Lower values mean more responsive reminders but higher CPU wakeups.

- `curation_interval_seconds` (default: 3600)
  - Controls how often the `CurationService` evaluates memories for archiving and deletion.
  - Lower values let low-importance memories be archived/deleted sooner; higher values reduce background I/O.

How to configure
----------------
```python
client = OpenAI(
    scheduler_interval_seconds=30,
    curation_interval_seconds=600
)
```

Practical guidance
------------------
- For local development: `curation_interval_seconds=10` is useful for tests.
- For production: `curation_interval_seconds` between 900 (15 minutes) and 3600 (1 hour) is reasonable.
- If your storage backend shows file locks (Windows + Chroma), increase `curation_interval_seconds` and ensure `close()` is called during shutdown.

Monitoring
----------
- Use `client.last_trace` and logging to observe curation cycles.
- The CurationService prints debug logs when archiving/deleting memories; watch these logs to validate settings.

» See also: `docs/services/curation.md`