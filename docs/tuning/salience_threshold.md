# Salience Threshold

`salience_threshold` controls how permissive the consolidation pipeline is when deciding whether a fact should become a stored memory.

- Lower (negative) values are permissive and save more content (useful for aggressive recall).
- Higher (positive) values are strict and save less content to minimize storage and noise.

Default: `0.0` (balanced)

How to set
----------
```python
client = OpenAI(salience_threshold=-0.1)  # more permissive
```

Guidance
--------
- For personal assistants where recall breadth is important, set `salience_threshold` to `-0.1` or lower.
- For privacy-sensitive or noisy inputs, set `salience_threshold` to `0.1` or higher.
- Combine with `operation_mode` to tune both accuracy and cost.

» See also: `docs/tuning/operation_mode.md`