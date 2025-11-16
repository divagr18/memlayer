# Tests

This folder contains tests, benchmarks, and debugging scripts for MemLayer.

## 🧪 Test Files

### Unit Tests
- `test_imports.py` - Tests import system and module loading
- `test_entity_extraction.py` - Tests knowledge graph entity extraction
- `test_entity_deduplication.py` - Tests entity deduplication logic
- `test_hybrid_search.py` - Tests vector + graph hybrid search
- `test_traversal.py` - Tests graph traversal algorithms
- `test_substring.py` - Tests substring matching utilities

### Benchmarks
- `benchmark_models.py` - Benchmarks different embedding models
- `benchmark_cache.py` - Tests embedding model caching performance

### Profiling & Debugging
- `profile_performance.py` - Performance profiling for bottleneck identification
- `debug_canonical.py` - Debugging canonical entity resolution
- `debug_graph.py` - Debugging knowledge graph operations

## 🚀 Running Tests

### Individual Tests
```bash
python tests/test_imports.py
python tests/test_hybrid_search.py
```

### Run All Tests (if pytest is configured)
```bash
pytest tests/
```

## 📊 Benchmarks

### Model Comparison
```bash
python tests/benchmark_models.py
```
Compares performance of different embedding models (sentence-transformers variants).

### Cache Performance
```bash
python tests/benchmark_cache.py
```
Tests the performance improvement from model caching.

## 🔍 Profiling

### Performance Analysis
```bash
python tests/profile_performance.py
```
Profiles the entire memory consolidation pipeline to identify bottlenecks.

## 💡 Writing Tests

When adding new features, add corresponding tests:

```python
# tests/test_new_feature.py
def test_new_feature():
    from memlayer import Memory
    
    # Setup
    memory = Memory(storage_path="./test_memory")
    
    # Test
    result = memory.some_new_method()
    
    # Assert
    assert result == expected_value
    
    # Cleanup
    import shutil
    shutil.rmtree("./test_memory")
```

## 🧹 Cleanup

Tests may create temporary storage directories. Clean them up:

```bash
# Remove all test storage directories
rm -rf test_* *_test_memory cache_test_*
```

Or on Windows PowerShell:
```powershell
Remove-Item test_*, *_test_memory, cache_test_* -Recurse -Force
```
