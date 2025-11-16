# PyPI Release Checklist for MemLayer

This checklist covers the essential steps to prepare and publish the package to PyPI.

## Documentation

- [x] `README.md` — clear project overview, quickstart, examples (UPDATED)
- [x] `docs/` — user-facing docs (tuning, services, storage, providers) (COMPLETE)

## Packaging & PyPI

- [x] Ensure `pyproject.toml` has correct metadata (name, version, classifiers, readme, license)
- [x] Add proper build-system configuration (`setuptools>=61.0`, `wheel`)
- [x] Add `MANIFEST.in` to include docs and examples in the distribution
- [x] Create `LICENSE` file (MIT)
- [x] Add `py.typed` marker for type hint support
- [ ] Install build tools: `pip install build twine`
- [ ] Build distribution packages: `python -m build`
  - This creates `dist/memlayer-0.1.0.tar.gz` and `dist/memlayer-0.1.0-py3-none-any.whl`
- [ ] Test the build locally in a clean venv:
  ```bash
  python -m venv test_env
  test_env\Scripts\activate  # Windows
  pip install dist/memlayer-0.1.0-py3-none-any.whl
  python -c "from memory_bank import MemoryClient; print('Success!')"
  ```
- [ ] Check package with twine: `twine check dist/*`
- [ ] Test upload to Test PyPI: `twine upload --repository testpypi dist/*`
- [ ] Install from Test PyPI and validate: `pip install --index-url https://test.pypi.org/simple/ memlayer`
- [ ] Upload to PyPI: `twine upload dist/*`
- [ ] Verify package on PyPI: https://pypi.org/project/memlayer/
- [ ] Test install from PyPI: `pip install memlayer`

## Versioning & Git Tagging

- [ ] Update version in `pyproject.toml` for each release
- [ ] Follow semantic versioning (MAJOR.MINOR.PATCH)
- [ ] Create annotated Git tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
- [ ] Push tags to GitHub: `git push origin v0.1.0`
- [ ] Consider GPG signing tags: `git tag -s v0.1.0 -m "Release v0.1.0"`

## Changelog & Release Notes

- [ ] Create `CHANGELOG.md` with format:
  ```
  # Changelog
  
  ## [Unreleased]
  
  ## [0.1.0] - 2025-XX-XX
  ### Added
  - Initial release
  - Support for OpenAI, Claude, Gemini, Ollama
  - Hybrid vector + knowledge graph storage
  - Three search tiers (fast/balanced/deep)
  - Operation modes (local/online/lightweight)
  ```
- [ ] Update changelog before each release
- [ ] Create GitHub Release from tag with changelog excerpt

## Security & Governance

- [x] Add maintainers list and contact method (in README)

## Tests & Examples

- [ ] Ensure `examples/` are runnable and have minimal required env var checks
- [ ] Test at least one example script to verify functionality

## Final Release Steps

- [ ] Bump version in `pyproject.toml`
- [ ] Create Git tag for release: `git tag -a v0.1.0 -m "Release v0.1.0"`
- [ ] Push tags to GitHub: `git push origin v0.1.0`
- [ ] Create GitHub Release with release notes
- [ ] Publish to PyPI: `twine upload dist/*`
- [ ] Verify installation: `pip install memlayer`

---

**Notes:**
- Start with Test PyPI first to validate everything works
- Ensure all API keys in examples use environment variables (already done via `.env.example`)
