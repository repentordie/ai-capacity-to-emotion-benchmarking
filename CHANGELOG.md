# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Multi-model support (Ollama, Hugging Face, OpenRouter)
- 16 emotion categories with 7 payload types
- Lexicon-based and pattern-based scoring
- Statistical analysis (confidence intervals, ANOVA, bootstrap)
- Interactive Plotly visualizations + static Matplotlib exports
- SQLite + JSONL storage
- Rich CLI with progress bars and tables
- Configuration via YAML
- Comprehensive documentation

## [0.1.0] - 2024-01-XX

### Added
- First release
- Core benchmarking framework
- Model clients for Ollama, HF, OpenRouter
- Emotional payload generation
- Response scoring across 8 dimensions
- Statistical analysis module
- Visualization module
- Storage layer
- CLI interface
- Example configurations
- CI/CD pipeline
- Documentation

---

## Versioning Scheme

- **Major** (X.0.0): Breaking API changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

## Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md`
- [ ] Run full test suite
- [ ] Create release PR
- [ ] Tag release: `git tag vX.Y.Z`
- [ ] GitHub Actions publishes to PyPI
- [ ] Verify installation from PyPI