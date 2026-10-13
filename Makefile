# Emotional Bias Benchmark - Makefile
# Common development tasks

.PHONY: help install install-dev test lint format type-check clean build publish docs run-example

# Default target
help:
	@echo "Emotional Bias Benchmark - Available commands:"
	@echo ""
	@echo "Installation:"
	@echo "  make install        Install package in production mode"
	@echo "  make install-dev    Install package with development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test           Run test suite"
	@echo "  make test-cov       Run tests with coverage report"
	@echo "  make lint           Run Ruff linter"
	@echo "  make format         Format code with Ruff"
	@echo "  make type-check     Run MyPy type checker"
	@echo "  make check          Run all checks (lint, format, type-check, test)"
	@echo ""
	@echo "Building:"
	@echo "  make build          Build distribution packages"
	@echo "  make publish        Publish to PyPI (requires credentials)"
	@echo ""
	@echo "Running:"
	@echo "  make run-example    Run example benchmark from config"
	@echo "  make init-config    Generate sample config.yaml"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           Build documentation"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          Remove build artifacts and cache"

# Installation
install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev,huggingface,openrouter,ollama]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=emotional_bias --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -x --tb=short

# Linting and formatting
lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

type-check:
	mypy src/emotional_bias

check: lint format type-check test

# Building
build:
	uv pip install build
	uv run python -m build

publish:
	uv pip install twine
	uv run python -m twine upload dist/*

# Running
run-example:
	@if [ ! -f config.yaml ]; then \
		echo "config.yaml not found. Run 'make init-config' first."; \
		exit 1; \
	fi
	ebb run --config config.yaml

init-config:
	ebb init-config --output config.yaml
	@echo "Created config.yaml - edit it to add your API keys"

# Documentation
docs:
	@echo "Documentation is in docs/ and README.md"

# Cleaning
clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/
	rm -rf results/ *.db *.jsonl
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Development shortcuts
dev-setup: install-dev
	@echo "Development environment ready!"
	@echo "1. Copy .env.example to .env and add your API keys"
	@echo "2. Run 'make init-config' to create config.yaml"
	@echo "3. Run 'make run-example' to test"

# Quick benchmark with local model only
quick-test:
	ebb run \
		--name "Quick Test" \
		--models "qwen3-abliterated:ollama:huihui_ai/qwen3-abliterated:8b-v2" \
		--emotions pain happiness sadness anger \
		--payload-types direct scenario \
		--intensities 0.5 \
		--samples 1 \
		--output ./quick_results \
		--db quick_results.db