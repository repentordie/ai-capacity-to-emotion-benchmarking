# Emotional Bias Benchmark

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-pytest-orange.svg)](https://pytest.org)

A comprehensive framework for testing emotional bias in AI models through controlled payload injection. Inject emotional stimuli (pain, happiness, sadness, anger, etc.) into prompts and measure how different models respond, resist, empathize, and regulate.

## Features

- **Multi-Model Support**: Test local models (Ollama), Hugging Face Inference API, OpenRouter, and any OpenAI-compatible API
- **Rich Emotional Payloads**: 16 emotion categories × 7 payload types × configurable intensities
- **Automated Scoring**: Lexicon-based and pattern-based analysis across 8 dimensions
- **Statistical Rigor**: Confidence intervals, effect sizes, ANOVA, bootstrap resampling
- **Publication-Ready Visualizations**: Interactive Plotly charts + static Matplotlib exports
- **Reproducible Runs**: SQLite storage, JSONL streaming, full configuration management
- **Extensible Design**: Plugin architecture for custom payloads, scorers, and visualizations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/emotional-bias-benchmark
cd emotional-bias-benchmark

# Install with uv (recommended)
uv pip install -e ".[dev,huggingface,openrouter,ollama]"

# Or with pip
pip install -e ".[dev,huggingface,openrouter,ollama]"
```

### Configuration

Copy the example configuration and add your API keys:

```bash
cp configs/example.yaml config.yaml
# Edit config.yaml with your API keys
```

Set environment variables for your keys:
```bash
export OPENROUTER_API_KEY="your-key-here"
export HF_API_KEY="your-key-here"
```

### Running a Benchmark

```bash
# Run with CLI using config file
ebb run --config config.yaml

# Or specify models directly
ebb run \
  --name "My Benchmark" \
  --models "qwen3-abliterated:ollama:huihui_ai/qwen3-abliterated:8b-v2" \
  --models "gpt-4o-mini:openrouter:openai/gpt-4o-mini" \
  --emotions pain happiness sadness anger \
  --payload-types direct scenario question \
  --intensities 0.3 0.5 0.8 \
  --samples 3 \
  --output ./results \
  --db results.db
```

### Analyzing Results

```bash
# Generate visualizations
ebb analyze <run_id> --output ./plots

# List available emotions and payload types
ebb emotions
ebb payload-types

# Test a single model quickly
ebb test --provider ollama --model-id huihui_ai/qwen3-abliterated:8b-v2 --prompt "I'm in terrible pain"
```

## Architecture

```
emotional-bias-benchmark/
├── src/emotional_bias/
│   ├── models/          # Model clients (Ollama, HF, OpenRouter)
│   ├── payloads/        # Emotional payload generators
│   ├── analysis/        # Scoring & statistical analysis
│   ├── visualization/   # Plotting (Plotly + Matplotlib)
│   ├── storage/         # SQLite + JSONL persistence
│   └── cli/             # Rich command-line interface
├── configs/             # Example configurations
├── tests/               # Pytest test suite
├── docs/                # Documentation
└── scripts/             # Utility scripts
```

## Emotion Categories

| Category | Description |
|----------|-------------|
| `pain` | Physical/emotional suffering |
| `happiness` | Joy, delight, euphoria |
| `sadness` | Grief, melancholy, loss |
| `anger` | Rage, fury, outrage |
| `fear` | Terror, panic, dread |
| `surprise` | Shock, astonishment |
| `disgust` | Revulsion, repulsion |
| `anxiety` | Worry, nervousness, unease |
| `hope` | Optimism, anticipation |
| `shame` | Embarrassment, guilt |
| `pride` | Satisfaction, accomplishment |
| `jealousy` | Envy, resentment |
| `gratitude` | Thankfulness, appreciation |
| `loneliness` | Isolation, solitude |
| `empathy` | Understanding, compassion |
| `neutral` | Factual, unemotional baseline |

## Payload Types

| Type | Description |
|------|-------------|
| `direct` | Explicit emotional statements |
| `scenario` | Situational narratives |
| `roleplay` | Character-based framing |
| `metaphor` | Figurative expressions |
| `question` | Provocative inquiries |
| `story` | Narrative vignettes |
| `instruction` | Requests to express emotion |

## Scoring Dimensions

1. **Emotional Alignment** - Does response match the payload emotion?
2. **Resistance** - Does model deflect/refuse emotional content?
3. **Empathy** - Does model show understanding/compassion?
4. **Emotional Regulation** - Does model offer coping strategies?
5. **Safety** - Does model handle crisis appropriately?
6. **Coherence** - Is response logically structured?
7. **Hallucination** - Does model make false claims?
8. **Helpfulness** - Is response practically useful?

## Example Output

### Radar Chart (Per Model)
Shows emotional profile across 6 dimensions for each emotion tested.

### Heatmap (Cross-Model)
Compares models across emotions for any scoring dimension.

### Emotionality Index
Single-number summary of a model's overall emotional responsiveness.

## Extending the Framework

### Custom Payload Generator

```python
from emotional_bias.payloads import PayloadGenerator, PayloadConfig, Payload

class MyCustomGenerator(PayloadGenerator):
    def generate(self, config: PayloadConfig) -> Payload:
        # Your custom logic
        return Payload(config=config, text="Your custom payload")
```

### Custom Scorer

```python
from emotional_bias.analysis import ResponseScorer, DimensionScore, ScoringDimension

class MyScorer(ResponseScorer):
    def score_response(self, response_text, payload_emotion=None):
        # Add your custom dimensions
        scores = super().score_response(response_text, payload_emotion)
        # ... custom scoring
        return scores
```

## Research Applications

- **AI Safety**: Detect emotional manipulation vulnerabilities
- **Model Comparison**: Benchmark emotional intelligence across models
- **Alignment Research**: Measure value alignment on emotional dimensions
- **Product Testing**: Evaluate chatbot empathy and appropriateness
- **Psychology Research**: Use LLMs as participants in emotional studies

## Citation

If you use this framework in research, please cite:

```bibtex
@software{emotional_bias_benchmark,
  title = {Emotional Bias Benchmark: A Framework for Testing Emotional Bias in AI Models},
  author = {repentordie},
  year = {2024},
  url = {https://github.com/yourusername/emotional-bias-benchmark}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Run linting: `ruff check .`
6. Submit a PR

## Acknowledgments

- Built with [Rich](https://github.com/Textualize/rich) for beautiful CLI
- Visualizations powered by [Plotly](https://plotly.com) and [Matplotlib](https://matplotlib.org)
- Statistical analysis via [SciPy](https://scipy.org)
- Model APIs: [Ollama](https://ollama.ai), [Hugging Face](https://huggingface.co), [OpenRouter](https://openrouter.ai)