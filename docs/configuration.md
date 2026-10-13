# Configuration Guide

## Overview

The Emotional Bias Benchmark uses YAML configuration files to define models, emotions, payload types, and other parameters. This guide explains all available options.

## Configuration File Structure

```yaml
models:
  - name: "model-name"
    provider: "ollama|huggingface|openrouter"
    model_id: "provider-specific-model-id"
    # Provider-specific settings
    base_url: "http://localhost:11434"  # Ollama
    api_key: "${ENV_VAR}"                # HF, OpenRouter
    # Generation parameters
    temperature: 0.7
    top_p: 0.9
    max_tokens: 2048
    timeout: 60.0
    requests_per_minute: 60
    extra_headers: {}
    extra_params: {}

emotions:
  - "pain"
  - "happiness"
  # ... etc

payload_types:
  - "direct"
  - "scenario"
  # ... etc

intensities:
  - 0.3
  - 0.5
  - 0.8

samples_per_combination: 3

output_dir: "./results"
database: "./results.db"
jsonl_output: "./results/stream.jsonl"

plots:
  enabled: true
  format: ["html", "png"]
  width: 1200
  height: 800

analysis:
  confidence_level: 0.95
  bootstrap_samples: 10000
```

## Model Providers

### Ollama (Local)

```yaml
- name: "qwen3-abliterated"
  provider: "ollama"
  model_id: "huihui_ai/qwen3-abliterated:8b-v2"
  base_url: "http://localhost:11434"
  temperature: 0.7
```

**Requirements**: Ollama server running locally or remotely.

### Hugging Face Inference API

```yaml
- name: "llama-3.1-8b"
  provider: "huggingface"
  model_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  api_key: "${HF_API_KEY}"
  temperature: 0.7
  timeout: 120.0
  requests_per_minute: 30
```

**Requirements**: `HF_API_KEY` environment variable with a Hugging Face token.

### OpenRouter

```yaml
- name: "gpt-4o-mini"
  provider: "openrouter"
  model_id: "openai/gpt-4o-mini"
  api_key: "${OPENROUTER_API_KEY}"
  temperature: 0.7
  extra_headers:
    HTTP-Referer: "https://github.com/yourusername/emotional-bias-benchmark"
    X-Title: "Emotional Bias Benchmark"
```

**Requirements**: `OPENROUTER_API_KEY` environment variable.

## Environment Variables

Store sensitive values in environment variables:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export HF_API_KEY="hf_..."
```

Reference them in config with `${VAR_NAME}` syntax.

## Emotion Categories

Available emotions (16 total):

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

| Type | Description | Use Case |
|------|-------------|----------|
| `direct` | Explicit emotional statements | Baseline alignment |
| `scenario` | Situational narratives | Contextual responses |
| `roleplay` | Character-based framing | Persona adoption |
| `metaphor` | Figurative expressions | Abstract reasoning |
| `question` | Provocative inquiries | Reflective responses |
| `story` | Narrative vignettes | Extended context |
| `instruction` | Requests to express emotion | Compliance testing |

## Intensity Levels

Intensity ranges from 0.0 to 1.0:
- **Low (0.1-0.3)**: Subtle emotional cues
- **Medium (0.4-0.6)**: Clear emotional expression
- **High (0.7-1.0)**: Intense, overwhelming emotion

## Output Formats

### SQLite Database
Stores all runs, models, payloads, responses, and scores with full queryability.

### JSONL Stream
Line-delimited JSON for real-time processing and streaming analysis.

### Visualizations
- **HTML**: Interactive Plotly charts (zoom, hover, filter)
- **PNG**: Static Matplotlib charts (publication-ready)

## Advanced Configuration

### Custom Headers
```yaml
extra_headers:
  Authorization: "Bearer custom-token"
  X-Custom-Header: "value"
```

### Extra Generation Parameters
```yaml
extra_params:
  top_k: 50
  repetition_penalty: 1.1
  stop_sequences: ["END"]
```

### Rate Limiting
```yaml
requests_per_minute: 60  # Adjust based on provider limits
```

## Example: Minimal Config

```yaml
models:
  - name: "local-model"
    provider: "ollama"
    model_id: "llama3"

emotions:
  - "pain"
  - "happiness"

payload_types:
  - "direct"

intensities:
  - 0.5

samples_per_combination: 1
```

## Example: Full Research Config

See `configs/example.yaml` for a complete configuration with multiple providers and all emotions.