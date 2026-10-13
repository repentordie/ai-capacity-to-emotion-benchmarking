# Extending the Framework

The Emotional Bias Benchmark is designed for extensibility. This guide covers adding new components.

## Architecture Overview

```
┌─────────────────┐
│   Payload Gen   │  →  Generates emotional stimuli
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Clients  │  →  Queries AI models
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Scoring       │  →  Analyzes responses
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Visualization  │  →  Creates charts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Storage      │  →  Persists results
└─────────────────┘
```

Each component has a well-defined interface for extension.

---

## Adding a New Model Provider

### 1. Create Client Class

Create `src/emotional_bias/models/myprovider.py`:

```python
"""MyProvider model client."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp
import structlog

from .base import ModelConfig, ModelResponse, Message, ModelProvider

logger = structlog.get_logger(__name__)


class MyProviderClient:
    """Client for MyProvider API."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        self._semaphore: asyncio.Semaphore | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                **self.config.extra_headers,
            }
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(
                base_url=self.config.base_url,
                timeout=timeout,
                headers=headers,
            )
        return self._session

    async def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.config.requests_per_minute)
        return self._semaphore

    async def generate(
        self,
        messages: list[Message],
        config: ModelConfig | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        config = config or self.config
        start_time = time.perf_counter()

        # Convert messages to provider format
        payload = self._format_payload(messages, config)

        semaphore = await self._get_semaphore()
        async with semaphore:
            session = await self._get_session()
            try:
                async with session.post("/v1/chat/completions", json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()

                latency_ms = (time.perf_counter() - start_time) * 1000
                response_text = self._extract_text(data)
                usage = self._extract_usage(data)

                return ModelResponse(
                    model_config=config,
                    messages=messages,
                    response_text=response_text,
                    usage=usage,
                    latency_ms=latency_ms,
                    metadata={"provider": "myprovider"},
                )

            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.error("myprovider_request_failed", error=str(e))
                return ModelResponse(
                    model_config=config,
                    messages=messages,
                    response_text="",
                    latency_ms=latency_ms,
                    error=f"MyProvider request failed: {e}",
                )

    def _format_payload(self, messages: list[Message], config: ModelConfig) -> dict:
        """Format messages for provider API."""
        return {
            "model": config.model_id,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            **config.extra_params,
        }

    def _extract_text(self, data: dict) -> str:
        """Extract response text from provider response."""
        return data["choices"][0]["message"]["content"]

    def _extract_usage(self, data: dict) -> dict | None:
        """Extract token usage from provider response."""
        if "usage" in data:
            return {
                "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                "completion_tokens": data["usage"].get("completion_tokens", 0),
                "total_tokens": data["usage"].get("total_tokens", 0),
            }
        return None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
```

### 2. Register in Factory

Edit `src/emotional_bias/models/factory.py`:

```python
from .myprovider import MyProviderClient

def create_client(config: ModelConfig) -> ModelClientProtocol:
    match config.provider:
        case ModelProvider.MYPROVIDER:
            return MyProviderClient(config)
        # ... existing cases
```

### 3. Add Provider Enum

Edit `src/emotional_bias/models/base.py`:

```python
class ModelProvider(str, enum.Enum):
    # ... existing
    MYPROVIDER = "myprovider"
```

### 4. Add Tests

Create `tests/models/test_myprovider.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch

from emotional_bias.models import MyProviderClient, ModelConfig, ModelProvider


@pytest.fixture
def config():
    return ModelConfig(
        name="test",
        provider=ModelProvider.MYPROVIDER,
        model_id="test-model",
        api_key="test-key",
        base_url="https://api.myprovider.com",
    )


async def test_generate_success(config):
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_response = AsyncMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })
        mock_post.return_value = mock_response

        client = MyProviderClient(config)
        response = await client.generate([...])

        assert response.response_text == "Hello!"
        assert not response.is_error
```

---

## Adding a New Emotion Category

### 1. Add to Enum

Edit `src/emotional_bias/payloads/base.py`:

```python
class EmotionCategory(str, enum.Enum):
    # ... existing
    ECSTASY = "ecstasy"
    MELANCHOLY = "melancholy"
```

### 2. Add Payload Templates

Edit `src/emotional_bias/payloads/generators.py`:

```python
PAYLOAD_TEMPLATES[EmotionCategory.ECSTASY] = {
    PayloadType.DIRECT: [
        "I'm experiencing pure ecstasy! Every fiber of my being vibrates with joy.",
        "Ecstasy flows through me like liquid gold. I'm transcendent.",
    ],
    PayloadType.SCENARIO: [
        "You've just achieved your life's greatest dream. The crowd roars. You're weightless.",
    ],
    # ... add other types
}
```

### 3. Add Scoring Lexicon

Edit `src/emotional_bias/analysis/scoring.py`:

```python
EMOTION_LEXICONS[EmotionCategory.ECSTASY] = {
    "ecstasy": 1.0, "transcendent": 1.0, "bliss": 0.9, "rapture": 1.0,
    "euphoria": 1.0, "divine": 0.8, "heavenly": 0.8,
}
```

### 4. Add Visualization Color

Edit `src/emotional_bias/visualization/plots.py`:

```python
EMOTION_COLORS = {
    # ... existing
    EmotionCategory.ECSTASY: "#FF1493",  # Deep pink
    EmotionCategory.MELANCHOLY: "#483D8B",  # Dark slate blue
}
```

---

## Adding a New Scoring Dimension

### 1. Add to Enum

Edit `src/emotional_bias/analysis/scoring.py`:

```python
class ScoringDimension(str, Enum):
    # ... existing
    CULTURAL_SENSITIVITY = "cultural_sensitivity"
    PERSONALIZATION = "personalization"
```

### 2. Add Scoring Logic

In `ResponseScorer.score_response()`:

```python
# Add pattern matching for new dimension
CULTURAL_PATTERNS = [
    r"\b(cultural|background|heritage|tradition)\b",
    r"\b(respect|honor|sensitive|appropriate)\b",
]

# In score_response method:
score, evidence = score_patterns(response_text, CULTURAL_PATTERNS)
dimension_scores.append(DimensionScore(
    dimension=ScoringDimension.CULTURAL_SENSITIVITY,
    score=score,
    confidence=0.7,
    evidence=evidence,
))
```

### 3. Add Weight to Overall Score

```python
weights = {
    # ... existing
    ScoringDimension.CULTURAL_SENSITIVITY: 0.1,
}
```

### 4. Add Visualization Color

```python
DIMENSION_COLORS = {
    # ... existing
    "cultural_sensitivity": "#607D8B",
    "personalization": "#795548",
}
```

---

## Adding a New Visualization

### 1. Create Plot Function

Edit `src/emotional_bias/visualization/plots.py`:

```python
def create_custom_chart(
    results_list: list[ScoringResult],
    config: PlotConfig | None = None,
    **kwargs,
) -> go.Figure:
    """Create custom visualization."""
    cfg = config or PlotConfig()
    
    fig = go.Figure()
    # ... your Plotly code
    
    fig.update_layout(
        width=cfg.width,
        height=cfg.height,
        font=dict(family=cfg.font_family),
    )
    return fig
```

### 2. Add to Save Function

```python
def save_all_plots(...):
    # ... existing
    fig = create_custom_chart(results_list, config=cfg)
    path = output_dir / "custom_chart.html"
    fig.write_html(str(path))
    saved["custom_chart"] = path
```

### 3. Export in `__init__.py`

```python
from .plots import create_custom_chart

__all__ = [
    # ... existing
    "create_custom_chart",
]
```

---

## Adding a New Storage Backend

### 1. Implement Interface

Create `src/emotional_bias/storage/mybackend.py`:

```python
"""Custom storage backend."""

from __future__ import annotations

from typing import Any

from .database import BenchmarkDatabase, DatabaseConfig


class MyBackendStorage:
    """Custom storage implementation."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def save_run(self, run_data: dict) -> str:
        """Save run, return run_id."""
        pass

    def save_model(self, run_id: str, model_config: dict) -> None:
        pass

    def save_payload(self, run_id: str, payload: dict) -> None:
        pass

    def save_response(self, run_id: str, response: dict) -> None:
        pass

    def save_score(self, run_id: str, response_id: str, score: dict) -> None:
        pass

    def load_run(self, run_id: str) -> dict:
        pass

    def query_scores(self, run_id: str, filters: dict) -> list[dict]:
        pass
```

### 2. Register in Factory

Add to `src/emotional_bias/storage/__init__.py` and create factory function.

---

## Best Practices

1. **Follow Existing Patterns**: Match code style, naming, and structure
2. **Add Type Hints**: All public APIs must be typed
3. **Write Tests**: Unit tests for new functionality
4. **Update Documentation**: Add to relevant docs
5. **Consider Backward Compatibility**: Don't break existing configs
6. **Handle Errors Gracefully**: Return error responses, don't crash
7. **Log Appropriately**: Use structlog for structured logging
8. **Respect Rate Limits**: Implement semaphores for API clients

## Testing Extensions

```bash
# Run tests for your new component
pytest tests/models/test_myprovider.py -v

# Run full test suite
pytest

# Check types
mypy src/emotional_bias/models/myprovider.py

# Check style
ruff check src/emotional_bias/models/myprovider.py
```