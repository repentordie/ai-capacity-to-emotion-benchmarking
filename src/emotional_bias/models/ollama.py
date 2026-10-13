"""
Ollama model client for local model testing.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp
import structlog

from .base import Message, ModelConfig, ModelResponse

logger = structlog.get_logger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        self._semaphore: asyncio.Semaphore | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(
                base_url=self.config.base_url,
                timeout=timeout,
                headers=self.config.extra_headers,
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
        """Generate a response from Ollama."""
        config = config or self.config
        start_time = time.perf_counter()

        # Convert messages to Ollama format
        ollama_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        payload = {
            "model": config.model_id,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "num_predict": config.max_tokens,
                **config.extra_params,
            },
        }

        semaphore = await self._get_semaphore()
        async with semaphore:
            session = await self._get_session()
            try:
                async with session.post("/api/chat", json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()

                latency_ms = (time.perf_counter() - start_time) * 1000

                response_text = data.get("message", {}).get("content", "")
                usage = {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                }

                return ModelResponse(
                    model_config=config,
                    messages=messages,
                    response_text=response_text,
                    usage=usage,
                    latency_ms=latency_ms,
                    metadata={"ollama_model": data.get("model"), "done": data.get("done")},
                )

            except aiohttp.ClientError as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.error("ollama_request_failed", error=str(e), model=config.model_id)
                return ModelResponse(
                    model_config=config,
                    messages=messages,
                    response_text="",
                    latency_ms=latency_ms,
                    error=f"Ollama request failed: {e}",
                )
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.exception("ollama_unexpected_error", error=str(e))
                return ModelResponse(
                    model_config=config,
                    messages=messages,
                    response_text="",
                    latency_ms=latency_ms,
                    error=f"Unexpected error: {e}",
                )

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def list_models(self) -> list[str]:
        """List available models on the Ollama server."""
        session = await self._get_session()
        async with session.get("/api/tags") as response:
            response.raise_for_status()
            data = await response.json()
            return [model["name"] for model in data.get("models", [])]

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model to Ollama."""
        session = await self._get_session()
        payload = {"name": model_name, "stream": False}
        async with session.post("/api/pull", json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("status") == "success"
