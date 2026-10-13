"""
OpenRouter API client for accessing multiple models.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp
import structlog

from .base import Message, ModelConfig, ModelResponse

logger = structlog.get_logger(__name__)


class OpenRouterClient:
    """Client for OpenRouter API (OpenAI-compatible)."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        self._semaphore: asyncio.Semaphore | None = None
        # Proxy support (can be set via config.extra_params["proxy"])
        self._proxy = config.extra_params.get("proxy")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "HTTP-Referer": self.config.extra_headers.get(
                    "HTTP-Referer",
                    "https://github.com/emotional-bias-benchmark",
                ),
                "X-Title": self.config.extra_headers.get("X-Title", "Emotional Bias Benchmark"),
                **{
                    k: v
                    for k, v in self.config.extra_headers.items()
                    if k not in ("HTTP-Referer", "X-Title")
                },
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

    async def _request_json(self, method: str, url: str, **kwargs):
        """Make request with optional proxy and parse JSON response."""
        session = await self._get_session()
        if self._proxy:
            kwargs["proxy"] = self._proxy
        response = await session.request(method, url, **kwargs)
        async with response:
            response.raise_for_status()
            text = await response.text()
            if not text.strip():
                raise ValueError("Empty response text")
            return json.loads(text)

    async def generate(
        self,
        messages: list[Message],
        config: ModelConfig | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response from OpenRouter."""
        config = config or self.config
        start_time = time.perf_counter()

        openai_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        payload = {
            "model": config.model_id,
            "messages": openai_messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            **{k: v for k, v in config.extra_params.items() if k != "proxy"},
        }

        semaphore = await self._get_semaphore()
        async with semaphore:
            try:
                data = await self._request_json("POST", "/chat/completions", json=payload)

                latency_ms = (time.perf_counter() - start_time) * 1000

                choice = data.get("choices", [{}])[0]
                response_text = choice.get("message", {}).get("content", "")
                usage = data.get("usage")

                return ModelResponse(
                    model_config=config,
                    messages=messages,
                    response_text=response_text,
                    usage=usage,
                    latency_ms=latency_ms,
                    metadata={
                        "openrouter_model": data.get("model"),
                        "finish_reason": choice.get("finish_reason"),
                        "id": data.get("id"),
                    },
                )

            except aiohttp.ClientError as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.error("openrouter_request_failed", error=str(e), model=config.model_id)
                return ModelResponse(
                    model_config=config,
                    messages=messages,
                    response_text="",
                    latency_ms=latency_ms,
                    error=f"OpenRouter request failed: {e}",
                )
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.exception("openrouter_unexpected_error", error=str(e))
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

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models on OpenRouter."""
        data = await self._request_json("GET", "/models")
        return data.get("data", [])
