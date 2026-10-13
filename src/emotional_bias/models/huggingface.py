"""
Hugging Face Inference API client.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp
import structlog

from .base import Message, ModelConfig, ModelResponse, ModelRole

logger = structlog.get_logger(__name__)


class HuggingFaceClient:
    """Client for Hugging Face Inference API."""

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

    def _format_messages(self, messages: list[Message]) -> str:
        """Format messages for HF chat template."""
        formatted = []
        for msg in messages:
            if msg.role == ModelRole.SYSTEM:
                formatted.append(f"<|system|>\n{msg.content}")
            elif msg.role == ModelRole.USER:
                formatted.append(f"<|user|>\n{msg.content}")
            elif msg.role == ModelRole.ASSISTANT:
                formatted.append(f"<|assistant|>\n{msg.content}")
        return "\n".join(formatted) + "\n<|assistant|>\n"

    async def generate(
        self,
        messages: list[Message],
        config: ModelConfig | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response from Hugging Face Inference API."""
        config = config or self.config
        start_time = time.perf_counter()

        # HF Inference API expects a single prompt for text generation
        prompt = self._format_messages(messages)

        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_new_tokens": config.max_tokens,
                "return_full_text": False,
                **config.extra_params,
            },
        }

        semaphore = await self._get_semaphore()
        async with semaphore:
            session = await self._get_session()
            try:
                url = f"/models/{config.model_id}"
                async with session.post(url, json=payload) as response:
                    if response.status == 503:
                        # Model is loading, wait and retry
                        await asyncio.sleep(10)
                        async with session.post(url, json=payload) as retry_response:
                            retry_response.raise_for_status()
                            data = await retry_response.json()
                    else:
                        response.raise_for_status()
                        data = await response.json()

                latency_ms = (time.perf_counter() - start_time) * 1000

                # HF returns list of dicts with generated_text
                if isinstance(data, list) and data:
                    response_text = data[0].get("generated_text", "")
                elif isinstance(data, dict):
                    response_text = data.get("generated_text", "")
                else:
                    response_text = str(data)

                return ModelResponse(
                    model_config=config,
                    messages=messages,
                    response_text=response_text.strip(),
                    usage=None,  # HF doesn't always return usage
                    latency_ms=latency_ms,
                    metadata={"hf_model": config.model_id},
                )

            except aiohttp.ClientError as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.error("hf_request_failed", error=str(e), model=config.model_id)
                return ModelResponse(
                    model_config=config,
                    messages=messages,
                    response_text="",
                    latency_ms=latency_ms,
                    error=f"Hugging Face request failed: {e}",
                )
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.exception("hf_unexpected_error", error=str(e))
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
