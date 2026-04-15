"""LLM client protocol and provider implementations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str


@runtime_checkable
class LLMClient(Protocol):
    """Minimal async interface for LLM completion."""

    async def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...


class OpenAIClient:
    """Async OpenAI chat completion client."""

    def __init__(self, api_key: str | None = None, default_model: str = "gpt-5.4"):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.default_model = default_model

    async def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        model = model or self.default_model
        kwargs: dict = dict(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens

        response = await self._client.chat.completions.create(**kwargs)
        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=model,
        )


class AnthropicClient:
    """Async Anthropic chat completion client."""

    def __init__(self, api_key: str | None = None, default_model: str = "claude-sonnet-4-20250514"):
        import anthropic

        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.default_model = default_model

    async def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        model = model or self.default_model

        # Anthropic takes system as a separate param
        system_parts = []
        non_system = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                non_system.append(msg)

        kwargs: dict = dict(
            model=model,
            messages=non_system,
            temperature=temperature,
            max_tokens=max_tokens or 4096,
        )
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        response = await self._client.messages.create(**kwargs)
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
        return LLMResponse(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
        )
