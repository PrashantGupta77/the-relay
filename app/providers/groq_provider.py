import os
import time

from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from groq import Groq

from app.providers.base import (
    BaseProvider,
    ProviderResponse,
    ProviderStreamEvent,
)


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


RATE_LIMIT_HEADERS = [
    "x-request-id",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
    "retry-after",
]


def extract_headers(headers) -> dict[str, str]:
    """Extract relevant provider headers without exposing credentials."""

    return {
        name: headers.get(name)
        for name in RATE_LIMIT_HEADERS
        if headers.get(name) is not None
    }


class GroqProvider(BaseProvider):
    """Real Groq API provider with synchronous generation and streaming."""

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 30.0,
        client: Groq | None = None,
    ):
        self.model = model or os.getenv("GROQ_MODEL")

        if not self.model:
            raise RuntimeError("GROQ_MODEL is missing.")

        if client is None:
            api_key = os.getenv("GROQ_API_KEY")

            if not api_key:
                raise RuntimeError("GROQ_API_KEY is missing.")

            self.client = Groq(
                api_key=api_key,
                timeout=timeout,
                max_retries=0,
            )
        else:
            self.client = client

    @property
    def provider_name(self) -> str:
        return "groq"

    def _request_arguments(
        self,
        messages: list[dict[str, str]],
        max_completion_tokens: int,
        temperature: float | None,
    ) -> dict:
        """Build shared completion arguments."""

        arguments = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
        }

        if temperature is not None:
            arguments["temperature"] = temperature

        return arguments

    def generate(
        self,
        messages: list[dict[str, str]],
        max_completion_tokens: int = 512,
        temperature: float | None = None,
    ) -> ProviderResponse:
        """Generate a complete, non-streaming response."""

        start_perf = time.perf_counter()

        arguments = self._request_arguments(
            messages,
            max_completion_tokens,
            temperature,
        )

        raw_response = self.client.chat.completions.with_raw_response.create(
            **arguments,
            stream=False,
        )

        response = raw_response.parse()

        latency_ms = round(
            (time.perf_counter() - start_perf) * 1000,
            2,
        )

        choice = response.choices[0]
        usage = response.usage

        return ProviderResponse(
            text=choice.message.content or "",
            model=self.model,
            input_tokens=(
                usage.prompt_tokens if usage else None
            ),
            output_tokens=(
                usage.completion_tokens if usage else None
            ),
            total_tokens=(
                usage.total_tokens if usage else None
            ),
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason,
            provider_request_id=raw_response.headers.get(
                "x-request-id"
            ),
            rate_limit_headers=extract_headers(
                raw_response.headers
            ),
        )

    def stream(
        self,
        messages: list[dict[str, str]],
        max_completion_tokens: int = 512,
        temperature: float | None = None,
    ) -> Iterator[ProviderStreamEvent]:
        """
        Stream visible content and emit final usage metadata.

        elapsed_ms on content events measures time from the beginning
        of the request to that content event.

        The first non-empty content event is the TTFT measurement.
        """

        start_perf = time.perf_counter()

        arguments = self._request_arguments(
            messages,
            max_completion_tokens,
            temperature,
        )

        final_input_tokens = None
        final_output_tokens = None
        final_total_tokens = None
        finish_reason = None
        provider_request_id = None
        captured_headers = {}

        with self.client.chat.completions.with_streaming_response.create(
            **arguments,
            stream=True,
        ) as raw_response:

            captured_headers = extract_headers(
                raw_response.headers
            )

            provider_request_id = captured_headers.get(
                "x-request-id"
            )

            stream = raw_response.parse()

            for chunk in stream:

                elapsed_ms = round(
                    (time.perf_counter() - start_perf) * 1000,
                    2,
                )

                # Usage may arrive in a final chunk without choices.
                usage = getattr(chunk, "usage", None)

                if usage is not None:
                    final_input_tokens = usage.prompt_tokens
                    final_output_tokens = usage.completion_tokens
                    final_total_tokens = usage.total_tokens

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                content = getattr(
                    choice.delta,
                    "content",
                    None,
                )

                # Only non-empty visible content is emitted.
                if content:
                    yield ProviderStreamEvent(
                        event_type="content",
                        content=content,
                        elapsed_ms=elapsed_ms,
                    )

        total_elapsed_ms = round(
            (time.perf_counter() - start_perf) * 1000,
            2,
        )

        # One final metadata event per successfully completed stream.
        yield ProviderStreamEvent(
            event_type="metadata",
            elapsed_ms=total_elapsed_ms,
            input_tokens=final_input_tokens,
            output_tokens=final_output_tokens,
            total_tokens=final_total_tokens,
            finish_reason=finish_reason,
            provider_request_id=provider_request_id,
            rate_limit_headers=captured_headers,
        )
