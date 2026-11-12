from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class ProviderResponse:
    """Normalized result from a completed model request."""

    text: str
    model: str

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    latency_ms: float | None = None
    finish_reason: str | None = None

    provider_request_id: str | None = None
    rate_limit_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class ProviderStreamEvent:
    """
    Normalized event from a streaming model request.

    event_type:
        content  - user-visible text content
        metadata - final usage or completion metadata
    """

    event_type: str

    content: str | None = None

    elapsed_ms: float | None = None

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    finish_reason: str | None = None
    provider_request_id: str | None = None

    rate_limit_headers: dict[str, str] = field(default_factory=dict)


class BaseProvider(ABC):
    """
    Common synchronous interface for real and simulated providers.

    Retries, admission control, deadlines and verification belong
    to the pipeline layer, not individual providers.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return a short provider identifier."""
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        max_completion_tokens: int = 512,
        temperature: float | None = None,
    ) -> ProviderResponse:
        """Generate a complete response."""
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, str]],
        max_completion_tokens: int = 512,
        temperature: float | None = None,
    ) -> Iterator[ProviderStreamEvent]:
        """Stream response content and final metadata."""
        raise NotImplementedError
