
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


class SimulationConfig(BaseModel):
    base_latency_ms: int = Field(default=500, ge=0, le=60_000)
    ttft_ms: int = Field(default=250, ge=0, le=60_000)
    chunk_delay_ms: int = Field(default=80, ge=0, le=10_000)
    chunks_per_response: int = Field(default=4, ge=1, le=20)

    max_concurrent_requests: int = Field(default=8, ge=1, le=500)
    overload_slowdown_ms: int = Field(default=500, ge=0, le=60_000)

    failure_every_n_requests: int = Field(default=0, ge=0)
    rate_limit_every_n_requests: int = Field(default=0, ge=0)

    requests_per_minute: int = Field(default=0, ge=0)
    tokens_per_minute: int = Field(default=0, ge=0)
    daily_request_budget: int = Field(default=0, ge=0)

    simulated_input_tokens: int = Field(default=1200, ge=1)
    simulated_output_tokens: int = Field(default=180, ge=1)


class CompletionRequest(BaseModel):
    model: str = "standin-model"
    messages: list[dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    max_tokens: int = 512


class ServiceState:
    def __init__(self) -> None:
        self.config = SimulationConfig()
        self.lock = asyncio.Lock()
        self.active_requests = 0
        self.total_requests = 0
        self.minute_requests: list[float] = []
        self.minute_tokens: list[tuple[float, int]] = []
        self.daily_requests = 0
        self.started_at = time.time()

    async def snapshot(self) -> dict[str, Any]:
        async with self.lock:
            return {
                "active_requests": self.active_requests,
                "total_requests": self.total_requests,
                "daily_requests": self.daily_requests,
                "config": self.config.model_dump(),
            }


state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="The Relay - Local Stand-in Model Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    snapshot = await state.snapshot()
    return {"status": "ok", **snapshot}


@app.get("/config")
async def get_config():
    return state.config.model_dump()


@app.put("/config")
async def update_config(config: SimulationConfig):
    async with state.lock:
        state.config = config
    return {"updated": True, "config": config.model_dump()}


def make_demo_answer(messages: list[dict[str, Any]]) -> str:
    """
    Deterministic service-behavior fixture.
    This is NOT an LLM and must not be used to measure answer quality.
    """
    question = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            question = str(message.get("content", ""))
            break

    return (
        "STAND-IN RESPONSE (deterministic fixture): "
        "The local service received the request successfully. "
        "This response is generated from a fixed template, not a language model. "
        f"Request question: {question[:300]}"
    )


async def admit_request(request_tokens: int) -> tuple[int, str | None]:
    now = time.time()

    async with state.lock:
        config = state.config
        state.total_requests += 1
        request_number = state.total_requests

        state.minute_requests = [
            timestamp for timestamp in state.minute_requests
            if now - timestamp < 60
        ]
        state.minute_tokens = [
            item for item in state.minute_tokens
            if now - item[0] < 60
        ]

        if (
            config.daily_request_budget > 0
            and state.daily_requests >= config.daily_request_budget
        ):
            return request_number, "daily_budget"

        if (
            config.requests_per_minute > 0
            and len(state.minute_requests) >= config.requests_per_minute
        ):
            return request_number, "rpm"

        current_tokens = sum(tokens for _, tokens in state.minute_tokens)
        if (
            config.tokens_per_minute > 0
            and current_tokens + request_tokens > config.tokens_per_minute
        ):
            return request_number, "tpm"

        state.daily_requests += 1
        state.minute_requests.append(now)
        state.minute_tokens.append((now, request_tokens))
        state.active_requests += 1

        if (
            config.rate_limit_every_n_requests > 0
            and request_number % config.rate_limit_every_n_requests == 0
        ):
            state.active_requests -= 1
            return request_number, "injected_429"

        if (
            config.failure_every_n_requests > 0
            and request_number % config.failure_every_n_requests == 0
        ):
            state.active_requests -= 1
            return request_number, "injected_503"

        return request_number, None


async def release_request():
    async with state.lock:
        state.active_requests = max(0, state.active_requests - 1)


def rate_limit_response(reason: str):
    headers = {}
    if reason in ("rpm", "tpm", "injected_429"):
        headers["Retry-After"] = "2"

    if reason == "daily_budget":
        return JSONResponse(
            status_code=429,
            content={"error": {"type": "daily_budget_exhausted"}},
        )

    return JSONResponse(
        status_code=429,
        headers=headers,
        content={"error": {"type": reason}},
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    payload: CompletionRequest,
    request: Request,
):
    config = state.config
    input_tokens = config.simulated_input_tokens

    reserved_tokens = input_tokens + config.simulated_output_tokens
    request_number, rejection = await admit_request(reserved_tokens)

    if rejection == "daily_budget":
        return rate_limit_response("daily_budget")
    if rejection == "rpm":
        return rate_limit_response("rpm")
    if rejection == "tpm":
        return rate_limit_response("tpm")
    if rejection == "injected_429":
        return rate_limit_response("injected_429")
    if rejection == "injected_503":
        raise HTTPException(status_code=503, detail="Injected transient failure")

    # Capture the active request count after this request has been admitted.
    async with state.lock:
        concurrent_requests = state.active_requests

    slowdown = max(
        0,
        concurrent_requests - config.max_concurrent_requests
    ) * config.overload_slowdown_ms

    print(
        f"[STANDIN] request={request_number} "
        f"active={concurrent_requests} "
        f"limit={config.max_concurrent_requests} "
        f"slowdown_ms={slowdown}"
    )

    total_delay_ms = config.base_latency_ms + slowdown
    answer = make_demo_answer(payload.messages)
    output_tokens = config.simulated_output_tokens

    if not payload.stream:
        try:
            await asyncio.sleep(total_delay_ms / 1000)
            return {
                "id": f"standin-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "model": payload.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                "simulation": True,
                "request_number": request_number,
            }
        finally:
            await release_request()

    async def stream_events():
        try:
            effective_ttft_ms = config.ttft_ms + slowdown
            await asyncio.sleep(effective_ttft_ms / 1000)

            pieces = [
                answer[i::config.chunks_per_response]
                for i in range(config.chunks_per_response)
            ]

            # Preserve readable output by splitting into contiguous pieces.
            piece_size = max(1, (len(answer) + config.chunks_per_response - 1)
                             // config.chunks_per_response)
            pieces = [
                answer[i:i + piece_size]
                for i in range(0, len(answer), piece_size)
            ]

            for piece in pieces:
                event = {
                    "id": f"standin-{request_number}",
                    "object": "chat.completion.chunk",
                    "model": payload.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": piece},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(config.chunk_delay_ms / 1000)

            final_event = {
                "id": f"standin-{request_number}",
                "object": "chat.completion.chunk",
                "model": payload.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            }
            yield f"data: {json.dumps(final_event)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            await release_request()

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
