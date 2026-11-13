
import asyncio
import json
import time

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from app.evaluation.load_logger import log_load_request


class StandInClient:
    """Async streaming client for the local OpenAI-compatible stand-in."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        max_retries: int = 2,
        deadline_seconds: float = 30.0,
    ):
        self.url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.max_retries = max_retries
        self.deadline_seconds = deadline_seconds

    async def generate(
        self,
        client: httpx.AsyncClient,
        *,
        scenario: str,
        messages: list[dict[str, Any]],
        queue_wait_ms: float = 0.0,
    ) -> dict[str, Any]:

        start_time = (
            datetime.now(timezone.utc)
            - timedelta(milliseconds=queue_wait_ms)
        )
        start_clock = (
            time.perf_counter()
            - (queue_wait_ms / 1000)
        )

        attempts: list[dict[str, Any]] = []
        first_token_ms = None
        tokens_in = None
        tokens_out = None
        final_status = "error"
        final_content = ""
        final_reason = None

        payload = {
            "model": "standin-model",
            "messages": messages,
            "stream": True,
            "max_tokens": 512,
        }

        for attempt_number in range(1, self.max_retries + 2):
            elapsed = time.perf_counter() - start_clock
            remaining = self.deadline_seconds - elapsed

            if remaining <= 0:
                final_status = "timeout"
                break

            attempt_start = time.perf_counter()
            attempt_trace = {
                "attempt": attempt_number,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }

            retry_after = None
            retry_reason = None
            attempt_status = "error"
            attempt_error = None
            attempt_content = ""
            attempt_tokens_in = None
            attempt_tokens_out = None
            done_seen = False

            try:
                async with client.stream(
                    "POST",
                    self.url,
                    json=payload,
                    timeout=httpx.Timeout(remaining),
                ) as response:

                    attempt_trace["http_status"] = response.status_code
                    retry_after = response.headers.get("Retry-After")

                    if response.status_code != 200:
                        body = await response.aread()
                        body_text = body.decode(
                            "utf-8",
                            errors="replace",
                        )

                        try:
                            error_body = json.loads(body_text)
                        except json.JSONDecodeError:
                            error_body = {"raw": body_text[:1000]}

                        attempt_trace["error_body"] = error_body

                        error_type = (
                            error_body.get("error", {}).get("type")
                            if isinstance(error_body, dict)
                            else None
                        )

                        if response.status_code == 429:
                            attempt_status = "rate_limited"
                            attempt_error = error_type or "http_429"

                            if error_type == "daily_budget_exhausted":
                                retry_reason = None
                            elif retry_after is not None:
                                retry_reason = "retry_after"

                        elif response.status_code == 503:
                            attempt_status = "error"
                            attempt_error = "http_503"
                            retry_reason = "transient_503"

                        else:
                            attempt_status = "error"
                            attempt_error = f"http_{response.status_code}"

                    else:
                        attempt_status = "ok"

                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue

                            data = line[len("data:"):].strip()

                            if data == "[DONE]":
                                done_seen = True
                                break

                            if not data:
                                continue

                            event = json.loads(data)

                            choices = event.get("choices", [])
                            if choices:
                                choice = choices[0]
                                delta = choice.get("delta", {})
                                content = delta.get("content")

                                if content:
                                    if first_token_ms is None:
                                        first_token_ms = (
                                            time.perf_counter()
                                            - start_clock
                                        ) * 1000

                                    attempt_content += content

                                if choice.get("finish_reason"):
                                    attempt_trace["finish_reason"] = (
                                        choice["finish_reason"]
                                    )

                            usage = event.get("usage")
                            if usage:
                                attempt_tokens_in = usage.get(
                                    "prompt_tokens"
                                )
                                attempt_tokens_out = usage.get(
                                    "completion_tokens"
                                )

                        if not done_seen:
                            attempt_status = "error"
                            attempt_error = "incomplete_stream"
                            retry_reason = "incomplete_stream"
                        else:
                            final_content = attempt_content
                            tokens_in = attempt_tokens_in
                            tokens_out = attempt_tokens_out
                            final_reason = attempt_trace.get(
                                "finish_reason"
                            )
                            final_status = "ok"
                            retry_reason = None

            except httpx.TimeoutException as exc:
                attempt_status = "timeout"
                attempt_error = type(exc).__name__
                retry_reason = "timeout"

            except httpx.HTTPError as exc:
                attempt_status = "error"
                attempt_error = type(exc).__name__
                retry_reason = "connection_error"

            except (json.JSONDecodeError, ValueError) as exc:
                attempt_status = "error"
                attempt_error = type(exc).__name__
                retry_reason = "invalid_stream"

            except Exception as exc:
                attempt_status = "error"
                attempt_error = type(exc).__name__
                retry_reason = None

            attempt_ms = (
                time.perf_counter() - attempt_start
            ) * 1000

            attempt_trace.update({
                "status": attempt_status,
                "latency_ms": round(attempt_ms, 2),
                "retry_after": retry_after,
                "error": attempt_error,
                "retry_reason": retry_reason,
                "content_length": len(attempt_content),
                "tokens_in": attempt_tokens_in,
                "tokens_out": attempt_tokens_out,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })

            attempts.append(attempt_trace)

            if final_status == "ok":
                break

            if retry_reason is None:
                final_status = attempt_status
                break

            if attempt_number > self.max_retries:
                final_status = attempt_status
                break

            if retry_reason == "retry_after":
                try:
                    delay = max(0.0, float(retry_after))
                except (TypeError, ValueError):
                    delay = 1.0
            else:
                delay = min(
                    0.5 * (2 ** (attempt_number - 1)),
                    2.0,
                )

            elapsed = time.perf_counter() - start_clock
            remaining = self.deadline_seconds - elapsed

            if delay >= remaining:
                final_status = "timeout"
                attempt_trace["retry_skipped"] = "deadline"
                break

            attempt_trace["retry_delay_seconds"] = delay
            await asyncio.sleep(delay)

        total_ms = (time.perf_counter() - start_clock) * 1000

        # Exactly one CSV row and one trace per logical request.
        request_id = log_load_request(
            scenario=scenario,
            start_time=start_time,
            total_ms=total_ms,
            first_token_ms=first_token_ms,
            status=final_status,
            attempts=attempts,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            queue_wait_ms=queue_wait_ms,
        )

        return {
            "request_id": request_id,
            "status": final_status,
            "content": final_content,
            "first_token_ms": first_token_ms,
            "total_ms": round(total_ms, 2),
            "retries": max(0, len(attempts) - 1),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "finish_reason": final_reason,
            "attempts": attempts,
        }
