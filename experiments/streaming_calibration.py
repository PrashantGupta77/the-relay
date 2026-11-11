import csv
import json
import os
import time
import uuid

import traceback

from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL")

REQUEST_COUNT = 3
INTERVAL_SECONDS = 10

if not API_KEY or not MODEL:
    raise RuntimeError("Groq configuration missing from .env")


# --------------------------------------------------
# Output paths
# --------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = RESULTS_DIR / "results.csv"
TRACE_FILE = RESULTS_DIR / "traces.jsonl"

CSV_COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
]


# --------------------------------------------------
# Client: no automatic retries
# --------------------------------------------------

client = Groq(
    api_key=API_KEY,
    timeout=30.0,
    max_retries=0,
)


# --------------------------------------------------
# Persistence helpers
# --------------------------------------------------

def save_csv_row(row):
    needs_header = (
        not CSV_FILE.exists()
        or CSV_FILE.stat().st_size == 0
    )

    with CSV_FILE.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS
        )

        if needs_header:
            writer.writeheader()

        writer.writerow(row)


def save_trace(record):
    with TRACE_FILE.open(
        "a", encoding="utf-8"
    ) as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def classify_error(exc):
    status_code = getattr(exc, "status_code", None)

    if status_code == 429:
        return "rate_limited"

    if type(exc).__name__ in (
        "APITimeoutError",
        "TimeoutException",
        "TimeoutError",
    ):
        return "timeout"

    return "error"


# --------------------------------------------------
# One streaming request
# --------------------------------------------------

def execute_request(index):
    logical_id = str(uuid.uuid4())

    start_time = datetime.now(timezone.utc).isoformat()
    start_perf = time.perf_counter()

    first_chunk_ms = None
    first_content_ms = None

    output_parts = []
    tokens_in = None
    tokens_out = None
    finish_reason = None

    status = "error"
    error_type = None
    http_status = None
    headers = {}

    print(f"\nStarting streaming request {index}/{REQUEST_COUNT}")

    try:
        with client.chat.completions.with_streaming_response.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Answer concisely and accurately."
                },
                {
                    "role": "user",
                    "content": (
                        "Explain in two short sentences why "
                        "measuring first-token latency is "
                        "different from measuring total latency."
                    )
                },
            ],
            max_completion_tokens=512,
            stream=True,
        ) as raw_response:

            http_status = raw_response.status_code

            header_names = [
                "x-request-id",
                "x-ratelimit-limit-requests",
                "x-ratelimit-remaining-requests",
                "x-ratelimit-reset-requests",
                "x-ratelimit-limit-tokens",
                "x-ratelimit-remaining-tokens",
                "x-ratelimit-reset-tokens",
                "retry-after",
            ]

            headers = {
                name: raw_response.headers.get(name)
                for name in header_names
                if raw_response.headers.get(name) is not None
            }

            provider_request_id = headers.get("x-request-id")

            stream = raw_response.parse()

            for chunk in stream:                                
                now_ms = round(
                    (time.perf_counter() - start_perf) * 1000,
                    2
                )

                if first_chunk_ms is None:
                    first_chunk_ms = now_ms

                # Capture usage from the final usage chunk.
                if getattr(chunk, "usage", None):
                    tokens_in = chunk.usage.prompt_tokens
                    tokens_out = chunk.usage.completion_tokens

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta
                content = getattr(delta, "content", None)

                if content:
                    if first_content_ms is None:
                        first_content_ms = now_ms

                    output_parts.append(content)

            status = "ok"

    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
        error_traceback = traceback.format_exc()

        status = classify_error(exc)

        print("\n--- STREAMING ERROR DETAILS ---")
        print(f"Error type: {error_type}")
        print(f"Error message: {error_message}")
        print(error_traceback)
        print("--- END ERROR DETAILS ---")

        response_obj = getattr(exc, "response", None)

        if response_obj is not None:
            http_status = response_obj.status_code

            headers.update({
                name: response_obj.headers.get(name)
                for name in [
                    "x-request-id",
                    "retry-after",
                    "x-ratelimit-remaining-requests",
                    "x-ratelimit-remaining-tokens",
                ]
                if response_obj.headers.get(name) is not None
            })

    total_ms = round(
        (time.perf_counter() - start_perf) * 1000,
        2
    )

    answer = "".join(output_parts)

    row = {
        "request_id": logical_id,
        "scenario": "streaming_calibration",
        "source": "real",
        "start_time": start_time,
        "first_token_ms": (
            first_content_ms
            if first_content_ms is not None
            else ""
        ),
        "total_ms": total_ms,
        "status": status,
        "retries": 0,
        "tokens_in": tokens_in if tokens_in is not None else "",
        "tokens_out": tokens_out if tokens_out is not None else "",
        "retrieval_setting": "none",
        "answer_grade": "",
    }

    save_csv_row(row)

    save_trace({
        "request_id": logical_id,
        "scenario": "streaming_calibration",
        "source": "real",
        "provider_request_id": headers.get("x-request-id"),
        "attempt": 1,
        "retries": 0,
        "status": status,
        "error_type": error_type,
        "error_message": error_message if error_type else None,
        "error_traceback": error_traceback if error_type else None,
        "http_status": http_status,
        "first_chunk_ms": first_chunk_ms,
        "first_content_ms": first_content_ms,
        "total_ms": total_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "finish_reason": finish_reason,
        "answer": answer,
        "events": [
            {
                "event": "request_started",
                "timestamp": start_time,
            },
            {
                "event": "first_chunk_received",
                "elapsed_ms": first_chunk_ms,
            },
            {
                "event": "first_text_content_received",
                "elapsed_ms": first_content_ms,
            },
            {
                "event": "stream_completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": total_ms,
            },
        ],
        "rate_limit_headers": headers,
    })

    print(f"Status: {status}")
    print(f"HTTP status: {http_status}")
    print(f"First chunk: {first_chunk_ms} ms")
    print(f"First text content / TTFT: {first_content_ms} ms")
    print(f"Total latency: {total_ms} ms")
    print(f"Input tokens: {tokens_in}")
    print(f"Output tokens: {tokens_out}")
    print(f"Finish reason: {finish_reason}")
    print(f"Answer: {answer}")

    if error_type:
        print(f"Error type: {error_type}")

    return status


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("=" * 55)
    print("GROQ STREAMING CALIBRATION")
    print("=" * 55)

    print(f"Model: {MODEL}")
    print(f"Requests: {REQUEST_COUNT}")
    print(f"Interval after completion: {INTERVAL_SECONDS}s")
    print("Concurrency: 1")
    print("Retries: 0")

    counts = {
        "ok": 0,
        "rate_limited": 0,
        "timeout": 0,
        "error": 0,
    }

    for index in range(1, REQUEST_COUNT + 1):
        status = execute_request(index)
        counts[status] += 1

        if index < REQUEST_COUNT:
            time.sleep(INTERVAL_SECONDS)

    print("\n" + "=" * 55)
    print("STREAMING CALIBRATION SUMMARY")
    print("=" * 55)

    for status, count in counts.items():
        print(f"{status}: {count}")

    print(f"\nCSV: {CSV_FILE}")
    print(f"Traces: {TRACE_FILE}")


if __name__ == "__main__":
    main()
