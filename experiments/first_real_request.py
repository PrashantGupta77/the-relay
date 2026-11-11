import json
import os
import time

from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# --------------------------------------------------
# 1. Load configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL")

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing.")

if not MODEL:
    raise RuntimeError("GROQ_MODEL is missing.")


# --------------------------------------------------
# 2. Initialize Groq client
# --------------------------------------------------

client = Groq(
    api_key=API_KEY,
    timeout=30.0,
    max_retries=0,
)


# --------------------------------------------------
# 3. Prepare a small calibration prompt
# --------------------------------------------------

messages = [
    {
        "role": "system",
        "content": "Follow the user's instruction exactly."
    },
    {
        "role": "user",
        "content": "Reply with exactly this text and nothing else: RELAY_CALIBRATION_OK"
    }
]


# --------------------------------------------------
# 4. Send one real API request
# --------------------------------------------------

start_time = datetime.now(timezone.utc)
start_perf = time.perf_counter()

print("Sending first real Groq completion request...")
print(f"Model: {MODEL}")

try:
    raw_response = client.chat.completions.with_raw_response.create(
        model=MODEL,
        messages=messages,
        max_completion_tokens=128,
        stream=False,
    )

    response = raw_response.parse()

    total_ms = round(
        (time.perf_counter() - start_perf) * 1000,
        2
    )

    content = response.choices[0].message.content or ""

    usage = response.usage

    headers = raw_response.headers

    relevant_headers = [
        "x-request-id",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
        "retry-after",
    ]

    captured_headers = {
        name: headers.get(name)
        for name in relevant_headers
        if headers.get(name) is not None
    }

    result = {
        "source": "real",
        "model": MODEL,
        "start_time": start_time.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "http_status": raw_response.status_code,
        "total_ms": total_ms,
        "request_id": captured_headers.get("x-request-id"),
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
        "expected_response": "RELAY_CALIBRATION_OK",
        "actual_response": content,
        "response_matches_expected": (
            content.strip() == "RELAY_CALIBRATION_OK"
        ),
        "rate_limit_headers": captured_headers,
    }

    # --------------------------------------------------
    # 5. Save results
    # --------------------------------------------------

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "first_real_request.json"

    output_file.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    # --------------------------------------------------
    # 6. Print safe summary
    # --------------------------------------------------

    print("\nRequest completed successfully.")
    print(f"HTTP status: {result['http_status']}")
    print(f"Total latency: {total_ms} ms")
    print(f"Input tokens: {result['input_tokens']}")
    print(f"Output tokens: {result['output_tokens']}")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Response: {content}")
    print(
        "Response matches expected:",
        result["response_matches_expected"]
    )

    print("\nCaptured rate-limit headers:")

    for name, value in captured_headers.items():
        print(f"{name}: {value}")

    print(f"\nSaved to: {output_file}")

except Exception as exc:
    elapsed_ms = round(
        (time.perf_counter() - start_perf) * 1000,
        2
    )

    safe_message = str(exc).replace(API_KEY, "[REDACTED]")

    print("\nAPI request failed.")
    print(f"Error type: {type(exc).__name__}")
    print(f"Error: {safe_message}")
    print(f"Elapsed time: {elapsed_ms} ms")

    raise
