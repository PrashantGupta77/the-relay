import csv
import json
import os
import time
import uuid
import httpx

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

REQUEST_COUNT = 10
INTERVAL_SECONDS = 6.0

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY missing.")

if not MODEL:
    raise RuntimeError("GROQ_MODEL missing.")


# --------------------------------------------------
# Output files
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
# CSV and trace helpers
# --------------------------------------------------

def existing_request_ids():
    if not CSV_FILE.exists():
        return set()

    with CSV_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        return {
            row["request_id"]
            for row in csv.DictReader(file)
        }


def save_result(row):
    file_exists = CSV_FILE.exists()

    with CSV_FILE.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS
        )

        if not file_exists or CSV_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(row)


def save_trace(record):
    with TRACE_FILE.open(
        "a", encoding="utf-8"
    ) as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def make_row(
    request_id,
    scenario,
    start_time,
    total_ms,
    status,
    tokens_in=None,
    tokens_out=None,
):
    return {
        "request_id": request_id,
        "scenario": scenario,
        "source": "real",
        "start_time": start_time,
        "first_token_ms": "",
        "total_ms": total_ms,
        "status": status,
        "retries": 0,
        "tokens_in": tokens_in if tokens_in is not None else "",
        "tokens_out": tokens_out if tokens_out is not None else "",
        "retrieval_setting": "none",
        "answer_grade": "",
    }


# --------------------------------------------------
# Include the previous single-request measurement
# --------------------------------------------------

def import_first_request():
    baseline_file = RESULTS_DIR / "first_real_request.json"

    if not baseline_file.exists():
        print("Baseline JSON not found. Skipping import.")
        return

    with baseline_file.open(
        "r", encoding="utf-8"
    ) as file:
        baseline = json.load(file)

    request_id = baseline.get("request_id")

    if not request_id:
        print("Baseline request ID missing. Skipping import.")
        return

    if request_id in existing_request_ids():
        print("Baseline already exists in results.csv.")
        return

    status = (
        "ok"
        if baseline.get("http_status") == 200
        else "error"
    )

    row = make_row(
        request_id=request_id,
        scenario="calibration_single",
        start_time=baseline["start_time"],
        total_ms=baseline["total_ms"],
        status=status,
        tokens_in=baseline.get("input_tokens"),
        tokens_out=baseline.get("output_tokens"),
    )

    save_result(row)

    save_trace({
        "request_id": request_id,
        "provider_request_id": request_id,
        "scenario": "calibration_single",
        "source": "real",
        "events": [
            {
                "event": "request_started",
                "timestamp": baseline["start_time"],
            },
            {
                "event": "request_completed",
                "timestamp": baseline["completed_at"],
                "total_ms": baseline["total_ms"],
                "http_status": baseline["http_status"],
            },
        ],
        "rate_limit_headers": baseline.get(
            "rate_limit_headers", {}
        ),
    })

    print("Imported previous baseline measurement.")


# --------------------------------------------------
# Groq client
# --------------------------------------------------

client = Groq(
    api_key=API_KEY,
    timeout=30.0,
    max_retries=0,
)


# --------------------------------------------------
# Fixed calibration prompt
# --------------------------------------------------

MESSAGES = [
    {
        "role": "system",
        "content": "Follow the user's instruction exactly."
    },
    {
        "role": "user",
        "content": (
            "Reply with exactly this text and nothing else: "
            "RELAY_CALIBRATION_OK"
        )
    }
]


# --------------------------------------------------
# Execute one request
# --------------------------------------------------

def execute_request(index):
    logical_id = str(uuid.uuid4())

    start_time = datetime.now(timezone.utc).isoformat()
    start_perf = time.perf_counter()

    provider_request_id = None
    captured_headers = {}
    tokens_in = None
    tokens_out = None
    status = "error"
    error_type = None

    try:
        raw = client.chat.completions.with_raw_response.create(
            model=MODEL,
            messages=MESSAGES,
            max_completion_tokens=128,
            stream=False,
        )

        response = raw.parse()

        total_ms = round(
            (time.perf_counter() - start_perf) * 1000,
            2
        )

        headers = raw.headers

        provider_request_id = headers.get("x-request-id")

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

        captured_headers = {
            name: headers.get(name)
            for name in header_names
            if headers.get(name) is not None
        }

        usage = response.usage

        if usage:
            tokens_in = usage.prompt_tokens
            tokens_out = usage.completion_tokens

        status = "ok"

    except Exception as exc:
        total_ms = round(
            (time.perf_counter() - start_perf) * 1000,
            2
        )

        response_obj = getattr(exc, "response", None)
        http_status = getattr(exc, "status_code", None)

        if response_obj is not None:
            captured_headers = {
                name: response_obj.headers.get(name)
                for name in [
                    "x-request-id",
                    "retry-after",
                    "x-ratelimit-remaining-requests",
                    "x-ratelimit-remaining-tokens",
                ]
                if response_obj.headers.get(name) is not None
            }

        if http_status == 429:
            status = "rate_limited"
        elif isinstance(exc, (TimeoutError, httpx.TimeoutException)):
            status = "timeout"
        else:
            status = "error"

        error_type = type(exc).__name__

    # Save one logical request row immediately.
    row = make_row(
        request_id=logical_id,
        scenario="calibration_sequential",
        start_time=start_time,
        total_ms=total_ms,
        status=status,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )

    save_result(row)

    save_trace({
        "request_id": logical_id,
        "provider_request_id": provider_request_id,
        "scenario": "calibration_sequential",
        "source": "real",
        "attempt": 1,
        "retries": 0,
        "status": status,
        "error_type": error_type,
        "total_ms": total_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "events": [
            {
                "event": "request_started",
                "timestamp": start_time,
            },
            {
                "event": "request_completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_ms": total_ms,
            },
        ],
        "rate_limit_headers": captured_headers,
    })

    print(
        f"[{index}/{REQUEST_COUNT}] "
        f"status={status}, "
        f"latency={total_ms} ms, "
        f"input={tokens_in}, "
        f"output={tokens_out}"
    )

    if error_type:
        print(f"    Error type: {error_type}")

    return status


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("=" * 55)
    print("GROQ SEQUENTIAL CALIBRATION")
    print("=" * 55)

    print(f"Model: {MODEL}")
    print(f"Requests: {REQUEST_COUNT}")
    print(f"Interval: {INTERVAL_SECONDS} seconds")
    print("Concurrency: 1")
    print("Retries: 0")
    print()

    import_first_request()

    counts = {
        "ok": 0,
        "rate_limited": 0,
        "timeout": 0,
        "error": 0,
    }

    for index in range(1, REQUEST_COUNT + 1):
        print(f"\nSending request {index}/{REQUEST_COUNT}...")

        status = execute_request(index)
        counts[status] += 1

        if index < REQUEST_COUNT:
            time.sleep(INTERVAL_SECONDS)

    print("\n" + "=" * 55)
    print("CALIBRATION SUMMARY")
    print("=" * 55)

    for status, count in counts.items():
        print(f"{status}: {count}")

    print(f"\nCSV saved: {CSV_FILE}")
    print(f"Traces saved: {TRACE_FILE}")
    print("\nCalibration completed.")


if __name__ == "__main__":
    main()
