
import csv
import json
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

CSV_FILE = RESULTS_DIR / "results.csv"
TRACE_FILE = RESULTS_DIR / "traces.jsonl"

CSV_FIELDS = [
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


def log_load_request(
    *,
    scenario: str,
    start_time: datetime,
    total_ms: float,
    first_token_ms: float | None,
    status: str,
    attempts: list[dict[str, Any]],
    tokens_in: int | None = None,
    tokens_out: int | None = None,
) -> str:
    """Log one logical stand-in request and all its attempts."""

    valid_statuses = {
        "ok",
        "rate_limited",
        "timeout",
        "error",
    }

    if status not in valid_statuses:
        raise ValueError(f"Invalid request status: {status}")

    if not attempts:
        raise ValueError("At least one attempt is required")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    request_id = str(uuid.uuid4())
    retries = len(attempts) - 1

    csv_row = {
        "request_id": request_id,
        "scenario": scenario,
        "source": "stand-in",
        "start_time": start_time.isoformat(),
        "first_token_ms": (
            round(first_token_ms, 2)
            if first_token_ms is not None
            else ""
        ),
        "total_ms": round(total_ms, 2),
        "status": status,
        "retries": retries,
        "tokens_in": tokens_in if tokens_in is not None else "",
        "tokens_out": tokens_out if tokens_out is not None else "",
        "retrieval_setting": "",
        "answer_grade": "",
    }

    if CSV_FILE.exists():
        with CSV_FILE.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            existing_fields = next(csv.reader(handle), [])

        if existing_fields != CSV_FIELDS:
            raise ValueError(
                "Existing results.csv header does not match "
                f"required schema: {existing_fields}"
            )

        write_header = False
    else:
        write_header = True

    with CSV_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(csv_row)

    trace = {
        "request_id": request_id,
        "scenario": scenario,
        "source": "stand-in",
        "start_time": start_time.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "total_ms": round(total_ms, 2),
        "first_token_ms": (
            round(first_token_ms, 2)
            if first_token_ms is not None
            else None
        ),
        "retries": retries,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "attempts": attempts,
    }

    with TRACE_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(trace, ensure_ascii=False) + "\n"
        )

    return request_id
