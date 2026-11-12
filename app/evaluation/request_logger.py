
import csv
import json
import traceback
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


def log_rag_request(
    *,
    result: dict[str, Any] | None,
    start_time: datetime,
    total_ms: float,
    first_token_ms: float | None = None,
    source: str = "real",
    scenario: str = "rag_end_to_end",
    error: Exception | None = None,
) -> str:
    """Append one logical RAG request to results.csv and traces.jsonl."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    request_id = str(uuid.uuid4())

    provider = (result or {}).get("provider") or {}
    retrieval = (result or {}).get("retrieval") or {}

    # Provider transport success is separate from answer quality.
    # A truncated response can still be a successful HTTP/API request.
    status = "error" if error is not None else "ok"

    csv_row = {
        "request_id": request_id,
        "scenario": scenario,
        "source": source,
        "start_time": start_time.isoformat(),
        # Current RAG test uses non-streaming generation.
        "first_token_ms": (
            round(first_token_ms, 2)
            if first_token_ms is not None
            else ""
        ),
        "total_ms": round(total_ms, 2),
        "status": status,
        "retries": 0,
        "tokens_in": provider.get("input_tokens", ""),
        "tokens_out": provider.get("output_tokens", ""),
        "retrieval_setting": retrieval.get("setting", ""),
        # Leave blank until manual grading.
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
                f"the required schema: {existing_fields}"
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
        "source": source,
        "start_time": start_time.isoformat(),
        "attempt": 1,
        "retries": 0,
        "status": status,
        "first_token_ms": (
            round(first_token_ms, 2)
            if first_token_ms is not None
            else None
        ),
        "first_token_ms": None,
        "provider": provider.get("provider"),
        "model": provider.get("model"),
        "provider_request_id": provider.get("provider_request_id"),
        "provider_latency_ms": provider.get("latency_ms"),
        "tokens_in": provider.get("input_tokens"),
        "tokens_out": provider.get("output_tokens"),
        "total_tokens": provider.get("total_tokens"),
        "finish_reason": provider.get("finish_reason"),
        "rate_limit_headers": provider.get("rate_limit_headers"),
        "pipeline_status": (result or {}).get("status"),
        "answer": (result or {}).get("answer"),
        "reason": (result or {}).get("reason"),
        "citations": (result or {}).get("citations", []),
        "retrieval": retrieval,
        "error_type": type(error).__name__ if error else None,
        "error_message": str(error) if error else None,
        "error_traceback": (
            traceback.format_exc() if error else None
        ),
        "events": [
            {
                "event": "request_completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": round(total_ms, 2),
            }
        ],
    }

    with TRACE_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(trace, ensure_ascii=False) + "\n"
        )

    return request_id
