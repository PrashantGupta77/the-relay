
import argparse
import csv
import json
import time
import traceback
import uuid

from datetime import datetime, timezone
from pathlib import Path

from app.evaluation.questions import QUESTIONS
from app.pipeline.orchestrator import (
    RAGOrchestrator,
    SYSTEM_PROMPT,
)
from app.providers.groq_provider import GroqProvider
from app.retrieval.retriever import Retriever, format_citation


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

CSV_FILE = RESULTS_DIR / "results.csv"
TRACE_FILE = RESULTS_DIR / "traces.jsonl"
EVALUATION_FILE = RESULTS_DIR / "rag_answer_evaluation.jsonl"

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

CROSS_PAPERS = [
    "attention_is_all_you_need",
    "dynamic_chunking",
]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def append_csv(record: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if CSV_FILE.exists():
        with CSV_FILE.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            existing_fields = next(csv.reader(handle), [])

        if existing_fields != CSV_FIELDS:
            raise ValueError(
                f"Unexpected results.csv header: {existing_fields}"
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

        writer.writerow(record)


def retrieve_evidence(
    retriever: Retriever,
    question: dict,
) -> dict:
    question_text = question["question"]
    question_id = question["id"]

    if question_id == "CROSS-01":
        combined = []

        for paper_id in CROSS_PAPERS:
            result = retriever.retrieve(
                question=question_text,
                top_k=5,
                paper_id=paper_id,
            )
            combined.extend(result["results"])

        combined.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        for rank, item in enumerate(combined, start=1):
            item["rank"] = rank

        return {
            "question": question_text,
            "setting": retriever.setting,
            "top_k": 10,
            "paper_filter": CROSS_PAPERS,
            "result_count": len(combined),
            "results": combined,
        }

    result = retriever.retrieve(
        question=question_text,
        top_k=5,
        paper_id=question["paper_id"],
    )

    return result


def build_messages(
    question_text: str,
    evidence_text: str,
) -> list[dict[str, str]]:
    user_prompt = f"""
Question:
{question_text}

Retrieved paper evidence:
{evidence_text}

Answer the question using only the evidence above.
Include valid evidence IDs in square brackets, for example [E1].
If the evidence is insufficient, begin with INSUFFICIENT_EVIDENCE:
"""

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def evaluate_one(question: dict, setting: str) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    request_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)
    start_perf = time.perf_counter()

    first_token_ms = None
    answer_parts = []
    metadata = None
    event_count = 0
    error = None
    retrieval = None
    evidence_lookup = {}
    evidence_text = ""

    provider = None
    answer = ""
    pipeline_status = "error"
    reason = None
    citation_ids = []
    citations = []

    try:
        retriever = Retriever(setting=setting)
        retrieval = retrieve_evidence(retriever, question)

        evidence_text, evidence_lookup = (
            RAGOrchestrator._build_evidence(
                retrieval["results"]
            )
        )

        messages = build_messages(
            question["question"],
            evidence_text,
        )

        provider = GroqProvider(timeout=45.0)

        for event in provider.stream(
            messages=messages,
            max_completion_tokens=768,
            temperature=0.0,
        ):
            event_count += 1

            if event.event_type == "content":
                if event.content:
                    if first_token_ms is None:
                        first_token_ms = event.elapsed_ms

                    answer_parts.append(event.content)

            elif event.event_type == "metadata":
                metadata = event

        answer = "".join(answer_parts).strip()

        if not answer:
            pipeline_status = "refused"
            reason = "Empty model response."

        elif metadata is None:
            pipeline_status = "refused"
            reason = "Stream ended without final metadata."

        elif metadata.finish_reason == "length":
            pipeline_status = "refused"
            reason = "Model response was truncated."

        elif answer.startswith("INSUFFICIENT_EVIDENCE:"):
            pipeline_status = "refused"
            reason = "Model reported insufficient evidence."

        else:
            valid, citation_ids, citation_error = (
                RAGOrchestrator._validate_citations(
                    answer,
                    evidence_lookup,
                )
            )

            if not valid:
                pipeline_status = "refused"
                reason = citation_error
            else:
                pipeline_status = "answered"

                for evidence_id in citation_ids:
                    item = evidence_lookup[evidence_id]

                    citations.append({
                        "evidence_id": evidence_id,
                        "chunk_id": item["chunk_id"],
                        "paper_id": item["paper_id"],
                        "citation": format_citation(item),
                    })

    except Exception as exc:
        error = exc
        reason = str(exc)

    total_ms = round(
        (time.perf_counter() - start_perf) * 1000,
        2,
    )

    status = "error" if error is not None else "ok"

    provider_data = {}

    if metadata is not None:
        provider_data = {
            "provider": "groq",
            "model": provider.model if provider else None,
            "latency_ms": metadata.elapsed_ms,
            "input_tokens": metadata.input_tokens,
            "output_tokens": metadata.output_tokens,
            "total_tokens": metadata.total_tokens,
            "finish_reason": metadata.finish_reason,
            "provider_request_id": metadata.provider_request_id,
            "rate_limit_headers": metadata.rate_limit_headers,
        }

    csv_record = {
        "request_id": request_id,
        "scenario": "rag_answer_eval",
        "source": "real",
        "start_time": start_time.isoformat(),
        "first_token_ms": (
            round(first_token_ms, 2)
            if first_token_ms is not None
            else ""
        ),
        "total_ms": total_ms,
        "status": status,
        "retries": 0,
        "tokens_in": provider_data.get("input_tokens", ""),
        "tokens_out": provider_data.get("output_tokens", ""),
        "retrieval_setting": setting.upper(),
        "answer_grade": "",
    }

    trace = {
        "request_id": request_id,
        "scenario": "rag_answer_eval",
        "source": "real",
        "start_time": start_time.isoformat(),
        "attempt": 1,
        "retries": 0,
        "status": status,
        "first_token_ms": first_token_ms,
        "total_ms": total_ms,
        "provider": provider_data.get("provider"),
        "model": provider_data.get("model"),
        "provider_request_id": provider_data.get(
            "provider_request_id"
        ),
        "provider_latency_ms": provider_data.get("latency_ms"),
        "tokens_in": provider_data.get("input_tokens"),
        "tokens_out": provider_data.get("output_tokens"),
        "total_tokens": provider_data.get("total_tokens"),
        "finish_reason": provider_data.get("finish_reason"),
        "rate_limit_headers": provider_data.get(
            "rate_limit_headers"
        ),
        "pipeline_status": pipeline_status,
        "answer": answer,
        "reason": reason,
        "citations": citations,
        "retrieval": retrieval,
        "error_type": type(error).__name__ if error else None,
        "error_message": str(error) if error else None,
        "error_traceback": (
            "".join(traceback.format_exception(error))
            if error else None
        ),
        "content_events": event_count,
    }

    evaluation_record = {
        "request_id": request_id,
        "question_id": question["id"],
        "question": question["question"],
        "expected_evidence": question["expected_evidence"],
        "answerable": question["answerable"],
        "setting": setting.upper(),
        "answer_grade": "pending_manual_review",
        "pipeline_status": pipeline_status,
        "answer": answer,
        "reason": reason,
        "citation_ids": citation_ids,
        "citations": citations,
        "retrieval": retrieval,
        "provider": provider_data,
        "first_token_ms": first_token_ms,
        "total_ms": total_ms,
        "error": str(error) if error else None,
    }

    append_csv(csv_record)
    append_jsonl(TRACE_FILE, trace)
    append_jsonl(EVALUATION_FILE, evaluation_record)

    print("\n" + "=" * 65)
    print("QUESTION:", question["id"])
    print("SETTING:", setting.upper())
    print("REQUEST ID:", request_id)
    print("TRANSPORT STATUS:", status)
    print("PIPELINE STATUS:", pipeline_status)
    print("FIRST TOKEN MS:", first_token_ms)
    print("TOTAL MS:", total_ms)
    print("INPUT TOKENS:", provider_data.get("input_tokens"))
    print("OUTPUT TOKENS:", provider_data.get("output_tokens"))
    print("FINISH REASON:", provider_data.get("finish_reason"))
    print("REASON:", reason)
    print("\nANSWER:\n", answer)
    print("\nCITATIONS:", citations)
    print("=" * 65)

    return request_id


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question",
        required=True,
        help="Question ID, e.g. ATT-01",
    )

    parser.add_argument(
        "--setting",
        required=True,
        choices=["a", "b", "c"],
    )

    args = parser.parse_args()

    question = next(
        (
            item for item in QUESTIONS
            if item["id"] == args.question.upper()
        ),
        None,
    )

    if question is None:
        raise SystemExit(
            f"Unknown question ID: {args.question}"
        )

    print("Starting one real Groq RAG evaluation request.")
    print("Question:", question["id"])
    print("Setting:", args.setting.upper())
    print("No automatic retries.")
    print("Answer grade will remain pending manual review.")

    evaluate_one(question, args.setting)


if __name__ == "__main__":
    main()
