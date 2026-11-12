from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


INPUT_PATH = Path("results/retrieval_benchmark_raw.jsonl")
OUTPUT_PATH = Path("results/retrieval_manual_grading.csv")

FIELDNAMES = [
    "question_id",
    "question",
    "answerable",
    "expected_paper_id",
    "expected_section",
    "expected_pages",
    "expected_evidence",
    "setting",
    "chunk_size",
    "top_k",
    "reranking",
    "paper_filter",
    "rank",
    "retrieval_score",
    "chunk_id",
    "retrieved_paper_id",
    "retrieved_section_number",
    "retrieved_section_heading",
    "retrieved_pages",
    "citation",
    "retrieved_text",
    "relevance_grade",
    "covered_facets",
    "grading_notes",
]


def compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {INPUT_PATH}\n"
            "Run this script from the project root."
        )

    records = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rows: list[dict[str, Any]] = []

    for record in records:
        for paper_run in record.get("paper_runs", []):
            for result in paper_run.get("results", []):
                metadata = result.get("metadata") or {}

                # Some benchmark versions place metadata fields directly
                # on the result; support both layouts.
                def get_value(*keys: str) -> Any:
                    for key in keys:
                        if key in result and result[key] is not None:
                            return result[key]
                        if key in metadata and metadata[key] is not None:
                            return metadata[key]
                    return None

                pages = get_value("page_numbers")
                if pages is None:
                    start = get_value("page_start")
                    end = get_value("page_end")
                    pages = (
                        str(start)
                        if start == end and start is not None
                        else f"{start}-{end}"
                        if start is not None or end is not None
                        else ""
                    )

                rows.append({
                    "question_id": record.get("question_id", ""),
                    "question": record.get("question", ""),
                    "answerable": record.get("answerable", ""),
                    "expected_paper_id": record.get("expected_paper_id", ""),
                    "expected_section": record.get("expected_section", ""),
                    "expected_pages": compact(record.get("expected_pages")),
                    "expected_evidence": compact(record.get("expected_evidence")),
                    "setting": record.get("setting", ""),
                    "chunk_size": record.get("chunk_size", ""),
                    "top_k": record.get("top_k", ""),
                    "reranking": record.get("reranking", ""),
                    "paper_filter": paper_run.get("paper_filter", ""),
                    "rank": result.get("rank", ""),
                    "retrieval_score": result.get("score", ""),
                    "chunk_id": get_value("chunk_id") or "",
                    "retrieved_paper_id": get_value("paper_id") or "",
                    "retrieved_section_number": get_value("section_number") or "",
                    "retrieved_section_heading": (
                        get_value("section_heading", "section_title") or ""
                    ),
                    "retrieved_pages": compact(pages),
                    "citation": get_value("citation") or "",
                    "retrieved_text": result.get("text", ""),
                    # Fill these three columns manually.
                    "relevance_grade": "",
                    "covered_facets": "",
                    "grading_notes": "",
                })

    if not rows:
        raise ValueError(
            "No retrieved chunks were found. Check the paper_runs/results "
            "structure in the raw benchmark."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    question_count = len({row["question_id"] for row in rows})
    setting_count = len({row["setting"] for row in rows})

    print("=" * 76)
    print("MANUAL RETRIEVAL GRADING TEMPLATE CREATED")
    print("=" * 76)
    print(f"Input records:       {len(records)}")
    print(f"Questions:           {question_count}")
    print(f"Settings:            {setting_count}")
    print(f"Retrieved chunks:    {len(rows)}")
    print(f"Output:              {OUTPUT_PATH}")
    print()
    print("The raw benchmark was read only; it was not modified.")
    print("Open the CSV in Excel and fill:")
    print("  relevance_grade: 0 / 1 / 2")
    print("  covered_facets: concise evidence facts supported by this chunk")
    print("  grading_notes: optional explanation")
    print()
    print("Do not grade heading-only chunks as direct evidence.")
    print("=" * 76)


if __name__ == "__main__":
    main()
