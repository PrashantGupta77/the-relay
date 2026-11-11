import json
from collections import defaultdict
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = ROOT / "data" / "chunks"

encoding = tiktoken.get_encoding("o200k_harmony")

EXPECTED = {
    "setting_a": {"setting": "A", "chunk_size": 300, "overlap": 50},
    "setting_b": {"setting": "B", "chunk_size": 600, "overlap": 80},
    "setting_c": {"setting": "C", "chunk_size": 900, "overlap": 120},
}

PAPERS = {
    "attention_is_all_you_need": ("1706.03762", "v7"),
    "dynamic_chunking": ("2507.07955", "v2"),
    "moshi": ("2410.00037", "v1"),
}


def load_chunks(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {path}, line {line_number}: {exc}"
                    )
    return records


def validate_setting(filename, expected):
    path = CHUNK_DIR / f"{filename}.jsonl"
    print("\n" + "=" * 75)
    print(f"VALIDATING {filename.upper()}")
    print("=" * 75)

    if not path.exists():
        print("FAIL: File does not exist")
        return

    chunks = load_chunks(path)
    errors = []
    warnings = []

    seen_ids = set()
    groups = defaultdict(list)
    token_counts = []
    tiny_chunks = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "<missing-id>")
        text = chunk.get("text", "")

        # 1. Unique IDs
        if chunk_id in seen_ids:
            errors.append(f"{chunk_id}: duplicate chunk_id")
        seen_ids.add(chunk_id)

        # 2. Independent token count
        actual_tokens = len(
            encoding.encode(text, disallowed_special=())
        )
        recorded_tokens = chunk.get("token_count")

        if recorded_tokens != actual_tokens:
            errors.append(
                f"{chunk_id}: token_count says {recorded_tokens}, "
                f"actual count is {actual_tokens}"
            )

        token_counts.append(actual_tokens)

        # 3. Configured token budget
        if actual_tokens > expected["chunk_size"]:
            errors.append(
                f"{chunk_id}: {actual_tokens} tokens exceeds "
                f"budget {expected['chunk_size']}"
            )

        if not text.strip():
            errors.append(f"{chunk_id}: empty text")

        if actual_tokens <= 10:
            tiny_chunks.append((chunk_id, actual_tokens))

        # 4. Character count
        if chunk.get("character_count") != len(text):
            errors.append(
                f"{chunk_id}: character_count does not match text length"
            )

        # 5. Configuration metadata
        for field in ("setting", "chunk_size", "overlap"):
            if chunk.get(field) != expected[field]:
                errors.append(
                    f"{chunk_id}: incorrect {field}: {chunk.get(field)}"
                )

        if chunk.get("tokenizer") != "o200k_harmony":
            errors.append(f"{chunk_id}: unexpected tokenizer")

        # 6. Paper identity and exact version
        paper_id = chunk.get("paper_id")
        if paper_id not in PAPERS:
            errors.append(f"{chunk_id}: unknown paper_id {paper_id}")
        else:
            expected_arxiv, expected_version = PAPERS[paper_id]
            if chunk.get("arxiv_id") != expected_arxiv:
                errors.append(f"{chunk_id}: incorrect arXiv ID")
            if chunk.get("arxiv_version") != expected_version:
                errors.append(f"{chunk_id}: incorrect arXiv version")

        # 7. Page/source-part consistency
        source_parts = chunk.get("source_parts") or []
        source_pages = sorted({
            part["page_number"]
            for part in source_parts
            if "page_number" in part
        })
        chunk_pages = chunk.get("page_numbers") or []

        if not source_parts:
            errors.append(f"{chunk_id}: source_parts is empty")

        if chunk_pages != source_pages:
            errors.append(
                f"{chunk_id}: page_numbers do not match source_parts"
            )

        if chunk_pages:
            if chunk.get("page_start") != min(chunk_pages):
                errors.append(f"{chunk_id}: incorrect page_start")
            if chunk.get("page_end") != max(chunk_pages):
                errors.append(f"{chunk_id}: incorrect page_end")

        # 8. Character ranges
        start = chunk.get("start_char")
        end = chunk.get("end_char")

        if not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"{chunk_id}: invalid character range")
        elif start < 0 or end <= start:
            errors.append(f"{chunk_id}: invalid range {start}:{end}")
        elif end - start != len(text):
            errors.append(
                f"{chunk_id}: range length {end-start} "
                f"does not match text length {len(text)}"
            )

        # Group chunks for coverage checks
        groups[chunk.get("group_id")].append(chunk)

    # 9. Check character-range continuity within each source group
    gaps = []
    overlaps = []

    for group_id, group_chunks in groups.items():
        ordered = sorted(group_chunks, key=lambda c: c["start_char"])
        covered_until = 0

        for chunk in ordered:
            start = chunk["start_char"]
            end = chunk["end_char"]

            if start > covered_until:
                gaps.append(
                    (group_id, covered_until, start, start - covered_until)
                )
            elif start < covered_until:
                overlaps.append(
                    (group_id, start, covered_until)
                )

            covered_until = max(covered_until, end)

    # 10. Summary
    print(f"Chunks: {len(chunks)}")
    print(f"Groups: {len(groups)}")

    if token_counts:
        print(
            f"Actual token range: {min(token_counts)}–{max(token_counts)}"
        )
        print(
            f"Average actual tokens: {sum(token_counts)/len(token_counts):.2f}"
        )

    print(f"Tiny chunks (<=10 tokens): {len(tiny_chunks)}")
    for item in tiny_chunks[:10]:
        print("  TINY:", item)

    print(f"Character coverage gaps between chunks: {len(gaps)}")
    for item in gaps[:10]:
        print("  GAP:", item)

    print(f"Overlapping character ranges: {len(overlaps)}")

    print(f"\nErrors: {len(errors)}")
    for error in errors[:30]:
        print("  FAIL:", error)

    if len(errors) > 30:
        print(f"  ... and {len(errors)-30} more errors")

    if gaps:
        warnings.append(
            "Character gaps detected between adjacent chunks"
        )

    if tiny_chunks:
        warnings.append(
            "Very small chunks need inspection; they may be trailing fragments"
        )

    print("\nWarnings:")
    if warnings:
        for warning in warnings:
            print("  WARNING:", warning)
    else:
        print("  None")

    print("\nRESULT:", "PASS" if not errors else "FAIL")


def main():
    print("=" * 75)
    print("THE RELAY - INDEPENDENT CHUNK VALIDATION")
    print("Tokenizer: o200k_harmony")
    print("=" * 75)

    for filename, expected in EXPECTED.items():
        validate_setting(filename, expected)


if __name__ == "__main__":
    main()
