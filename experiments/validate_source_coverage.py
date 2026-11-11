import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.chunker import (
    INPUT_FILES,
    load_jsonl,
    group_consecutive_segments,
    prepare_group_text,
    find_source_parts,
)

CHUNK_DIR = Path("data/chunks")

SETTINGS = {
    "setting_a": "A",
    "setting_b": "B",
    "setting_c": "C",
}


def validate_setting(filename, setting_name):
    print("\n" + "=" * 75)
    print(f"SOURCE COVERAGE: {filename.upper()}")
    print("=" * 75)

    chunk_path = CHUNK_DIR / f"{filename}.jsonl"

    chunks_by_group = defaultdict(list)
    errors = []

    # Load generated chunks.
    with chunk_path.open("r", encoding="utf-8") as f:
        import json

        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            chunk = json.loads(line)

            if chunk["setting"] != setting_name:
                errors.append(
                    f"Line {line_number}: incorrect setting"
                )

            chunks_by_group[chunk["group_id"]].append(chunk)

    total_groups = 0
    total_source_chars = 0
    total_covered_chars = 0
    total_overlap_chars = 0
    total_chunks = 0
    groups_with_gaps = 0

    expected_group_ids = set()

    # Reconstruct original groups independently.
    for paper_id, input_path in INPUT_FILES.items():
        pages = load_jsonl(input_path)
        groups = group_consecutive_segments(pages)

        for group_index, group in enumerate(groups):
            total_groups += 1

            group_id = f"{paper_id}_group_{group_index:04d}"
            expected_group_ids.add(group_id)

            full_text, source_ranges = prepare_group_text(group)
            source_length = len(full_text)

            group_chunks = chunks_by_group.get(group_id, [])
            group_chunks.sort(key=lambda c: c["start_char"])

            if not group_chunks:
                errors.append(
                    f"{group_id}: no generated chunks"
                )
                groups_with_gaps += 1
                continue

            total_chunks += len(group_chunks)
            total_source_chars += source_length

            # Verify each chunk against original source text.
            for chunk in group_chunks:
                start = chunk["start_char"]
                end = chunk["end_char"]

                if start < 0 or end > source_length or end <= start:
                    errors.append(
                        f"{chunk['chunk_id']}: invalid source range "
                        f"{start}:{end}, source length={source_length}"
                    )
                    continue

                expected_text = full_text[start:end]

                if chunk["text"] != expected_text:
                    errors.append(
                        f"{chunk['chunk_id']}: text is not an exact "
                        "substring of original source group"
                    )

                expected_parts = find_source_parts(
                    start,
                    end,
                    source_ranges,
                )

                if chunk["source_parts"] != expected_parts:
                    errors.append(
                        f"{chunk['chunk_id']}: source_parts mismatch"
                    )

                expected_pages = sorted({
                    part["page_number"]
                    for part in expected_parts
                })

                if chunk["page_numbers"] != expected_pages:
                    errors.append(
                        f"{chunk['chunk_id']}: page_numbers mismatch"
                    )

                if chunk["group_id"] != group_id:
                    errors.append(
                        f"{chunk['chunk_id']}: wrong group_id"
                    )

            # Verify beginning and ending boundaries.
            if group_chunks[0]["start_char"] != 0:
                errors.append(
                    f"{group_id}: first chunk does not start at 0"
                )

            if group_chunks[-1]["end_char"] != source_length:
                errors.append(
                    f"{group_id}: last chunk ends at "
                    f"{group_chunks[-1]['end_char']}, "
                    f"expected {source_length}"
                )

            # Calculate union coverage and overlap.
            covered_until = 0
            group_gaps = []

            for chunk in group_chunks:
                start = chunk["start_char"]
                end = chunk["end_char"]

                if start > covered_until:
                    group_gaps.append(
                        (covered_until, start)
                    )
                elif start < covered_until:
                    total_overlap_chars += min(
                        covered_until, end
                    ) - start

                covered_until = max(covered_until, end)

            if group_gaps or covered_until < source_length:
                groups_with_gaps += 1

                for gap_start, gap_end in group_gaps:
                    errors.append(
                        f"{group_id}: missing characters "
                        f"{gap_start}:{gap_end}"
                    )

                if covered_until < source_length:
                    errors.append(
                        f"{group_id}: trailing source gap "
                        f"{covered_until}:{source_length}"
                    )

            # Count source characters covered by the union.
            total_covered_chars += min(
                covered_until, source_length
            )

    # Detect unexpected group IDs.
    actual_group_ids = set(chunks_by_group)
    unexpected_groups = actual_group_ids - expected_group_ids

    for group_id in sorted(unexpected_groups):
        errors.append(f"Unexpected group_id: {group_id}")

    print(f"Original groups: {total_groups}")
    print(f"Generated chunks: {total_chunks}")
    print(f"Original source characters: {total_source_chars}")
    print(f"Characters covered by chunk union: {total_covered_chars}")
    print(f"Overlapping characters: {total_overlap_chars}")
    print(f"Groups with coverage gaps: {groups_with_gaps}")

    if total_source_chars:
        coverage = (
            total_covered_chars / total_source_chars
        ) * 100

        print(f"Source coverage: {coverage:.4f}%")

    print(f"\nErrors: {len(errors)}")

    for error in errors[:40]:
        print("  FAIL:", error)

    if len(errors) > 40:
        print(f"  ... and {len(errors)-40} more errors")

    print(
        "\nRESULT:",
        "PASS" if not errors else "FAIL"
    )


def main():
    print("=" * 75)
    print("THE RELAY - ORIGINAL SOURCE COVERAGE VALIDATION")
    print("=" * 75)

    for filename, setting_name in SETTINGS.items():
        validate_setting(filename, setting_name)


if __name__ == "__main__":
    main()
