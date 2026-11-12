import json
import statistics
import sys

from pathlib import Path
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.embedding_windows import (
    MAX_CONTENT_TOKENS,
    WINDOW_OVERLAP_TOKENS,
    split_text_into_windows,
    validate_windows,
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_FILES = {
    "A": PROJECT_ROOT / "data/chunks/setting_a.jsonl",
    "B": PROJECT_ROOT / "data/chunks/setting_b.jsonl",
    "C": PROJECT_ROOT / "data/chunks/setting_c.jsonl",
}


def main():
    print("=" * 70)
    print("EMBEDDING WINDOW VALIDATION")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    print("Tokenizer:", tokenizer.__class__.__name__)
    print("Fast tokenizer:", tokenizer.is_fast)
    print("Maximum content tokens:", MAX_CONTENT_TOKENS)
    print("Overlap tokens:", WINDOW_OVERLAP_TOKENS)

    total_chunks = 0
    total_windows = 0
    total_errors = 0

    for setting, path in CHUNK_FILES.items():
        print("\n" + "-" * 70)
        print(f"SETTING {setting}")
        print("-" * 70)

        if not path.exists():
            raise FileNotFoundError(path)

        chunk_count = 0
        window_count = 0
        window_token_counts = []
        errors = []

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                chunk = json.loads(line)
                chunk_count += 1

                windows = split_text_into_windows(
                    text=chunk["text"],
                    tokenizer=tokenizer,
                )

                validation = validate_windows(
                    source_text=chunk["text"],
                    windows=windows,
                )

                if not validation["valid"]:
                    errors.extend(
                        [
                            f"Line {line_number}, chunk {chunk['chunk_id']}: {e}"
                            for e in validation["errors"]
                        ]
                    )

                for window in windows:
                    window_token_counts.append(window["token_count"])

                window_count += len(windows)

        total_chunks += chunk_count
        total_windows += window_count
        total_errors += len(errors)

        print("Original chunks:", chunk_count)
        print("Generated windows:", window_count)

        if window_token_counts:
            print("Minimum window tokens:", min(window_token_counts))
            print("Maximum window tokens:", max(window_token_counts))
            print(
                "Average window tokens:",
                round(statistics.mean(window_token_counts), 2),
            )

        print("Validation errors:", len(errors))

        for error in errors[:10]:
            print("ERROR:", error)

        if errors:
            print("RESULT: FAIL")
        else:
            print("RESULT: PASS")

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print("Total original chunks:", total_chunks)
    print("Total embedding windows:", total_windows)
    print("Total validation errors:", total_errors)

    if total_errors:
        print("OVERALL RESULT: FAIL")
        raise SystemExit(1)

    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()
