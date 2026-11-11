import json
from pathlib import Path


CLEANED_DIR = Path("data/cleaned_pages")
SECTIONED_DIR = Path("data/sectioned_pages")

PAPERS = [
    "attention_is_all_you_need",
    "dynamic_chunking",
    "moshi",
]


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def verify_paper(paper_id):
    original_path = CLEANED_DIR / f"{paper_id}.jsonl"
    sectioned_path = SECTIONED_DIR / f"{paper_id}.jsonl"

    original_pages = load_jsonl(original_path)
    sectioned_pages = load_jsonl(sectioned_path)

    checks = {
        "page_count_matches": (
            len(original_pages) == len(sectioned_pages)
        ),
        "original_text_preserved": True,
        "retrieval_text_preserved": True,
        "all_text_covered_by_segments": True,
        "section_metadata_present": True,
    }

    errors = []

    for original, sectioned in zip(original_pages, sectioned_pages):
        page_number = original.get("page_number")

        if original.get("text") != sectioned.get("text"):
            checks["original_text_preserved"] = False
            errors.append(f"Page {page_number}: original text changed")

        original_text = original.get("retrieval_text", "")
        sectioned_text = sectioned.get("retrieval_text", "")

        if original_text != sectioned_text:
            checks["retrieval_text_preserved"] = False
            errors.append(f"Page {page_number}: retrieval text changed")

        segments = sectioned.get("section_segments", [])

        # Compare the complete word sequence, ignoring whitespace
        # differences introduced by segment boundary trimming.
        original_words = original_text.split()
        segmented_words = " ".join(
            segment.get("text", "")
            for segment in segments
        ).split()

        if original_words != segmented_words:
            checks["all_text_covered_by_segments"] = False
            errors.append(
                f"Page {page_number}: segment text does not match page text"
            )

        if not isinstance(segments, list):
            checks["section_metadata_present"] = False
            errors.append(f"Page {page_number}: segments is not a list")
            continue

        for segment in segments:
            if "section_path" not in segment:
                checks["section_metadata_present"] = False
                errors.append(f"Page {page_number}: section_path field missing")
                break

            if segment.get("section_number") and segment.get("section_title"):
                expected_heading = (
                    f"{segment['section_number']} {segment['section_title']}"
                )

                if segment.get("section_heading") != expected_heading:
                    checks["section_metadata_present"] = False
                    errors.append(
                        f"Page {page_number}: normalized section heading mismatch"
                    )
                    break

    passed = all(checks.values())

    print("\n" + "=" * 72)
    print(f"PAPER: {paper_id}")
    print("=" * 72)

    for name, result in checks.items():
        print(f"{name}: {'PASS' if result else 'FAIL'}")

    if errors:
        print("\nErrors:")
        for error in errors[:20]:
            print("-", error)

    print(f"\nOverall: {'PASS' if passed else 'FAIL'}")

    return passed


def main():
    results = {
        paper_id: verify_paper(paper_id)
        for paper_id in PAPERS
    }

    print("\n" + "=" * 72)
    print("FINAL VALIDATION")
    print("=" * 72)

    for paper_id, passed in results.items():
        print(f"{paper_id}: {'PASS' if passed else 'FAIL'}")

    print(
        "\nALL PAPERS PASSED"
        if all(results.values())
        else "\nVALIDATION NEEDS ATTENTION"
    )


if __name__ == "__main__":
    main()
