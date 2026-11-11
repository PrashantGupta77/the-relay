import json
from pathlib import Path

RAW_DIR = Path("data/raw_pages")
CLEAN_DIR = Path("data/cleaned_pages")

FILES = [
    "attention_is_all_you_need.jsonl",
    "moshi.jsonl",
    "dynamic_chunking.jsonl",
]

MOSHI_HEADER = (
    "Moshi: a speech-text foundation model for real-time dialogue"
)


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def verify_file(filename):
    raw_records = load_jsonl(RAW_DIR / filename)
    clean_records = load_jsonl(CLEAN_DIR / filename)

    assert len(raw_records) == len(clean_records), (
        f"Page count mismatch in {filename}"
    )

    original_preserved = True
    retrieval_field_present = True
    page_mapping_preserved = True

    for raw, cleaned in zip(raw_records, clean_records):
        if raw["text"] != cleaned["text"]:
            original_preserved = False

        if "retrieval_text" not in cleaned:
            retrieval_field_present = False

        if (
            raw["paper_id"] != cleaned["paper_id"]
            or raw["page_number"] != cleaned["page_number"]
            or raw["source_filename"] != cleaned["source_filename"]
        ):
            page_mapping_preserved = False

    print(f"\n{filename}")
    print("Pages:", len(raw_records))
    print("Original text preserved:", original_preserved)
    print("Retrieval text present:", retrieval_field_present)
    print("Page mapping preserved:", page_mapping_preserved)

    if filename == "moshi.jsonl":
        page1 = clean_records[0]["retrieval_text"]
        later_pages = [
            record["retrieval_text"]
            for record in clean_records[1:]
        ]

        print(
            "Moshi title preserved on page 1:",
            MOSHI_HEADER in page1,
        )
        print(
            "Moshi running header removed from later pages:",
            all(MOSHI_HEADER not in text for text in later_pages),
        )

    assert original_preserved
    assert retrieval_field_present
    assert page_mapping_preserved


def main():
    for filename in FILES:
        verify_file(filename)

    print("\nAll cleaning verification checks passed.")


if __name__ == "__main__":
    main()
