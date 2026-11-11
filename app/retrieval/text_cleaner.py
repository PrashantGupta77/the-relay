import json
import re
from pathlib import Path

INPUT_DIR = Path("data/raw_pages")
OUTPUT_DIR = Path("data/cleaned_pages")

PAPER_FILES = [
    "attention_is_all_you_need.jsonl",
    "moshi.jsonl",
    "dynamic_chunking.jsonl",
]

MOSHI_RUNNING_HEADER = (
    "Moshi: a speech-text foundation model for real-time dialogue"
)


def clean_text(text: str, paper_id: str, page_number: int) -> str:
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Remove only the repeated Moshi running header.
        # Preserve the title on the first page.
        if (
            paper_id == "moshi"
            and page_number > 1
            and stripped == MOSHI_RUNNING_HEADER
        ):
            continue

        # Normalize horizontal whitespace without joining paragraphs.
        normalized = re.sub(r"[ \t]+", " ", line).strip()

        cleaned_lines.append(normalized)

    # Remove excessive blank lines while preserving paragraph breaks.
    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def process_file(filename: str):
    input_path = INPUT_DIR / filename
    output_path = OUTPUT_DIR / filename

    total_pages = 0
    total_original_chars = 0
    total_cleaned_chars = 0

    with input_path.open("r", encoding="utf-8") as source, \
            output_path.open("w", encoding="utf-8") as destination:

        for line in source:
            record = json.loads(line)

            original_text = record["text"]
            cleaned = clean_text(
                text=original_text,
                paper_id=record["paper_id"],
                page_number=record["page_number"],
            )

            # Keep all original fields and add a separate retrieval field.
            record["retrieval_text"] = cleaned
            record["retrieval_character_count"] = len(cleaned)

            destination.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            total_pages += 1
            total_original_chars += len(original_text)
            total_cleaned_chars += len(cleaned)

    print(f"\nPaper file: {filename}")
    print("Pages:", total_pages)
    print("Original characters:", total_original_chars)
    print("Cleaned characters:", total_cleaned_chars)
    print("Characters removed:", total_original_chars - total_cleaned_chars)
    print("Output:", output_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in PAPER_FILES:
        if not (INPUT_DIR / filename).exists():
            raise FileNotFoundError(INPUT_DIR / filename)

        process_file(filename)

    print("\nText cleaning completed.")


if __name__ == "__main__":
    main()
