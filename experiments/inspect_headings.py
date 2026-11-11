import json
import re
from pathlib import Path

INPUT_DIR = Path("data/cleaned_pages")

PAPER_FILES = [
    "attention_is_all_you_need.jsonl",
    "moshi.jsonl",
    "dynamic_chunking.jsonl",
]

PATTERNS = [
    re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{2,120}$"),
    re.compile(r"^[A-Z]\.\s+[A-Z][^\n]{2,120}$"),
    re.compile(
        r"^(ABSTRACT|REFERENCES|ACKNOWLEDGMENTS|"
        r"ACKNOWLEDGEMENTS|APPENDIX(?:\s+[A-Z])?)$",
        re.IGNORECASE,
    ),
]


def is_heading_candidate(line):
    line = line.strip()

    if not line or len(line) > 120:
        return False

    return any(pattern.match(line) for pattern in PATTERNS)


def inspect_file(filename):
    path = INPUT_DIR / filename
    candidates = []

    with path.open("r", encoding="utf-8") as file:
        for record_line in file:
            record = json.loads(record_line)
            text = record["retrieval_text"]

            for line in text.splitlines():
                if is_heading_candidate(line):
                    candidates.append({
                        "page": record["page_number"],
                        "text": line.strip(),
                    })

    print("\n" + "=" * 75)
    print("PAPER:", filename)
    print("Heading candidates:", len(candidates))
    print("=" * 75)

    for item in candidates:
        print(f"Page {item['page']:>2}: {item['text']}")


def main():
    for filename in PAPER_FILES:
        inspect_file(filename)


if __name__ == "__main__":
    main()
