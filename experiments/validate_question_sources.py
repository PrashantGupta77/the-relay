import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.questions import QUESTIONS

FILES = {
    "attention_is_all_you_need": "attention_is_all_you_need.jsonl",
    "moshi": "moshi.jsonl",
    "dynamic_chunking": "dynamic_chunking.jsonl",
}

source_map = {}

for paper_id, filename in FILES.items():
    path = PROJECT_ROOT / "data" / "sectioned_pages" / filename
    sections = defaultdict(set)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)

            for segment in page["section_segments"]:
                number = segment.get("section_number")
                title = segment.get("section_title")

                if number:
                    sections[(number, title)].add(page["page_number"])

    source_map[paper_id] = sections

print("=" * 85)
print("QUESTION SOURCE VALIDATION")
print("=" * 85)

errors = []

for q in QUESTIONS:
    print(f"\n{q['id']}: {q['question']}")

    if not q["answerable"]:
        print("  Unanswerable question: no section/page required")
        continue

    if q["paper_id"] is None:
        print("  Cross-paper question: requires evidence from both papers")
        continue

    sections = source_map[q["paper_id"]]
    expected_section = q["expected_section"]

    matches = [
        (key, sorted(pages))
        for key, pages in sections.items()
        if (
            key[0] == expected_section
            or key[0].startswith(expected_section + ".")
        )
    ]

    if not matches:
        print("  ERROR: Expected section not found:", q["expected_section"])
        errors.append(q["id"])
        continue

    actual_pages = set()
    for (number, title), pages in matches:
        actual_pages.update(pages)
        print(f"  Section: {number} {title}")
        print(f"  Actual PDF pages: {sorted(pages)}")

    expected_pages = set(q["expected_pages"] or [])
    missing = expected_pages - actual_pages

    if missing:
        print("  WARNING: Expected pages not found in section:", sorted(missing))
        errors.append(q["id"])
    else:
        print("  Expected pages: verified")

print("\n" + "=" * 85)
print("QUESTIONS WITH SOURCE MISMATCHES:", errors)
print("VALIDATION:", "CHECK REQUIRED" if errors else "PASS")
print("=" * 85)
