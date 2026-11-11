import json
from pathlib import Path

checks = {
    "dynamic_chunking": ("2.1.1",),
    "attention_is_all_you_need": ("3.2.3",),
}

for paper_id, target_numbers in checks.items():
    path = Path("data/sectioned_pages") / f"{paper_id}.jsonl"

    print(f"\nPAPER: {paper_id}")

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            page = json.loads(line)

            for segment in page.get("section_segments", []):
                if segment.get("section_number") in target_numbers:
                    print("Page:", page.get("page_number"))
                    print("Section number:", repr(segment.get("section_number")))
                    print("Section title:", repr(segment.get("section_title")))
                    print("Section heading:", repr(segment.get("section_heading")))
                    print("Section path:", repr(segment.get("section_path")))
