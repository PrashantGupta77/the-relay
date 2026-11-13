import json
from pathlib import Path

path = Path("data/chunks/setting_b.jsonl")

target_ids = {
    "moshi_B_0004_0000",
    "moshi_B_0005_0000",
}

with path.open("r", encoding="utf-8") as file:
    for line in file:
        chunk = json.loads(line)

        if chunk["chunk_id"] in target_ids:
            print("\n" + "=" * 90)
            print("CHUNK:", chunk["chunk_id"])
            print("SECTION:", chunk.get("section_heading"))
            print("PAGES:", chunk.get("page_numbers"))
            print("=" * 90)
            print(chunk["text"])
