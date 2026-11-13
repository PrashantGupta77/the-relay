
import json
from pathlib import Path

CHUNK_FILE = Path("data/chunks/setting_b.jsonl")

TARGET_SECTIONS = {
    "3.3 Audio Tokenization",
    "3.4.1 Hierarchical autoregressive modeling with RQ-Transformer",
    "4.4 Training stages and hyper-parameters",
}

print("Reading:", CHUNK_FILE.resolve())

if not CHUNK_FILE.exists():
    raise FileNotFoundError(CHUNK_FILE)

matched = 0

with CHUNK_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)

        if record.get("paper_id") != "moshi":
            continue

        heading = record.get("section_heading")
        title = record.get("section_title")

        if heading not in TARGET_SECTIONS and title not in TARGET_SECTIONS:
            continue

        matched += 1

        print("\n" + "=" * 90)
        print("CHUNK:", record.get("chunk_id"))
        print("SECTION:", heading or title)
        print("PAGES:", record.get("page_numbers"))
        print("TOKENS:", record.get("token_count"))
        print("TEXT:\n", record.get("text", ""))

print("\n" + "=" * 90)
print("Matching chunks:", matched)
