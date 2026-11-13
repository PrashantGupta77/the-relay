import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNK_FILE = PROJECT_ROOT / "data" / "chunks" / "setting_b.jsonl"

TARGET_SECTIONS = {"2.2.1", "2.2.2", "2.2.3"}

TARGET_CHUNK_IDS = {
    "dynamic_chunking_B_0008_0001",
    "dynamic_chunking_B_0005_0000",
    "dynamic_chunking_B_0012_0005",
}


def main():
    with CHUNK_FILE.open("r", encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    print("=" * 80)
    print("DYNAMIC CHUNKING: SOURCE EVIDENCE INSPECTION")
    print("=" * 80)

    for chunk in chunks:
        if chunk["paper_id"] != "dynamic_chunking":
            continue

        section = chunk.get("section_number")
        chunk_id = chunk["chunk_id"]

        if section in TARGET_SECTIONS or chunk_id in TARGET_CHUNK_IDS:
            print("\n" + "-" * 80)
            print("Chunk ID:", chunk_id)
            print("Section:", chunk.get("section_heading"))
            print("Section path:", chunk.get("section_path"))
            print("Pages:", chunk.get("page_numbers"))
            print("Tokens:", chunk.get("token_count"))
            print("Characters:", chunk.get("character_count"))
            print("\nFULL CHUNK TEXT:\n")
            print(chunk["text"])
            print()


if __name__ == "__main__":
    main()
