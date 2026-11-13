import json
from pathlib import Path

CHUNK_FILE = Path("data/chunks/setting_b.jsonl")

# Top retrieved chunks from the Setting B benchmark.
RETRIEVED = {
    "DYN-01": [
        "dynamic_chunking_B_0008_0001",
        "dynamic_chunking_B_0005_0000",
        "dynamic_chunking_B_0012_0005",
    ],
    "DYN-02": [
        "dynamic_chunking_B_0001_0002",
        "dynamic_chunking_B_0012_0005",
        "dynamic_chunking_B_0001_0003",
    ],
    "MOS-03": [
        "moshi_B_0015_0002",
        "moshi_B_0004_0000",
        "moshi_B_0035_0024",
    ],
    "MOS-04": [
        "moshi_B_0001_0002",
        "moshi_B_0029_0006",
        "moshi_B_0015_0002",
    ],
    "MOS-05": [
        "moshi_B_0029_0006",
        "moshi_B_0000_0000",
        "moshi_B_0030_0000",
    ],
}

# Correct evidence chunks, identified from the source inspection.
EXPECTED = {
    "DYN-01": [
        "dynamic_chunking_B_0006_0000",
        "dynamic_chunking_B_0006_0001",
    ],
    "DYN-02": [
        "dynamic_chunking_B_0006_0000",
        "dynamic_chunking_B_0006_0001",
    ],
    "MOS-03": [
        "moshi_B_0008_0000",
    ],
    "MOS-04": [
        "moshi_B_0012_0000",
        "moshi_B_0012_0001",
        "moshi_B_0012_0002",
    ],
    "MOS-05": [
        "moshi_B_0020_0000",
        "moshi_B_0020_0001",
        "moshi_B_0020_0002",
    ],
}

wanted_ids = {
    chunk_id
    for ids in RETRIEVED.values()
    for chunk_id in ids
}
wanted_ids.update(
    chunk_id
    for ids in EXPECTED.values()
    for chunk_id in ids
)

chunks = {}

with CHUNK_FILE.open("r", encoding="utf-8") as file:
    for line in file:
        chunk = json.loads(line)
        chunk_id = chunk["chunk_id"]

        if chunk_id in wanted_ids:
            chunks[chunk_id] = chunk

for question_id in RETRIEVED:
    print("\n" + "#" * 100)
    print("QUESTION:", question_id)
    print("#" * 100)

    for category, mapping in [
        ("RETRIEVED TOP RESULTS", RETRIEVED),
        ("EXPECTED SOURCE EVIDENCE", EXPECTED),
    ]:
        print("\n" + "=" * 90)
        print(category)
        print("=" * 90)

        for chunk_id in mapping[question_id]:
            chunk = chunks.get(chunk_id)

            if chunk is None:
                print("\nMISSING CHUNK:", chunk_id)
                continue

            print("\n" + "-" * 90)
            print("CHUNK:", chunk_id)
            print("SECTION:", chunk.get("section_heading"))
            print("PAGES:", chunk.get("page_numbers"))
            print("TOKENS:", chunk.get("token_count"))
            print("-" * 90)
            print(chunk["text"])
