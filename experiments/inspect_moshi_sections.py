import json
from pathlib import Path
from collections import Counter

path = Path("data/chunks/setting_b.jsonl")

chunks = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

print("=" * 80)
print("SETTING B - CHUNK STRUCTURE INSPECTION")
print("=" * 80)

print("Total chunks:", len(chunks))

print("\nFIRST CHUNK KEYS:")
print(json.dumps(chunks[0], indent=2, ensure_ascii=False)[:3000])

moshi = [
    c for c in chunks
    if c.get("paper_id") == "moshi"
]

print("\nMOSHI CHUNKS:", len(moshi))

sections = Counter(
    (c.get("section_number"), c.get("section_title"))
    for c in moshi
)

print("\nMOSHI SECTION INVENTORY:")
for (number, title), count in sorted(
    sections.items(),
    key=lambda x: (str(x[0][0]), str(x[0][1]))
):
    if number and number.startswith(("3.", "4.", "6")):
        print(f"{number} | {title} | chunks={count}")

print("\nSELECTED SECTION TEXT SAMPLES:")

targets = {"3.1", "3.2", "3.3", "3.4.1", "3.4.4", "4.4", "6"}

for chunk in moshi:
    if chunk.get("section_number") in targets:
        print("\n" + "-" * 75)
        print("Chunk:", chunk.get("chunk_id"))
        print("Section:", chunk.get("section_heading"))
        print("Pages:", chunk.get("page_start"), "-", chunk.get("page_end"))
        print("Text:")
        print((chunk.get("text") or "")[:1200])

print("\n" + "=" * 80)
