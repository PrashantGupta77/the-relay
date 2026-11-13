import json
from pathlib import Path

path = Path("data/sectioned_pages/moshi.jsonl")

records = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

record = next(
    r for r in records
    if r.get("page_number") == 7
)

print("=" * 90)
print("PAGE 7 - SECTION METADATA")
print("=" * 90)

print("Page:", record.get("page_number"))
print("Section:", record.get("section"))
print("Parser version:", record.get("section_parser_version"))
print("Parser source:", record.get("section_parser_source"))

segments = record.get("section_segments", [])

print("\nNumber of section segments:", len(segments))

for i, segment in enumerate(segments):
    print("\n" + "-" * 90)
    print("SEGMENT:", i)
    print("TYPE:", type(segment).__name__)

    if isinstance(segment, dict):
        for key, value in segment.items():
            if isinstance(value, str):
                print(f"{key}: {value[:1800]}")
            else:
                print(f"{key}: {value}")
    else:
        print(str(segment)[:2500])

print("\n" + "=" * 90)
print("PAGE 7 COMPLETE")
print("=" * 90)
