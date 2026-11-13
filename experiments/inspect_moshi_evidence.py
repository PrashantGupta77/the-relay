import json
from pathlib import Path

chunks_path = Path("data/chunks/setting_b.jsonl")
results_path = Path("results/retrieval_benchmark_raw.jsonl")

chunks = [
    json.loads(line)
    for line in chunks_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

records = [
    json.loads(line)
    for line in results_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

targets = {"3.3.1", "3.3.2", "3.4.1", "4.4", "6.1", "6.2", "6.3", "6.4"}

print("=" * 85)
print("PART 1 - ORIGINAL MOSHI EVIDENCE CHUNKS")
print("=" * 85)

for chunk in chunks:
    if chunk.get("paper_id") != "moshi":
        continue

    if chunk.get("section_number") not in targets:
        continue

    print("\n" + "-" * 85)
    print("ID:", chunk.get("chunk_id"))
    print("SECTION:", chunk.get("section_heading"))
    print("PAGES:", chunk.get("page_start"), "-", chunk.get("page_end"))
    print("TOKENS:", chunk.get("token_count"))
    print((chunk.get("text") or "")[:2200])

print("\n" + "=" * 85)
print("PART 2 - BASELINE B RETRIEVED RESULTS")
print("=" * 85)

for record in records:
    if record.get("setting") != "b":
        continue

    if record.get("question_id") not in {
        "MOS-02", "MOS-03", "MOS-04", "MOS-05", "MOS-06"
    }:
        continue

    print("\n" + "-" * 85)
    print(record["question_id"], "|", record["question"])
    print("Expected section:", record.get("expected_section"))

    for run in record.get("paper_runs", []):
        for result in run.get("results", []):
            print(
                f"\nRank {result['rank']} | "
                f"Score {result['score']:.4f} | "
                f"{result['chunk_id']} | "
                f"{result.get('section_heading')}"
            )
            print((result.get("text") or "")[:700])

print("\n" + "=" * 85)
print("INSPECTION COMPLETE - NO SOURCE FILES MODIFIED")
print("=" * 85)
