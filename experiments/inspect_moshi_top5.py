import json

target_ids = {"MOS-03", "MOS-04", "MOS-05"}

with open("results/retrieval_benchmark_raw.jsonl", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

for record in records:
    if record.get("question_id") not in target_ids:
        continue
    if record.get("setting") != "b":
        continue

    print("\n" + "=" * 90)
    print(record.get("question_id"), "-", record.get("question"))

    for paper_run in record.get("paper_runs", []):
        print("\nPAPER:", paper_run.get("paper_id"))

        for result in paper_run.get("results", [])[:5]:
            print(
                "Rank:", result.get("rank"),
                "| Section:", result.get("section_heading"),
                "| Page:", result.get("page_numbers"),
                "| Chunk:", result.get("chunk_id"),
                "| Score:", result.get("score")
            )
