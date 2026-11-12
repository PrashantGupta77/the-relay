import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "results" / "retrieval_benchmark_raw.jsonl"

with INPUT.open("r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

settings = ["a", "b", "c"]
eligible = [
    r for r in records
    if r["answerable"]
    and r["expected_paper_id"] is not None
    and r["expected_section"] is not None
]

print("=" * 78)
print("RETRIEVAL BASELINE - EXACT SECTION HIT DIAGNOSTIC")
print("=" * 78)
print("Raw records:", len(records))
print("Eligible single-paper answerable questions:", len(eligible) // 3)
print("Settings:", len(settings))
print("Metric: exact expected_section match within top 5")
print("Note: This is a section-match proxy, not semantic evidence grading.")

for setting in settings:
    setting_records = [r for r in eligible if r["setting"] == setting]
    hits = 0
    reciprocal_ranks = []

    print(f"\nSETTING {setting.upper()}")

    for record in setting_records:
        expected = record["expected_section"]
        paper_id = record["expected_paper_id"]

        matching_runs = [
            run for run in record["paper_runs"]
            if run["paper_filter"] == paper_id
        ]

        results = matching_runs[0]["results"] if matching_runs else []

        ranks = [
            item["rank"]
            for item in results
            if item["section_number"] == expected
        ]

        rank = min(ranks) if ranks else None

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        print(
            f"  {record['question_id']}: "
            f"expected={expected}, "
            f"first_exact_rank={rank if rank is not None else 'MISS'}"
        )

    total = len(setting_records)
    hit_rate = hits / total if total else 0
    mrr = sum(reciprocal_ranks) / total if total else 0

    print(f"\n  Exact Section Hit@5: {hits}/{total} = {hit_rate:.3f}")
    print(f"  Exact Section MRR@5: {mrr:.3f}")

print("\n" + "=" * 78)
print("EXCLUDED FROM THIS METRIC")
print("CROSS-01: requires evidence from both papers.")
print("UNANS-01: requires abstention, not section retrieval.")
print("=" * 78)
