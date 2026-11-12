import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.questions import QUESTIONS

INPUT_PATH = PROJECT_ROOT / "results" / "query_expansion_setting_b.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "query_expansion_analysis.csv"

METHODS = {
    "baseline": "baseline_results",
    "expanded_only": "expanded_query_results",
    "fused": "merged_results",
}


def get_target_rank(results, expected_section):
    matching_ranks = [
        item["rank"]
        for item in results
        if item.get("section_number") == expected_section
    ]

    return min(matching_ranks) if matching_ranks else None


def main():
    question_map = {q["id"]: q for q in QUESTIONS}

    records = [
        json.loads(line)
        for line in INPUT_PATH.open(encoding="utf-8")
        if line.strip()
    ]

    rows = []

    for record in records:
        question = question_map[record["question_id"]]
        expected_section = question["expected_section"]

        for paper_run in record["paper_runs"]:
            for method, result_key in METHODS.items():
                results = paper_run[result_key]

                target_rank = (
                    get_target_rank(results, expected_section)
                    if expected_section is not None
                    else None
                )

                rows.append({
                    "question_id": record["question_id"],
                    "paper_id": paper_run["paper_id"],
                    "method": method,
                    "expected_section": expected_section,
                    "target_rank": target_rank,
                    "hit_at_5": (
                        target_rank is not None
                        and target_rank <= 5
                    ) if expected_section is not None else None,
                    "answerable": question["answerable"],
                })

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question_id",
                "paper_id",
                "method",
                "expected_section",
                "target_rank",
                "hit_at_5",
                "answerable",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    eligible = [
        row for row in rows
        if row["expected_section"] is not None
        and row["answerable"]
    ]

    print("\n" + "=" * 75)
    print("QUERY EXPANSION - TARGET SECTION ANALYSIS")
    print("=" * 75)

    summaries = {}

    for method in METHODS:
        subset = [
            row for row in eligible
            if row["method"] == method
        ]

        hits = sum(row["hit_at_5"] for row in subset)
        total = len(subset)

        summaries[method] = {
            row["question_id"]: row["target_rank"]
            for row in subset
        }

        print(
            f"{method:15s}: "
            f"{hits}/{total} target sections in top 5 "
            f"({hits / total * 100:.1f}%)"
        )

    print("\nPer-question target ranks:")
    print("-" * 75)
    print(
        f"{'Question':12s}"
        f"{'Baseline':>12s}"
        f"{'Expanded':>12s}"
        f"{'Fused':>12s}"
    )

    question_ids = list(dict.fromkeys(
        row["question_id"] for row in eligible
    ))

    for question_id in question_ids:
        values = []

        for method in METHODS:
            rank = summaries[method].get(question_id)
            values.append(str(rank) if rank is not None else "MISS")

        print(
            f"{question_id:12s}"
            f"{values[0]:>12s}"
            f"{values[1]:>12s}"
            f"{values[2]:>12s}"
        )

    print("\nSaved:", OUTPUT_PATH)
    print("Eligible answerable question-paper pairs:", len(eligible))
    print("=" * 75)


if __name__ == "__main__":
    main()
