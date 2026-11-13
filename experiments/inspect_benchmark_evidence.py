import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "retrieval_benchmark_raw.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "retrieval_evidence_review_setting_b.txt"
)

TARGET_IDS = {
    "MOS-03",
    "MOS-04",
    "MOS-05",
    "DYN-01",
    "DYN-02",
}

def main():
    records = []

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    selected = [
        record
        for record in records
        if (
            record["question_id"] in TARGET_IDS
            and record["setting"] == "b"
        )
    ]

    selected.sort(
        key=lambda item: item["question_id"]
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for record in selected:
            output.write("=" * 90 + "\n")
            output.write(
                f"QUESTION: {record['question_id']}\n"
            )
            output.write(
                f"TEXT: {record['question']}\n"
            )
            output.write(
                f"EXPECTED EVIDENCE: "
                f"{record['expected_evidence']}\n"
            )
            output.write("=" * 90 + "\n")

            for paper_run in record["paper_runs"]:
                output.write(
                    f"\nPAPER: {paper_run['paper_filter']}\n"
                )

                for result in paper_run["results"][:3]:
                    output.write("\n" + "-" * 80 + "\n")
                    output.write(
                        f"RANK: {result['rank']}\n"
                    )
                    output.write(
                        f"SCORE: {result['score']:.4f}\n"
                    )
                    output.write(
                        f"CHUNK: {result['chunk_id']}\n"
                    )
                    output.write(
                        f"CITATION: {result['citation']}\n"
                    )
                    output.write(
                        f"SECTION PATH: {result['section_path']}\n"
                    )
                    output.write("\nFULL SOURCE TEXT:\n")
                    output.write(result["text"])
                    output.write("\n")

    print("Questions selected:", len(selected))
    print("Evidence chunks per question: up to 3")
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
