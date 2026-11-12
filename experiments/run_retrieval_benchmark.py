import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.questions import QUESTIONS
from app.retrieval.embeddings import load_embedding_model
from app.retrieval.retriever import Retriever, format_citation


SETTINGS = ["a", "b", "c"]
TOP_K = 5

CROSS_PAPERS = [
    "attention_is_all_you_need",
    "dynamic_chunking",
]


def retrieve_for_paper(retriever, question, paper_id):
    response = retriever.retrieve(
        question=question,
        top_k=TOP_K,
        paper_id=paper_id,
    )

    results = []

    for item in response["results"]:
        results.append({
            **item,
            "citation": format_citation(item),
        })

    return {
        "paper_filter": paper_id,
        "result_count": len(results),
        "results": results,
    }


def main():
    print("=" * 85)
    print("THE RELAY - RETRIEVAL QUALITY BENCHMARK")
    print("=" * 85)

    print("\nLoading embedding model...")
    model = load_embedding_model()

    retrievers = {}

    for setting in SETTINGS:
        print(f"\nLoading Setting {setting.upper()}...")
        retrievers[setting] = Retriever(
            setting=setting,
            model=model,
        )

        print(
            f"  Index vectors: "
            f"{retrievers[setting].index.ntotal}"
        )

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "retrieval_benchmark_raw.jsonl"

    total_records = 0
    total_paper_calls = 0

    with output_path.open("w", encoding="utf-8") as output_file:

        for question_number, question_data in enumerate(
            QUESTIONS, start=1
        ):

            print("\n" + "-" * 85)
            print(
                f"QUESTION {question_number}/{len(QUESTIONS)}: "
                f"{question_data['id']}"
            )
            print(question_data["question"])

            for setting in SETTINGS:
                retriever = retrievers[setting]

                if question_data["id"] == "CROSS-01":
                    paper_ids = CROSS_PAPERS
                else:
                    paper_ids = [question_data["paper_id"]]

                paper_runs = []

                for paper_id in paper_ids:
                    run = retrieve_for_paper(
                        retriever,
                        question_data["question"],
                        paper_id,
                    )

                    paper_runs.append(run)
                    total_paper_calls += 1

                    print(
                        f"\n  Setting {setting.upper()} | "
                        f"{paper_id} | "
                        f"results={run['result_count']}"
                    )

                    for result in run["results"]:
                        print(
                            f"    Rank {result['rank']} | "
                            f"Score {result['score']:.4f} | "
                            f"{result['chunk_id']} | "
                            f"{result['section_heading']}"
                        )

                record = {
                    "benchmark_timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "question_id": question_data["id"],
                    "question": question_data["question"],
                    "expected_paper_id": question_data["paper_id"],
                    "expected_section": question_data["expected_section"],
                    "expected_pages": question_data["expected_pages"],
                    "answerable": question_data["answerable"],
                    "expected_evidence": question_data["expected_evidence"],
                    "setting": setting,
                    "chunk_size": {
                        "a": 300,
                        "b": 600,
                        "c": 900,
                    }[setting],
                    "top_k": TOP_K,
                    "reranking": False,
                    "paper_runs": paper_runs,
                    "manual_grade": "pending",
                    "manual_notes": None,
                }

                output_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                total_records += 1

    print("\n" + "=" * 85)
    print("BENCHMARK COMPLETED")
    print("=" * 85)
    print("Questions:", len(QUESTIONS))
    print("Settings:", len(SETTINGS))
    print("Benchmark records:", total_records)
    print("Individual paper retrieval calls:", total_paper_calls)
    print("Top K:", TOP_K)
    print("Reranking: Disabled")
    print("Manual grades: Pending")
    print("Saved:", output_path)
    print("=" * 85)


if __name__ == "__main__":
    main()
