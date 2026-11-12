import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.embeddings import load_embedding_model
from app.retrieval.retriever import Retriever, format_citation


TEST_CASES = [
    {
        "question": "How does scaled dot-product attention compute attention weights?",
        "paper_id": "attention_is_all_you_need",
    },
    {
        "question": "What is the purpose of inner monologue in Moshi?",
        "paper_id": "moshi",
    },
    {
        "question": "How does dynamic chunking decide when to create a boundary?",
        "paper_id": "dynamic_chunking",
    },
]

SETTING = "b"
TOP_K = 3


def main():
    print("=" * 70)
    print("THE RELAY - QUERY-TIME RETRIEVAL SMOKE TEST")
    print("=" * 70)

    print("\nLoading embedding model...")
    model = load_embedding_model()

    retriever = Retriever(
        setting=SETTING,
        model=model,
    )

    saved_results = []

    for case_number, case in enumerate(TEST_CASES, start=1):
        print("\n" + "=" * 70)
        print(f"TEST CASE {case_number}")
        print("=" * 70)

        print("Question:", case["question"])
        print("Paper filter:", case["paper_id"])

        response = retriever.retrieve(
            question=case["question"],
            top_k=TOP_K,
            paper_id=case["paper_id"],
        )

        if response["result_count"] == 0:
            raise RuntimeError(
                f"No results returned for test case {case_number}"
            )

        returned_chunk_ids = [
            result["chunk_id"]
            for result in response["results"]
        ]

        if len(returned_chunk_ids) != len(set(returned_chunk_ids)):
            raise RuntimeError("Duplicate source chunks were returned.")

        for result in response["results"]:
            if result["paper_id"] != case["paper_id"]:
                raise RuntimeError("Paper filter was not respected.")

            if not result["text"].strip():
                raise RuntimeError("An empty source chunk was returned.")

            citation = format_citation(result)

            print("\nRank:", result["rank"])
            print("Score:", round(result["score"], 4))
            print("Chunk:", result["chunk_id"])
            print("Citation:", citation)
            print("Section path:", result["section_path"])
            print("Source text preview:")
            print(result["text"][:700].replace("\n", " "))

        saved_results.append(response)

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "retrieval_smoke_test.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            saved_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("RESULT: PASS")
    print("Setting:", SETTING.upper())
    print("Questions tested:", len(TEST_CASES))
    print("Top K:", TOP_K)
    print("Saved results:", output_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
