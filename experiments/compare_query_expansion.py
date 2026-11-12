import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.questions import QUESTIONS
from app.retrieval.embeddings import load_embedding_model
from app.retrieval.retriever import Retriever, format_citation


SETTING = "b"
TOP_K = 5

# Neutral alternate terminology, not answer text.
# Every question receives the same experimental treatment.
QUERY_EXPANSIONS = {
    "ATT-01": "attention score computation query key similarity scaling softmax",
    "ATT-02": "parallel attention heads representation subspaces attention mechanisms",
    "ATT-03": "token position information sequence order position representation",
    "ATT-04": "Transformer encoder decoder architecture layer composition",
    "ATT-05": "training optimization algorithm learning rate schedule",
    "ATT-06": "causal masking autoregressive decoder attention future tokens",

    "MOS-01": "text speech alignment spoken language generation",
    "MOS-02": "language model backbone text model architecture",
    "MOS-03": "audio representation discretization codec tokenization",
    "MOS-04": "autoregressive sequence modeling multiple streams codebooks temporal depth",
    "MOS-05": "training phases pretraining post-training finetuning hyperparameters",
    "MOS-06": "safety evaluation analysis toxicity memorization voice watermark",

    "DYN-01": "adaptive segmentation sequence boundary detection",
    "DYN-02": "H-Net routing network segmentation boundary detection",
    "DYN-03": "similarity score adjacent representations boundary probability equation",
    "DYN-04": "boundary decision threshold binary indicator",
    "DYN-05": "downsampling sequence reduction boundary selection",
    "DYN-06": "compression ratio regularization objective loss",

    "CROSS-01": "sequence modeling attention token-level processing hierarchical segmentation",
    "UNANS-01": "commercial API pricing cost per token usage charges",
}


def retrieve_paper(retriever, query, paper_id):
    response = retriever.retrieve(
        question=query,
        top_k=TOP_K,
        paper_id=paper_id,
    )

    results = []

    for item in response["results"]:
        results.append({
            **item,
            "citation": format_citation(item),
        })

    return results


def merge_multi_query_results(result_lists):
    """
    Merge results by original chunk ID.
    Retain the highest cosine score for each chunk.
    """
    best_by_chunk = {}

    for query_index, results in enumerate(result_lists):
        for item in results:
            chunk_id = item["chunk_id"]

            if (
                chunk_id not in best_by_chunk
                or item["score"] > best_by_chunk[chunk_id]["score"]
            ):
                best_by_chunk[chunk_id] = {
                    **item,
                    "matched_query_index": query_index,
                }

    ranked = sorted(
        best_by_chunk.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    for rank, item in enumerate(ranked[:TOP_K], start=1):
        item["rank"] = rank

    return ranked[:TOP_K]


def main():
    print("=" * 85)
    print("THE RELAY - QUERY EXPANSION EXPERIMENT")
    print("=" * 85)

    print("\nLoading embedding model...")
    model = load_embedding_model()

    print("\nLoading Setting B retriever...")
    retriever = Retriever(setting=SETTING, model=model)

    output_path = (
        PROJECT_ROOT
        / "results"
        / "query_expansion_setting_b.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0

    with output_path.open("w", encoding="utf-8") as output_file:

        for question_data in QUESTIONS:
            question_id = question_data["id"]
            original_query = question_data["question"]
            expansion = QUERY_EXPANSIONS[question_id]

            expanded_query = f"{original_query} {expansion}"

            if question_id == "CROSS-01":
                paper_ids = [
                    "attention_is_all_you_need",
                    "dynamic_chunking",
                ]
            else:
                paper_ids = [question_data["paper_id"]]

            paper_runs = []

            print("\n" + "-" * 85)
            print(question_id, "-", original_query)

            for paper_id in paper_ids:

                baseline_results = retrieve_paper(
                    retriever,
                    original_query,
                    paper_id,
                )

                expanded_results = retrieve_paper(
                    retriever,
                    expanded_query,
                    paper_id,
                )

                merged_results = merge_multi_query_results([
                    baseline_results,
                    expanded_results,
                ])

                paper_runs.append({
                    "paper_id": paper_id,
                    "baseline_results": baseline_results,
                    "expanded_query_results": expanded_results,
                    "merged_results": merged_results,
                })

                print("\nPaper:", paper_id)

                for label, results in [
                    ("BASELINE", baseline_results),
                    ("EXPANDED", merged_results),
                ]:
                    print("\n", label)

                    for item in results:
                        print(
                            f"Rank {item['rank']} | "
                            f"Score {item['score']:.4f} | "
                            f"{item['chunk_id']} | "
                            f"{item['section_heading']}"
                        )

            record = {
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "question_id": question_id,
                "original_query": original_query,
                "expansion_terms": expansion,
                "expanded_query": expanded_query,
                "setting": SETTING,
                "chunk_size": 600,
                "top_k": TOP_K,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "reranking": False,
                "fusion_method": "maximum cosine score per chunk",
                "paper_runs": paper_runs,
                "manual_grade": "pending",
            }

            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            total_records += 1

    print("\n" + "=" * 85)
    print("QUERY EXPANSION EXPERIMENT COMPLETED")
    print("=" * 85)
    print("Questions:", total_records)
    print("Setting:", SETTING.upper())
    print("Top K:", TOP_K)
    print("Reranking: Disabled")
    print("Saved:", output_path)
    print("=" * 85)


if __name__ == "__main__":
    main()
