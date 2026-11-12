import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import faiss
import numpy as np

from app.retrieval.embeddings import (
    encode_texts,
    load_embedding_model,
)

INDEX_DIR = PROJECT_ROOT / "data" / "indexes"

QUESTIONS = [
    "How does dynamic chunking decide when to create a boundary?",
    "How does the H-Net routing module detect chunk boundaries?",
    "How are boundary probabilities calculated from adjacent encoder representations?",
    "What cosine similarity threshold is used to identify a boundary?",
]

TARGET_CHUNK = "dynamic_chunking_B_0006_0000"

# This is a deliberately small experimental candidate pool.
CANDIDATE_K = 20

# Test several reranking strengths rather than assuming one is optimal.
ALPHAS = [0.05, 0.10, 0.20]


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def retrieve_candidates(index, metadata, model, question):
    query_vector = encode_texts(model, [question])
    query_vector = np.ascontiguousarray(
        query_vector,
        dtype=np.float32,
    )

    scores, ids = index.search(
        query_vector,
        min(CANDIDATE_K * 3, index.ntotal),
    )

    best_by_chunk = {}

    for score, vector_id in zip(scores[0], ids[0]):
        if vector_id < 0:
            continue

        record = metadata[int(vector_id)]
        chunk_id = record["chunk_id"]

        if chunk_id not in best_by_chunk:
            best_by_chunk[chunk_id] = {
                "score": float(score),
                "record": record,
            }

    candidates = sorted(
        best_by_chunk.items(),
        key=lambda item: item[1]["score"],
        reverse=True,
    )

    return candidates[:CANDIDATE_K]


def section_similarity(question, candidates, model):
    section_texts = []

    for _, result in candidates:
        record = result["record"]

        section = record.get("section_path") or "Front matter"

        section_texts.append(
            f"Paper: {record['paper_title']}. "
            f"Section: {section}"
        )

    question_vector = encode_texts(model, [question])[0]
    section_vectors = encode_texts(model, section_texts)

    return np.dot(section_vectors, question_vector)


def main():
    print("=" * 80)
    print("SECTION-AWARE RERANKING EXPERIMENT")
    print("=" * 80)

    model = load_embedding_model()

    index = faiss.read_index(
        str(INDEX_DIR / "setting_b.faiss")
    )

    metadata = load_jsonl(
        INDEX_DIR / "setting_b_windows.jsonl"
    )

    all_results = []

    for question in QUESTIONS:
        print("\nQUESTION:", question)

        candidates = retrieve_candidates(
            index,
            metadata,
            model,
            question,
        )

        section_scores = section_similarity(
            question,
            candidates,
            model,
        )

        semantic_scores = np.array(
            [result["score"] for _, result in candidates],
            dtype=np.float32,
        )

        # Min-max normalize semantic and section scores separately.
        semantic_range = np.ptp(semantic_scores)
        section_range = np.ptp(section_scores)

        semantic_norm = (
            (semantic_scores - semantic_scores.min()) / semantic_range
            if semantic_range > 1e-8
            else np.zeros_like(semantic_scores)
        )

        section_norm = (
            (section_scores - section_scores.min()) / section_range
            if section_range > 1e-8
            else np.zeros_like(section_scores)
        )

        baseline_rank = next(
            (
                rank
                for rank, (chunk_id, _) in enumerate(candidates, 1)
                if chunk_id == TARGET_CHUNK
            ),
            None,
        )

        print("Baseline target rank:", baseline_rank)

        for alpha in ALPHAS:
            combined_scores = (
                (1 - alpha) * semantic_norm
                + alpha * section_norm
            )

            order = np.argsort(-combined_scores)

            ranked_ids = [
                candidates[int(i)][0]
                for i in order
            ]

            target_rank = (
                ranked_ids.index(TARGET_CHUNK) + 1
                if TARGET_CHUNK in ranked_ids
                else None
            )

            print(
                f"Alpha={alpha:.2f} | "
                f"Target rank={target_rank}"
            )

            top_five = []

            for position, idx in enumerate(order[:5], 1):
                chunk_id, result = candidates[int(idx)]

                entry = {
                    "rank": position,
                    "chunk_id": chunk_id,
                    "section": result["record"].get("section_heading"),
                    "combined_score": float(combined_scores[int(idx)]),
                }

                top_five.append(entry)

            for item in top_five:
                print(item)

            all_results.append({
                "question": question,
                "alpha": alpha,
                "baseline_rank": baseline_rank,
                "reranked_target_rank": target_rank,
                "top_five": top_five,
            })

    output_path = (
        PROJECT_ROOT
        / "results"
        / "section_reranking_experiment.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSaved:", output_path)
    print("Experiment complete.")


if __name__ == "__main__":
    main()
