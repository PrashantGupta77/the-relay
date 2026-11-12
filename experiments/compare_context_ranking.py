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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"

QUESTIONS = [
    "How does dynamic chunking decide when to create a boundary?",
    "How does the H-Net routing module detect chunk boundaries?",
    "How are boundary probabilities calculated from adjacent encoder representations?",
    "What cosine similarity threshold is used to identify a boundary?",
]

TARGET_CHUNK = "dynamic_chunking_B_0006_0000"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def rank_target(index, metadata, model, question):
    query_vector = encode_texts(model, [question])
    query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

    scores, ids = index.search(query_vector, index.ntotal)

    best_by_chunk = {}

    for score, vector_id in zip(scores[0], ids[0]):
        if vector_id < 0:
            continue

        record = metadata[int(vector_id)]
        chunk_id = record["chunk_id"]

        if chunk_id not in best_by_chunk:
            best_by_chunk[chunk_id] = float(score)

    ranked = sorted(
        best_by_chunk.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for rank, (chunk_id, score) in enumerate(ranked, start=1):
        if chunk_id == TARGET_CHUNK:
            return rank, score

    return None, None


def main():
    print("=" * 80)
    print("BASELINE VS SECTION-CONTEXT RETRIEVAL")
    print("=" * 80)

    model = load_embedding_model()

    baseline_index = faiss.read_index(
        str(INDEX_DIR / "setting_b.faiss")
    )
    baseline_metadata = load_jsonl(
        INDEX_DIR / "setting_b_windows.jsonl"
    )

    context_index = faiss.read_index(
        str(INDEX_DIR / "setting_b_context.faiss")
    )
    context_metadata = load_jsonl(
        INDEX_DIR / "setting_b_context_windows.jsonl"
    )

    print("Baseline vectors:", baseline_index.ntotal)
    print("Context vectors:", context_index.ntotal)

    if baseline_index.ntotal != context_index.ntotal:
        raise RuntimeError("The two indexes have different vector counts.")

    results = []

    for question in QUESTIONS:
        baseline_rank, baseline_score = rank_target(
            baseline_index,
            baseline_metadata,
            model,
            question,
        )

        context_rank, context_score = rank_target(
            context_index,
            context_metadata,
            model,
            question,
        )

        result = {
            "question": question,
            "target_chunk": TARGET_CHUNK,
            "baseline_rank": baseline_rank,
            "baseline_score": baseline_score,
            "context_rank": context_rank,
            "context_score": context_score,
        }

        results.append(result)

        print("\nQuestion:", question)
        print("Baseline rank:", baseline_rank, "score:", baseline_score)
        print("Context rank:", context_rank, "score:", context_score)

    output_path = PROJECT_ROOT / "results" / "context_ranking_comparison.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("\nSaved:", output_path)
    print("\nComparison complete.")


if __name__ == "__main__":
    main()
