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
]

TARGET_CHUNK = "dynamic_chunking_B_0006_0000"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def inspect(index_name, question, model):
    index = faiss.read_index(str(INDEX_DIR / f"{index_name}.faiss"))
    metadata = load_jsonl(
        INDEX_DIR / f"{index_name}_windows.jsonl"
    )

    query = encode_texts(model, [question])
    query = np.ascontiguousarray(query, dtype=np.float32)

    scores, ids = index.search(query, index.ntotal)

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

    ranked = sorted(
        best_by_chunk.items(),
        key=lambda item: item[1]["score"],
        reverse=True,
    )

    print(f"\nINDEX: {index_name}")
    target_rank = None

    for rank, (chunk_id, result) in enumerate(ranked, start=1):
        record = result["record"]

        if chunk_id == TARGET_CHUNK:
            target_rank = rank

        if rank <= 5 or chunk_id == TARGET_CHUNK:
            print({
                "rank": rank,
                "chunk_id": chunk_id,
                "section": record.get("section_heading"),
                "score": round(result["score"], 4),
                "window_index": record.get("window_index"),
            })

    print("Correct chunk rank:", target_rank)


def main():
    model = load_embedding_model()

    for question in QUESTIONS:
        print("\n" + "=" * 80)
        print("QUESTION:", question)
        print("=" * 80)

        inspect("setting_b", question, model)
        inspect("setting_b_context", question, model)


if __name__ == "__main__":
    main()
