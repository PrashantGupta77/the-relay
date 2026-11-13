import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.embeddings import (
    encode_texts,
    load_embedding_model,
)

import faiss
import numpy as np


INDEX_PATH = PROJECT_ROOT / "data/indexes/setting_b.faiss"
METADATA_PATH = PROJECT_ROOT / "data/indexes/setting_b_windows.jsonl"

TARGET_CHUNK = "dynamic_chunking_B_0006_0000"

QUESTIONS = [
    "How does dynamic chunking decide when to create a boundary?",
    "How does the H-Net routing module detect chunk boundaries?",
    "How are boundary probabilities calculated from adjacent encoder representations?",
    "What cosine similarity threshold is used to identify a boundary?",
]


def main():
    model = load_embedding_model()
    index = faiss.read_index(str(INDEX_PATH))

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    for question in QUESTIONS:
        print("\n" + "=" * 80)
        print("QUESTION:", question)
        print("=" * 80)

        query_vector = encode_texts(model, [question])
        query_vector = np.ascontiguousarray(
            query_vector,
            dtype=np.float32,
        )

        scores, ids = index.search(
            query_vector,
            index.ntotal,
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

        ranked = sorted(
            best_by_chunk.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )

        target_rank = None

        for rank, (chunk_id, result) in enumerate(ranked, start=1):
            if chunk_id == TARGET_CHUNK:
                target_rank = rank

            if rank <= 5 or chunk_id == TARGET_CHUNK:
                print({
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "section": result["record"]["section_heading"],
                    "score": round(result["score"], 4),
                })

        print("Correct chunk rank:", target_rank)


if __name__ == "__main__":
    main()
