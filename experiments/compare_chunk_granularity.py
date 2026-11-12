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

TARGET_SECTION = "2.2.1 Chunking Layer"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def retrieve(index, metadata, model, question):
    query = encode_texts(model, [question])
    query = np.ascontiguousarray(query, dtype=np.float32)

    scores, ids = index.search(query, index.ntotal)

    best_by_chunk = {}

    for score, vector_id in zip(scores[0], ids[0]):
        if vector_id < 0:
            continue

        record = metadata[int(vector_id)]

        if record["paper_id"] != "dynamic_chunking":
            continue

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

    return ranked


def main():
    print("=" * 80)
    print("CHUNK GRANULARITY COMPARISON: SETTING A VS B")
    print("=" * 80)

    model = load_embedding_model()
    results = []

    for setting in ["a", "b"]:
        index = faiss.read_index(
            str(INDEX_DIR / f"setting_{setting}.faiss")
        )

        metadata = load_jsonl(
            INDEX_DIR / f"setting_{setting}_windows.jsonl"
        )

        print(f"\nSETTING {setting.upper()}")
        print("FAISS vectors:", index.ntotal)

        for question in QUESTIONS:
            ranked = retrieve(
                index,
                metadata,
                model,
                question,
            )

            matching = [
                (rank, chunk_id, item)
                for rank, (chunk_id, item) in enumerate(ranked, 1)
                if item["record"].get("section_title") == "Chunking Layer"
            ]

            best_match = matching[0] if matching else None

            print("\nQuestion:", question)

            if best_match:
                rank, chunk_id, item = best_match

                print("Best Chunking Layer rank:", rank)
                print("Chunk ID:", chunk_id)
                print("Score:", round(item["score"], 4))
                print("Pages:", item["record"].get("page_numbers"))
                print("Window:", item["record"].get("window_index"))
            else:
                rank = None
                chunk_id = None
                print("Chunking Layer result: NOT FOUND")

            print("Top five:")

            for position, (chunk_id, item) in enumerate(ranked[:5], 1):
                record = item["record"]

                print({
                    "rank": position,
                    "chunk_id": chunk_id,
                    "section": record.get("section_heading"),
                    "score": round(item["score"], 4),
                })

            results.append({
                "setting": setting,
                "question": question,
                "best_chunking_layer_rank": rank,
                "best_chunk_id": chunk_id,
            })

    output = PROJECT_ROOT / "results" / "chunk_granularity_comparison.json"

    with output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("\nSaved:", output)
    print("\nComparison complete.")


if __name__ == "__main__":
    main()
