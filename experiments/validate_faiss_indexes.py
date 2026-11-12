import json
import sys
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.embeddings import (
    encode_texts,
    load_embedding_model,
)


INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
CHUNK_DIR = PROJECT_ROOT / "data" / "chunks"

SETTINGS = ("a", "b", "c")

SAMPLE_COUNT = 3

CITATION_FIELDS = (
    "paper_id",
    "paper_title",
    "arxiv_id",
    "arxiv_version",
    "page_start",
    "page_end",
    "page_numbers",
    "section_number",
    "section_title",
    "section_heading",
    "section_path",
)


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def validate_setting(setting, model):
    print("\n" + "=" * 70)
    print(f"VALIDATING SETTING {setting.upper()}")
    print("=" * 70)

    index_path = INDEX_DIR / f"setting_{setting}.faiss"
    metadata_path = INDEX_DIR / f"setting_{setting}_windows.jsonl"
    chunk_path = CHUNK_DIR / f"setting_{setting}.jsonl"

    index = faiss.read_index(str(index_path))
    metadata = load_jsonl(metadata_path)
    chunks = load_jsonl(chunk_path)

    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    errors = []

    # 1. Vector count and dimension.
    if index.ntotal != len(metadata):
        errors.append(
            f"Vector/metadata count mismatch: {index.ntotal}/{len(metadata)}"
        )

    if index.d != model.get_embedding_dimension():
        errors.append(
            f"Dimension mismatch: {index.d}"
        )

    # 2. Row ordering and source provenance.
    for row_number, record in enumerate(metadata):
        if record["vector_id"] != row_number:
            errors.append(
                f"Row {row_number}: vector_id is {record['vector_id']}"
            )

        chunk = chunk_lookup.get(record["chunk_id"])

        if chunk is None:
            errors.append(
                f"Row {row_number}: missing source chunk {record['chunk_id']}"
            )
            continue

        for field in CITATION_FIELDS:
            if record[field] != chunk[field]:
                errors.append(
                    f"Row {row_number}: citation field mismatch: {field}"
                )

        start = record["window_start_char"]
        end = record["window_end_char"]

        if not (0 <= start < end <= len(chunk["text"])):
            errors.append(
                f"Row {row_number}: invalid source character offsets"
            )
            continue

        expected_text = chunk["text"][start:end]

        if expected_text != record["window_text"]:
            errors.append(
                f"Row {row_number}: window text differs from source chunk"
            )

    # 3. Check actual FAISS vector contents against fresh embeddings.
    if metadata:
        sample_indices = sorted(set([
            0,
            len(metadata) // 2,
            len(metadata) - 1,
        ]))

        sample_texts = [
            metadata[index]["window_text"]
            for index in sample_indices
        ]

        fresh_embeddings = encode_texts(
            model,
            sample_texts,
            batch_size=SAMPLE_COUNT,
        )

        sample_similarities = []

        for sample_position, vector_id in enumerate(sample_indices):
            saved_vector = index.reconstruct(vector_id)
            fresh_vector = fresh_embeddings[sample_position]

            similarity = float(
                np.dot(saved_vector, fresh_vector)
            )

            sample_similarities.append(similarity)

            if not np.allclose(
                saved_vector,
                fresh_vector,
                atol=1e-5,
            ):
                errors.append(
                    f"Vector {vector_id}: saved vector differs from fresh embedding"
                )

    else:
        sample_similarities = []

    # 4. Actual nearest-neighbour search.
    if metadata:
        query_embedding = encode_texts(
            model,
            [metadata[0]["window_text"]],
        )

        scores, ids = index.search(
            np.ascontiguousarray(query_embedding, dtype=np.float32),
            k=min(5, index.ntotal),
        )

        returned_ids = ids[0].tolist()
        returned_scores = scores[0].tolist()

        for vector_id in returned_ids:
            if vector_id < 0 or vector_id >= len(metadata):
                errors.append(
                    f"Search returned invalid vector ID: {vector_id}"
                )

        search_results = [
            {
                "vector_id": vector_id,
                "chunk_id": metadata[vector_id]["chunk_id"],
                "paper_id": metadata[vector_id]["paper_id"],
                "score": float(score),
            }
            for vector_id, score in zip(
                returned_ids,
                returned_scores,
            )
            if vector_id >= 0
        ]
    else:
        search_results = []

    print("FAISS vectors:", index.ntotal)
    print("Metadata records:", len(metadata))
    print("Source chunks:", len(chunks))
    print("Sample vector similarities:", sample_similarities)
    print("\nSample nearest-neighbour results:")

    for result in search_results:
        print(result)

    print("\nValidation errors:", len(errors))

    for error in errors[:20]:
        print("ERROR:", error)

    if errors:
        raise RuntimeError(
            f"SETTING {setting.upper()} VALIDATION FAILED"
        )

    print(f"SETTING {setting.upper()} PASS")


def main():
    print("=" * 70)
    print("INDEPENDENT FAISS INDEX VALIDATION")
    print("=" * 70)

    model = load_embedding_model()

    for setting in SETTINGS:
        validate_setting(setting, model)

    print("\n" + "=" * 70)
    print("ALL THREE FAISS INDEXES VALIDATED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
