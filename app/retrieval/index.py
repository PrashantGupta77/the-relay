import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from app.retrieval.embedding_windows import (
    MAX_CONTENT_TOKENS,
    WINDOW_OVERLAP_TOKENS,
    split_text_into_windows,
)
from app.retrieval.embeddings import (
    MODEL_NAME,
    encode_texts,
    load_embedding_model,
    load_offset_tokenizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNK_DIR = PROJECT_ROOT / "data" / "chunks"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
MANIFEST_DIR = PROJECT_ROOT / "data" / "metadata"

SETTINGS = ("a", "b", "c")
BATCH_SIZE = 32


def load_chunks(path):
    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def build_window_records(chunks, tokenizer):
    records = []

    for chunk in chunks:
        windows = split_text_into_windows(
            chunk["text"],
            tokenizer,
            max_content_tokens=MAX_CONTENT_TOKENS,
            overlap_tokens=WINDOW_OVERLAP_TOKENS,
        )

        for window in windows:
            records.append({
                "vector_id": len(records),
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "paper_title": chunk["paper_title"],
                "arxiv_id": chunk["arxiv_id"],
                "arxiv_version": chunk["arxiv_version"],
                "source_filename": chunk["source_filename"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "page_numbers": chunk["page_numbers"],
                "section_number": chunk["section_number"],
                "section_title": chunk["section_title"],
                "section_heading": chunk["section_heading"],
                "section_path": chunk["section_path"],
                "section_source": chunk["section_source"],
                "setting": chunk["setting"],
                "chunk_size": chunk["chunk_size"],
                "chunk_token_count": chunk["token_count"],
                "window_index": window["window_index"],
                "window_start_token": window["start_token"],
                "window_end_token": window["end_token"],
                "window_start_char": window["start_char"],
                "window_end_char": window["end_char"],
                "window_token_count": window["token_count"],
                "window_character_count": window["character_count"],
                "window_text": window["text"],
            })

    return records


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def build_setting_index(setting, model, tokenizer):
    chunk_path = CHUNK_DIR / f"setting_{setting}.jsonl"

    if not chunk_path.exists():
        raise FileNotFoundError(chunk_path)

    print("\n" + "=" * 70)
    print(f"BUILDING FAISS INDEX: SETTING {setting.upper()}")
    print("=" * 70)

    chunks = load_chunks(chunk_path)

    if not chunks:
        raise RuntimeError(f"No chunks found in {chunk_path}")

    print("Original chunks:", len(chunks))

    records = build_window_records(chunks, tokenizer)

    if not records:
        raise RuntimeError(f"No embedding windows generated for setting {setting}")

    texts = [record["window_text"] for record in records]

    print("Embedding windows:", len(texts))
    print("Embedding in batches of:", BATCH_SIZE)

    embeddings = encode_texts(
        model,
        texts,
        batch_size=BATCH_SIZE,
    )

    expected_dimension = model.get_embedding_dimension()

    if embeddings.shape != (len(records), expected_dimension):
        raise RuntimeError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError("Non-finite values found in embeddings.")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise RuntimeError("Embeddings are not normalized.")

    # Normalized vectors + inner product = cosine similarity.
    index = faiss.IndexFlatIP(expected_dimension)

    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32,
    )

    index.add(embeddings)

    if index.ntotal != len(records):
        raise RuntimeError(
            f"FAISS vector count mismatch: {index.ntotal} vs {len(records)}"
        )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / f"setting_{setting}.faiss"
    metadata_path = INDEX_DIR / f"setting_{setting}_windows.jsonl"
    manifest_path = MANIFEST_DIR / f"setting_{setting}_index_manifest.json"

    faiss.write_index(index, str(index_path))
    write_jsonl(metadata_path, records)

    # Confirm the saved index can be loaded and has the expected dimensions.
    saved_index = faiss.read_index(str(index_path))

    if saved_index.ntotal != len(records):
        raise RuntimeError("Saved FAISS index has an unexpected vector count.")

    if saved_index.d != expected_dimension:
        raise RuntimeError("Saved FAISS index has an unexpected dimension.")

    manifest = {
        "setting": setting,
        "source_chunk_file": str(chunk_path.relative_to(PROJECT_ROOT)),
        "index_file": str(index_path.relative_to(PROJECT_ROOT)),
        "metadata_file": str(metadata_path.relative_to(PROJECT_ROOT)),
        "embedding_model": MODEL_NAME,
        "embedding_dimension": expected_dimension,
        "model_max_sequence_length": 256,
        "embedding_content_token_limit": MAX_CONTENT_TOKENS,
        "embedding_window_overlap_tokens": WINDOW_OVERLAP_TOKENS,
        "similarity_metric": "inner_product_on_normalized_vectors",
        "faiss_index_type": "IndexFlatIP",
        "original_chunk_count": len(chunks),
        "embedding_window_count": len(records),
        "vector_count": saved_index.ntotal,
        "metadata_record_count": len(records),
        "embedding_dtype": str(embeddings.dtype),
        "normalized_embeddings": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print("Embedding shape:", embeddings.shape)
    print("Vector dimension:", saved_index.d)
    print("Vectors in FAISS:", saved_index.ntotal)
    print("Metadata records:", len(records))
    print("Vector norm range:", float(norms.min()), "to", float(norms.max()))
    print("Index file:", index_path)
    print("Metadata file:", metadata_path)
    print("Manifest file:", manifest_path)
    print("SETTING", setting.upper(), "PASS")


def main():
    print("=" * 70)
    print("THE RELAY - FAISS INDEX BUILDER")
    print("=" * 70)

    print("\nLoading embedding model...")
    model = load_embedding_model()

    print("Loading offset tokenizer...")
    tokenizer = load_offset_tokenizer()

    print("Model:", MODEL_NAME)
    print("Embedding dimension:", model.get_embedding_dimension())
    print("Model sequence length:", model.max_seq_length)
    print("Embedding window content limit:", MAX_CONTENT_TOKENS)
    print("Embedding window overlap:", WINDOW_OVERLAP_TOKENS)

    for setting in SETTINGS:
        build_setting_index(
            setting,
            model,
            tokenizer,
        )

    print("\n" + "=" * 70)
    print("ALL THREE FAISS INDEXES BUILT SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
