import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.embedding_windows import (
    split_text_into_windows,
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_FILE = PROJECT_ROOT / "data/chunks/setting_a.jsonl"

BATCH_SIZE = 8
MAX_MODEL_TOKENS = 256


def main():
    print("=" * 70)
    print("EMBEDDING GENERATION SMOKE TEST")
    print("=" * 70)

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    # This tokenizer is used only for calculating source offsets.
    # It does not change the SentenceTransformer model's input limit.
    tokenizer.model_max_length = 1_000_000_000

    print("Model loaded.")
    print("Embedding dimension:", model.get_embedding_dimension())
    print("Model maximum sequence length:", model.max_seq_length)

    windows = []

    with CHUNK_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            chunk = json.loads(line)

            chunk_windows = split_text_into_windows(
                chunk["text"],
                tokenizer,
            )

            for window in chunk_windows:
                window["chunk_id"] = chunk["chunk_id"]
                window["paper_id"] = chunk["paper_id"]
                windows.append(window)

            if len(windows) >= 12:
                break

    if not windows:
        raise RuntimeError("No embedding windows generated.")

    print("\nWindows selected:", len(windows))

    texts = [window["text"] for window in windows]

    # Check the actual model input tensor before inference.
    model_inputs = model.tokenize(texts)
    input_shape = tuple(model_inputs["input_ids"].shape)

    print("Model input tensor shape:", input_shape)
    print("Maximum actual input length:", input_shape[1])

    assert input_shape[1] <= MAX_MODEL_TOKENS

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    norms = np.linalg.norm(embeddings, axis=1)

    print("\nEmbedding results:")
    print("Shape:", embeddings.shape)
    print("Dtype:", embeddings.dtype)
    print("All finite:", bool(np.isfinite(embeddings).all()))
    print("Minimum vector norm:", float(norms.min()))
    print("Maximum vector norm:", float(norms.max()))
    print("Maximum absolute value:", float(np.abs(embeddings).max()))

    assert embeddings.shape == (
        len(texts),
        model.get_embedding_dimension(),
    )

    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    assert np.allclose(norms, 1.0, atol=1e-5)

    print("\nSample window provenance:")

    for window in windows[:3]:
        print({
            "chunk_id": window["chunk_id"],
            "paper_id": window["paper_id"],
            "window_index": window["window_index"],
            "token_count": window["token_count"],
        })

    print("\n" + "=" * 70)
    print("RESULT: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
