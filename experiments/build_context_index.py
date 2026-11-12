import json
import sys
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.embeddings import (
    load_embedding_model,
    load_offset_tokenizer,
    encode_texts,
)


SETTING = "b"
BATCH_SIZE = 32
MAX_CONTENT_TOKENS = 254

INDEX_DIR = PROJECT_ROOT / "data" / "indexes"

BASE_METADATA = INDEX_DIR / f"setting_{SETTING}_windows.jsonl"
INDEX_OUTPUT = INDEX_DIR / f"setting_{SETTING}_context.faiss"
METADATA_OUTPUT = INDEX_DIR / f"setting_{SETTING}_context_windows.jsonl"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def build_context_text(record, tokenizer):
    paper_title = record["paper_title"]
    section_path = record.get("section_path") or "Front matter"

    prefix = (
        f"Paper: {paper_title}. "
        f"Section: {section_path}. "
        "Content: "
    )

    prefix_tokens = tokenizer(
        prefix,
        add_special_tokens=False,
    )["input_ids"]

    content_budget = MAX_CONTENT_TOKENS - len(prefix_tokens)

    if content_budget < 1:
        raise RuntimeError(
            f"Context prefix is too long for vector {record['vector_id']}"
        )

    content = record["window_text"]

    content_encoding = tokenizer(
        content,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )

    offsets = content_encoding["offset_mapping"]

    if len(offsets) > content_budget:
        end_char = offsets[content_budget - 1][1]
        content = content[:end_char]

    embedding_text = prefix + content

    final_tokens = tokenizer(
        embedding_text,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    if len(final_tokens) > 256:
        raise RuntimeError(
            f"Embedding input exceeds model limit: {len(final_tokens)}"
        )

    updated_record = dict(record)
    updated_record["embedding_text"] = embedding_text
    updated_record["embedding_prefix_token_count"] = len(prefix_tokens)
    updated_record["embedding_content_token_count"] = len(
        tokenizer(content, add_special_tokens=False)["input_ids"]
    )
    updated_record["embedding_input_token_count"] = len(final_tokens)

    return embedding_text, updated_record


def main():
    print("=" * 70)
    print("BUILDING SECTION-CONTEXT EXPERIMENTAL INDEX")
    print("=" * 70)

    if not BASE_METADATA.exists():
        raise FileNotFoundError(BASE_METADATA)

    model = load_embedding_model()
    tokenizer = load_offset_tokenizer()

    records = load_jsonl(BASE_METADATA)

    print("Baseline metadata records:", len(records))

    embedding_texts = []
    context_records = []

    for record in records:
        embedding_text, updated_record = build_context_text(
            record,
            tokenizer,
        )

        embedding_texts.append(embedding_text)
        context_records.append(updated_record)

    token_counts = [
        record["embedding_input_token_count"]
        for record in context_records
    ]

    print("Context embedding inputs:", len(embedding_texts))
    print("Minimum input tokens:", min(token_counts))
    print("Maximum input tokens:", max(token_counts))
    print("Average input tokens:", round(sum(token_counts) / len(token_counts), 2))

    embeddings = encode_texts(
        model,
        embedding_texts,
        batch_size=BATCH_SIZE,
    )

    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32,
    )

    if not np.isfinite(embeddings).all():
        raise RuntimeError("Non-finite embeddings found.")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise RuntimeError("Context embeddings are not normalized.")

    index = faiss.IndexFlatIP(
        model.get_embedding_dimension()
    )

    index.add(embeddings)

    if index.ntotal != len(context_records):
        raise RuntimeError("Context index vector count mismatch.")

    faiss.write_index(index, str(INDEX_OUTPUT))

    with METADATA_OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for record in context_records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    # Reload and verify the saved experimental index.
    saved_index = faiss.read_index(str(INDEX_OUTPUT))

    if saved_index.ntotal != len(context_records):
        raise RuntimeError("Saved context index count mismatch.")

    print("\nIndex:", INDEX_OUTPUT)
    print("Metadata:", METADATA_OUTPUT)
    print("FAISS vectors:", saved_index.ntotal)
    print("Dimension:", saved_index.d)
    print("Vector norm range:", float(norms.min()), "to", float(norms.max()))
    print("All inputs within 256 tokens: PASS")
    print("All embeddings normalized: PASS")

    print("\nSample context input:")
    print(embedding_texts[0][:500])

    print("\n" + "=" * 70)
    print("CONTEXT INDEX BUILD: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
