import json
from pathlib import Path

import faiss
import numpy as np

from app.retrieval.embeddings import (
    encode_texts,
    load_embedding_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
CHUNK_DIR = PROJECT_ROOT / "data" / "chunks"

VALID_SETTINGS = {"a", "b", "c"}


class Retriever:
    def __init__(self, setting="b", model=None):
        setting = setting.lower()

        if setting not in VALID_SETTINGS:
            raise ValueError(
                f"Invalid setting '{setting}'. Choose from A, B, or C."
            )

        self.setting = setting
        self.model = model or load_embedding_model()

        index_path = INDEX_DIR / f"setting_{setting}.faiss"
        metadata_path = INDEX_DIR / f"setting_{setting}_windows.jsonl"
        chunk_path = CHUNK_DIR / f"setting_{setting}.jsonl"

        for path in (index_path, metadata_path, chunk_path):
            if not path.exists():
                raise FileNotFoundError(path)

        self.index = faiss.read_index(str(index_path))
        self.window_metadata = self._load_jsonl(metadata_path)
        chunks = self._load_jsonl(chunk_path)

        self.chunk_lookup = {
            chunk["chunk_id"]: chunk
            for chunk in chunks
        }

        self._validate_loaded_data()

    @staticmethod
    def _load_jsonl(path):
        records = []

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))

        return records

    def _validate_loaded_data(self):
        if self.index.ntotal != len(self.window_metadata):
            raise RuntimeError(
                "FAISS index and window metadata counts do not match."
            )

        if self.index.d != self.model.get_embedding_dimension():
            raise RuntimeError(
                "FAISS index and embedding model dimensions do not match."
            )

        for row_number, record in enumerate(self.window_metadata):
            if record["vector_id"] != row_number:
                raise RuntimeError(
                    f"Metadata row {row_number} has an invalid vector ID."
                )

            if record["chunk_id"] not in self.chunk_lookup:
                raise RuntimeError(
                    f"Missing original chunk: {record['chunk_id']}"
                )

    def retrieve(self, question, top_k=5, paper_id=None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if paper_id is not None and paper_id not in {
            chunk["paper_id"] for chunk in self.chunk_lookup.values()
        }:
            raise ValueError(f"Unknown paper_id: {paper_id}")

        query_vector = encode_texts(
            self.model,
            [question.strip()],
        )

        query_vector = np.ascontiguousarray(
            query_vector,
            dtype=np.float32,
        )

        # The indexes are small, so search all vectors to ensure
        # filtering and deduplication do not discard relevant candidates.
        scores, vector_ids = self.index.search(
            query_vector,
            self.index.ntotal,
        )

        # Retain the best matching window for each original chunk.
        best_by_chunk = {}

        for score, vector_id in zip(scores[0], vector_ids[0]):
            if vector_id < 0:
                continue

            window = self.window_metadata[int(vector_id)]

            if paper_id is not None and window["paper_id"] != paper_id:
                continue

            chunk_id = window["chunk_id"]

            if chunk_id not in best_by_chunk:
                best_by_chunk[chunk_id] = {
                    "score": float(score),
                    "matched_window": window,
                }

        ranked = sorted(
            best_by_chunk.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )[:top_k]

        results = []

        for rank, (chunk_id, match) in enumerate(ranked, start=1):
            chunk = self.chunk_lookup[chunk_id]

            results.append({
                "rank": rank,
                "score": match["score"],
                "chunk_id": chunk_id,
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
                "chunk_token_count": chunk["token_count"],
                "matched_window_index": match["matched_window"]["window_index"],
                "matched_window_score": match["score"],
                "text": chunk["text"],
            })

        return {
            "question": question.strip(),
            "setting": self.setting,
            "top_k": top_k,
            "paper_filter": paper_id,
            "result_count": len(results),
            "results": results,
        }


def format_citation(result):
    pages = sorted(set(result["page_numbers"]))

    if len(pages) == 1:
        page_text = f"PDF p. {pages[0]}"
    elif pages == list(range(pages[0], pages[-1] + 1)):
        page_text = f"PDF pp. {pages[0]}–{pages[-1]}"
    else:
        page_text = "PDF pp. " + ", ".join(map(str, pages))

    version = str(result["arxiv_version"])
    if not version.startswith("v"):
        version = f"v{version}"

    section = result.get("section_heading")

    citation = (
        f'{result["paper_title"]}, '
        f'arXiv:{result["arxiv_id"]}{version}, '
        f'{page_text}'
    )

    if section:
        citation += f", §{section}"

    return citation
