from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf


# Resolve paths relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PAPERS_DIR = PROJECT_ROOT / "papers"
RAW_PAGES_DIR = PROJECT_ROOT / "data" / "raw_pages"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

MANIFEST_PATH = METADATA_DIR / "paper_manifest.json"


# These are the exact paper versions required by the assignment.
PAPERS: dict[str, dict[str, str]] = {
    "attention_is_all_you_need": {
        "filename": "attention_is_all_you_need_v7.pdf",
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "version": "v7",
    },
    "moshi": {
        "filename": "moshi_v1.pdf",
        "title": (
            "Moshi: a speech-text foundation model "
            "for real-time dialogue"
        ),
        "arxiv_id": "2410.00037",
        "version": "v1",
    },
    "dynamic_chunking": {
        "filename": "dynamic_chunking_v2.pdf",
        "title": (
            "Dynamic Chunking for End-to-End "
            "Hierarchical Sequence Modeling"
        ),
        "arxiv_id": "2507.07955",
        "version": "v2",
    },
}


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest().upper()


def extract_paper(
    paper_id: str,
    paper_info: dict[str, str],
) -> dict[str, Any]:
    """
    Extract text page by page from a research paper.

    Each page is saved as one JSONL record.

    PDF page numbers are one-based.
    Printed page numbers are not inferred here.
    """

    pdf_path = PAPERS_DIR / paper_info["filename"]

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Paper not found: {pdf_path}"
        )

    output_path = RAW_PAGES_DIR / f"{paper_id}.jsonl"

    RAW_PAGES_DIR.mkdir(parents=True, exist_ok=True)

    file_hash = calculate_sha256(pdf_path)

    total_characters = 0
    nonempty_pages = 0
    page_count = 0

    with pymupdf.open(pdf_path) as document:

        if document.is_encrypted:
            raise ValueError(
                f"Encrypted PDF cannot be processed: {pdf_path}"
            )

        page_count = document.page_count

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:

            for page_index, page in enumerate(document):

                # sort=True improves reading order for many PDFs.
                page_text = page.get_text("text").strip()

                if page_text:
                    nonempty_pages += 1

                total_characters += len(page_text)

                record = {
                    "paper_id": paper_id,
                    "paper_title": paper_info["title"],
                    "arxiv_id": paper_info["arxiv_id"],
                    "arxiv_version": paper_info["version"],
                    "source_filename": paper_info["filename"],

                    # One-based physical PDF page number.
                    "page_number": page_index + 1,

                    # Zero-based index for internal processing.
                    "pdf_page_index": page_index,

                    # Filled later only when verified.
                    "printed_page_number": None,
                    "section": None,

                    "text": page_text,
                    "character_count": len(page_text),
                    "sha256": file_hash,
                }

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    ) + "\n"
                )

    summary = {
        "paper_id": paper_id,
        "title": paper_info["title"],
        "arxiv_id": paper_info["arxiv_id"],
        "version": paper_info["version"],
        "filename": paper_info["filename"],
        "file_size_bytes": pdf_path.stat().st_size,
        "sha256": file_hash,
        "pdf_pages": page_count,
        "pages_with_text": nonempty_pages,
        "total_extracted_characters": total_characters,
        "extraction_file": str(
            output_path.relative_to(PROJECT_ROOT)
        ),
        "extraction_method": "PyMuPDF get_text(sort=True)",
        "extracted_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    logger.info(
        "%s: %s pages, %s characters",
        paper_id,
        page_count,
        total_characters,
    )

    return summary


def extract_all_papers() -> dict[str, Any]:
    """Extract all assignment papers and save a manifest."""

    summaries = []

    for paper_id, paper_info in PAPERS.items():

        logger.info("Extracting %s", paper_info["title"])

        summary = extract_paper(
            paper_id,
            paper_info,
        )

        summaries.append(summary)

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "project": "The Relay",
        "extraction_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "paper_count": len(summaries),
        "total_pdf_pages": sum(
            item["pdf_pages"] for item in summaries
        ),
        "total_extracted_characters": sum(
            item["total_extracted_characters"]
            for item in summaries
        ),
        "papers": summaries,
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Manifest saved to %s",
        MANIFEST_PATH,
    )

    return manifest


if __name__ == "__main__":

    result = extract_all_papers()

    print("\n" + "=" * 65)
    print("PAPER EXTRACTION SUMMARY")
    print("=" * 65)

    for paper in result["papers"]:
        print(f"\nPaper: {paper['title']}")
        print(f"Version: {paper['arxiv_id']}{paper['version']}")
        print(f"Pages: {paper['pdf_pages']}")
        print(f"Pages with text: {paper['pages_with_text']}")
        print(
            "Characters:",
            paper["total_extracted_characters"],
        )
        print("SHA256:", paper["sha256"])

    print("\nTotal pages:", result["total_pdf_pages"])
    print(
        "Total characters:",
        result["total_extracted_characters"],
    )
    print("\nExtraction completed.")
