import json
from pathlib import Path
from collections import defaultdict

import tiktoken


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILES = {
    "attention_is_all_you_need": (
        PROJECT_ROOT
        / "data"
        / "sectioned_pages"
        / "attention_is_all_you_need.jsonl"
    ),
    "dynamic_chunking": (
        PROJECT_ROOT
        / "data"
        / "sectioned_pages"
        / "dynamic_chunking.jsonl"
    ),
    "moshi": (
        PROJECT_ROOT
        / "data"
        / "sectioned_pages"
        / "moshi.jsonl"
    ),
}

OUTPUT_DIR = PROJECT_ROOT / "data" / "chunks"
SUMMARY_PATH = PROJECT_ROOT / "data" / "metadata" / "chunking_summary.json"

# Consistent local chunk-sizing convention for the selected gpt-oss model.
TOKENIZER_NAME = "o200k_harmony"
TOKENIZER_VERSION = "tiktoken"

CHUNK_CONFIGS = {
    "A": {"chunk_size": 300, "overlap": 50},
    "B": {"chunk_size": 600, "overlap": 80},
    "C": {"chunk_size": 900, "overlap": 120},
}


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from exc

    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def group_consecutive_segments(pages: list[dict]) -> list[dict]:
    """
    Combine consecutive segments belonging to the same section.

    A group never crosses a paper boundary or a section boundary.
    Page and segment provenance is retained.
    """

    groups = []
    current_group = None

    for page in sorted(pages, key=lambda item: item["page_number"]):
        for segment_index, segment in enumerate(
            page.get("section_segments", [])
        ):
            text = segment.get("text", "")

            if not text.strip():
                continue

            section_path = segment.get("section_path")

            same_group = (
                current_group is not None
                and current_group["section_path"] == section_path
            )

            if not same_group:
                if current_group is not None:
                    groups.append(current_group)

                current_group = {
                    "paper_id": page["paper_id"],
                    "paper_title": page["paper_title"],
                    "arxiv_id": page["arxiv_id"],
                    "arxiv_version": page["arxiv_version"],
                    "source_filename": page["source_filename"],
                    "section_number": segment.get("section_number"),
                    "section_title": segment.get("section_title"),
                    "section_heading": segment.get("section_heading"),
                    "section_path": section_path,
                    "section_source": segment.get("section_source"),
                    "parts": [],
                }

            current_group["parts"].append({
                "page_number": page["page_number"],
                "pdf_page_index": page["pdf_page_index"],
                "line_start": segment.get("line_start"),
                "line_end": segment.get("line_end"),
                "segment_index": segment_index,
                "text": text,
            })

    if current_group is not None:
        groups.append(current_group)

    return groups


def prepare_group_text(group: dict) -> tuple[str, list[dict]]:
    """
    Join segment text while retaining character ranges for each source part.
    """

    pieces = []
    source_ranges = []
    cursor = 0

    for index, part in enumerate(group["parts"]):
        if index > 0:
            separator = "\n\n"
            pieces.append(separator)
            cursor += len(separator)

        text = part["text"]
        start = cursor
        end = start + len(text)

        pieces.append(text)

        source_ranges.append({
            "start_char": start,
            "end_char": end,
            "page_number": part["page_number"],
            "pdf_page_index": part["pdf_page_index"],
            "line_start": part["line_start"],
            "line_end": part["line_end"],
            "segment_index": part["segment_index"],
        })

        cursor = end

    return "".join(pieces), source_ranges


def find_source_parts(
    start_char: int,
    end_char: int,
    source_ranges: list[dict],
) -> list[dict]:
    """
    Find source segments whose text overlaps a chunk's character range.
    """

    matches = []

    for source in source_ranges:
        overlaps = (
            source["start_char"] < end_char
            and source["end_char"] > start_char
        )

        if overlaps:
            matches.append({
                "page_number": source["page_number"],
                "pdf_page_index": source["pdf_page_index"],
                "line_start": source["line_start"],
                "line_end": source["line_end"],
                "segment_index": source["segment_index"],
            })

    return matches


def chunk_group(
    group: dict,
    group_index: int,
    encoding,
    chunk_size: int,
    overlap: int,
    setting_name: str,
) -> list[dict]:

    full_text, source_ranges = prepare_group_text(group)

    token_ids = encoding.encode(
        full_text,
        disallowed_special=(),
    )

    if not token_ids:
        return []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    step = chunk_size - overlap

    # Character offsets corresponding to token boundaries.
    _, token_offsets = encoding.decode_with_offsets(token_ids)

    chunks = []
    start_token = 0
    chunk_index = 0

    while start_token < len(token_ids):
        end_token = min(
            start_token + chunk_size,
            len(token_ids),
        )

        # Map token boundaries back to the original text.
        start_char = token_offsets[start_token]

        if end_token < len(token_offsets):
            end_char = token_offsets[end_token]
        else:
            end_char = len(full_text)

        # Slice the original source text instead of decoding an
        # isolated token window. This preserves the original text.
        chunk_text = full_text[start_char:end_char]

        # Re-encoding an isolated text window can produce a slightly
        # different token count because token boundaries are contextual.
        actual_token_count = len(
            encoding.encode(chunk_text, disallowed_special=())
        )

        # Never allow a chunk to exceed its configured token budget.
        # Move the ending character boundary backwards if necessary.
        while actual_token_count > chunk_size and end_char > start_char:
            end_char -= 1
            chunk_text = full_text[start_char:end_char]

            actual_token_count = len(
                encoding.encode(chunk_text, disallowed_special=())
            )

        if not chunk_text:
            raise ValueError(
                f"Empty chunk generated: {group['paper_id']}, "
                f"group {group_index}, chunk {chunk_index}"
            )

        source_parts = find_source_parts(
            start_char,
            end_char,
            source_ranges,
        )

        page_numbers = sorted({
            item["page_number"]
            for item in source_parts
        })

        if not page_numbers:
            raise ValueError(
                "Chunk has no source-page mapping: "
                f"{group['paper_id']}, group {group_index}"
            )

        chunk = {
            "chunk_id": (
                f"{group['paper_id']}_"
                f"{setting_name}_"
                f"{group_index:04d}_"
                f"{chunk_index:04d}"
            ),
            "paper_id": group["paper_id"],
            "paper_title": group["paper_title"],
            "arxiv_id": group["arxiv_id"],
            "arxiv_version": group["arxiv_version"],
            "source_filename": group["source_filename"],
            "page_numbers": page_numbers,
            "page_start": min(page_numbers),
            "page_end": max(page_numbers),
            "section_number": group["section_number"],
            "section_title": group["section_title"],
            "section_heading": group["section_heading"],
            "section_path": group["section_path"],
            "section_source": group["section_source"],
            "setting": setting_name,
            "tokenizer": TOKENIZER_NAME,
            "tokenizer_library": TOKENIZER_VERSION,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "token_count": actual_token_count,
            "group_id": f"{group['paper_id']}_group_{group_index:04d}",
            "group_token_start": start_token,
            "group_token_end": end_token,
            "start_char": start_char,
            "end_char": end_char,
            "source_parts": source_parts,
            "character_count": len(chunk_text),
            "text": chunk_text,
        }

        # Verify that the actual chunk never exceeds its token budget.
        if chunk["token_count"] > chunk_size:
            raise ValueError(
                f"Chunk exceeds token budget: {chunk['chunk_id']}"
            )

        chunks.append(chunk)
        chunk_index += 1

        if end_token == len(token_ids):
            break

        start_token += step

    return chunks


def build_chunks(
    encoding,
    setting_name: str,
    chunk_size: int,
    overlap: int,
) -> tuple[list[dict], dict]:

    all_chunks = []
    group_summaries = []

    for paper_id, input_path in INPUT_FILES.items():
        pages = load_jsonl(input_path)

        if not pages:
            raise ValueError(f"No page records found: {input_path}")

        if any(page["paper_id"] != paper_id for page in pages):
            raise ValueError(
                f"Unexpected paper_id in {input_path}"
            )

        groups = group_consecutive_segments(pages)

        paper_chunks = []

        for group_index, group in enumerate(groups):
            chunks = chunk_group(
                group=group,
                group_index=group_index,
                encoding=encoding,
                chunk_size=chunk_size,
                overlap=overlap,
                setting_name=setting_name,
            )

            paper_chunks.extend(chunks)

            full_text, _ = prepare_group_text(group)
            group_token_count = len(
                encoding.encode(full_text, disallowed_special=())
            )

            group_summaries.append({
                "paper_id": paper_id,
                "group_id": (
                    f"{paper_id}_group_{group_index:04d}"
                ),
                "section_path": group["section_path"],
                "token_count": group_token_count,
                "chunk_count": len(chunks),
            })

        all_chunks.extend(paper_chunks)

        print(
            f"{paper_id}: "
            f"{len(pages)} pages, "
            f"{len(groups)} consecutive section groups, "
            f"{len(paper_chunks)} chunks"
        )

    output_path = OUTPUT_DIR / f"setting_{setting_name.lower()}.jsonl"
    write_jsonl(output_path, all_chunks)

    token_counts = [
        chunk["token_count"]
        for chunk in all_chunks
    ]

    page_counts = [
        len(chunk["page_numbers"])
        for chunk in all_chunks
    ]

    summary = {
        "setting": setting_name,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "tokenizer": TOKENIZER_NAME,
        "tokenizer_library": TOKENIZER_VERSION,
        "chunk_count": len(all_chunks),
        "total_chunk_tokens": sum(token_counts),
        "average_chunk_tokens": (
            round(sum(token_counts) / len(token_counts), 2)
            if token_counts else 0
        ),
        "minimum_chunk_tokens": min(token_counts) if token_counts else 0,
        "maximum_chunk_tokens": max(token_counts) if token_counts else 0,
        "average_pages_per_chunk": (
            round(sum(page_counts) / len(page_counts), 2)
            if page_counts else 0
        ),
        "papers": {
            paper_id: sum(
                1 for chunk in all_chunks
                if chunk["paper_id"] == paper_id
            )
            for paper_id in INPUT_FILES
        },
        "groups": group_summaries,
        "output_file": str(output_path.relative_to(PROJECT_ROOT)),
    }

    return all_chunks, summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    encoding = tiktoken.get_encoding(TOKENIZER_NAME)

    print("=" * 75)
    print("THE RELAY - TOKEN-BASED CHUNKING")
    print("=" * 75)
    print("Tokenizer:", TOKENIZER_NAME)
    print("tiktoken version:", tiktoken.__version__)

    summaries = []

    for setting_name, config in CHUNK_CONFIGS.items():
        print("\n" + "-" * 75)
        print(
            f"SETTING {setting_name}: "
            f"chunk_size={config['chunk_size']}, "
            f"overlap={config['overlap']}"
        )

        _, summary = build_chunks(
            encoding=encoding,
            setting_name=setting_name,
            chunk_size=config["chunk_size"],
            overlap=config["overlap"],
        )

        summaries.append(summary)

        print("Chunks:", summary["chunk_count"])
        print("Total chunk tokens:", summary["total_chunk_tokens"])
        print("Average chunk tokens:", summary["average_chunk_tokens"])
        print(
            "Minimum / maximum:",
            summary["minimum_chunk_tokens"],
            "/",
            summary["maximum_chunk_tokens"],
        )

    output = {
        "tokenizer": TOKENIZER_NAME,
        "tokenizer_library": TOKENIZER_VERSION,
        "settings": summaries,
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("CHUNKING COMPLETE")
    print("Summary:", SUMMARY_PATH)
    print("=" * 75)


if __name__ == "__main__":
    main()
