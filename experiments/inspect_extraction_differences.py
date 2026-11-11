from pathlib import Path
from difflib import SequenceMatcher
import pymupdf

PAPERS = {
    "attention_is_all_you_need_v7.pdf": [1, 9],
    "moshi_v1.pdf": [1, 15],
    "dynamic_chunking_v2.pdf": [3, 15],
}

PAPERS_DIR = Path("papers")
PREVIEW_CHARS = 1000


def inspect_page(pdf_path: Path, page_number: int):
    with pymupdf.open(pdf_path) as document:
        page = document[page_number - 1]

        default_text = page.get_text("text").strip()
        sorted_text = page.get_text("text", sort=True).strip()

    similarity = SequenceMatcher(
        None, default_text, sorted_text
    ).ratio()

    print("\n" + "-" * 80)
    print(f"PAGE {page_number}")
    print("-" * 80)
    print("Default characters:", len(default_text))
    print("Sorted characters:", len(sorted_text))
    print(f"Sequence similarity: {similarity:.3f}")

    print("\n[DEFAULT EXTRACTION]")
    print(default_text[:PREVIEW_CHARS])

    print("\n[SORTED EXTRACTION]")
    print(sorted_text[:PREVIEW_CHARS])


if __name__ == "__main__":
    for filename, pages in PAPERS.items():
        pdf_path = PAPERS_DIR / filename

        print("\n" + "=" * 80)
        print("PAPER:", filename)
        print("=" * 80)

        for page_number in pages:
            inspect_page(pdf_path, page_number)
