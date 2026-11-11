from pathlib import Path
from collections import Counter
import pymupdf
import re

PAPERS = {
    "attention_is_all_you_need_v7.pdf": range(1, 16),
    "moshi_v1.pdf": range(1, 43),
    "dynamic_chunking_v2.pdf": range(1, 26),
}

PAPERS_DIR = Path("papers")


def inspect_paper(filename, page_numbers):
    pdf_path = PAPERS_DIR / filename

    with pymupdf.open(pdf_path) as document:
        font_sizes = []

        # Determine the typical font size for body text.
        for page in document:
            data = page.get_text("dict")

            for block in data["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        if span["text"].strip():
                            font_sizes.append(round(span["size"], 1))

        median_size = sorted(font_sizes)[len(font_sizes) // 2]

        print("\n" + "=" * 85)
        print("PAPER:", filename)
        print("Median span font size:", median_size)
        print("Common font sizes:", Counter(font_sizes).most_common(8))
        print("=" * 85)

        for page_number in page_numbers:
            page = document[page_number - 1]
            data = page.get_text("dict")

            print(f"\n--- PAGE {page_number} ---")

            for block in data["blocks"]:
                for line in block.get("lines", []):
                    spans = line["spans"]
                    if not spans:
                        continue

                    text = "".join(
                        span["text"] for span in spans
                    ).strip()

                    if not text or len(text) > 150:
                        continue

                    max_size = max(
                        round(span["size"], 1)
                        for span in spans
                    )

                    is_bold = any(
                        "bold" in span["font"].lower()
                        for span in spans
                    )

                    # Show typography that may indicate a heading.
                    if max_size >= median_size + 1.0 or (
                        is_bold and len(text) <= 100
                    ):
                        print(
                            f"size={max_size:>4}, "
                            f"bold={str(is_bold):<5} | {text}"
                        )


def main():
    for filename, pages in PAPERS.items():
        inspect_paper(filename, pages)


if __name__ == "__main__":
    main()
