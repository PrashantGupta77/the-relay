from pathlib import Path
import pymupdf
import re

PAPERS = {
    "attention_is_all_you_need_v7.pdf": range(1, 16),
    "moshi_v1.pdf": range(1, 43),
    "dynamic_chunking_v2.pdf": range(1, 26),
}

PAPERS_DIR = Path("papers")
MIN_HEADING_SIZE = 10.5

NUMBERED = re.compile(
    r"^\d+(?:\.\d+)*\.?(?:\s+|$)"
)

def get_line_text(line):
    return "".join(
        span["text"] for span in line["spans"]
    ).strip()

def get_line_size(line):
    sizes = [
        span["size"]
        for span in line["spans"]
        if span["text"].strip()
    ]
    return round(max(sizes), 1) if sizes else 0

def inspect_paper(filename, page_numbers):
    print("\n" + "=" * 90)
    print("PAPER:", filename)
    print("=" * 90)

    with pymupdf.open(PAPERS_DIR / filename) as document:
        for page_number in page_numbers:
            page = document[page_number - 1]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                for line in block.get("lines", []):
                    text = get_line_text(line)
                    size = get_line_size(line)

                    if not text or size < MIN_HEADING_SIZE:
                        continue

                    if NUMBERED.match(text) and len(text) <= 140:
                        print(
                            f"Page {page_number:>2} | "
                            f"size={size:>4} pt | "
                            f"text={text}"
                        )

def main():
    for filename, pages in PAPERS.items():
        inspect_paper(filename, pages)

if __name__ == "__main__":
    main()
