import json
from pathlib import Path


SECTIONED_DIR = Path("data/sectioned_pages")

# PDF page numbers are one-based.
PAGES_TO_INSPECT = {
    "attention_is_all_you_need": [3, 4, 5],
    "dynamic_chunking": [4, 5, 6],
    "moshi": [15, 16, 17],
}


def load_pages(paper_id):
    path = SECTIONED_DIR / f"{paper_id}.jsonl"

    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def inspect_paper(paper_id, target_pages):
    pages = load_pages(paper_id)

    print("\n" + "=" * 100)
    print(f"PAPER: {paper_id}")
    print("=" * 100)

    for page in pages:
        page_number = page.get("page_number")

        if page_number not in target_pages:
            continue

        print(f"\n{'-' * 90}")
        print(f"PDF PAGE: {page_number}")
        print(f"{'-' * 90}")

        segments = page.get("section_segments", [])

        if not segments:
            print("No segments found.")
            continue

        for index, segment in enumerate(segments, start=1):
            print(f"\nSegment {index}")
            print("Section number:", segment.get("section_number"))
            print("Section title:", segment.get("section_title"))
            print("Section path:", segment.get("section_path"))
            print(
                "Source:",
                segment.get("section_source")
            )
            print(
                "Line range:",
                segment.get("line_start"),
                "-",
                segment.get("line_end")
            )

            excerpt = segment.get("text", "")
            excerpt = " ".join(excerpt.split())

            print("Text:", excerpt[:350])

            if len(excerpt) > 350:
                print("[excerpt truncated]")


def main():
    for paper_id, target_pages in PAGES_TO_INSPECT.items():
        inspect_paper(paper_id, target_pages)


if __name__ == "__main__":
    main()
