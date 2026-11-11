from pathlib import Path
import pymupdf

PAPERS = [
    "attention_is_all_you_need_v7.pdf",
    "moshi_v1.pdf",
    "dynamic_chunking_v2.pdf",
]

PAPERS_DIR = Path("papers")


def compare_extraction(pdf_path: Path):

    print("\n" + "=" * 80)
    print("PAPER:", pdf_path.name)
    print("=" * 80)

    total_default = 0
    total_sorted = 0

    differences = []

    with pymupdf.open(pdf_path) as document:

        for page_index, page in enumerate(document):

            default_text = page.get_text("text").strip()

            sorted_text = page.get_text(
                "text",
                sort=True,
            ).strip()

            default_count = len(default_text)
            sorted_count = len(sorted_text)

            total_default += default_count
            total_sorted += sorted_count

            difference = sorted_count - default_count

            if difference != 0:

                differences.append({
                    "page": page_index + 1,
                    "default": default_count,
                    "sorted": sorted_count,
                    "difference": difference,
                })

    print("Total characters (default):", total_default)
    print("Total characters (sorted):", total_sorted)
    print("Difference:", total_sorted - total_default)

    print("\nPages with different character counts:")

    for item in differences[:15]:

        print(
            f"Page {item['page']:>3}: "
            f"default={item['default']:>6}, "
            f"sorted={item['sorted']:>6}, "
            f"delta={item['difference']:>6}"
        )

    print(
        "\nPages with different counts:",
        len(differences),
    )


if __name__ == "__main__":

    for filename in PAPERS:

        pdf_path = PAPERS_DIR / filename

        if not pdf_path.exists():
            print("MISSING:", pdf_path)
            continue

        compare_extraction(pdf_path)
