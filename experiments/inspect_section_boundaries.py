from pathlib import Path
import pymupdf

PAGES = {
    "attention_is_all_you_need_v7.pdf": [3, 4, 5, 9],
    "moshi_v1.pdf": [2, 3, 4, 5, 8, 10, 11, 13, 15, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
    "dynamic_chunking_v2.pdf": [2, 3, 5, 7, 11, 13, 14, 16, 18, 19, 20, 21, 22, 23],
}

def main():
    for filename, page_numbers in PAGES.items():
        path = Path("papers") / filename

        print("\n" + "=" * 100)
        print("PAPER:", filename)
        print("=" * 100)

        with pymupdf.open(path) as document:
            for page_number in page_numbers:
                page = document[page_number - 1]
                lines = page.get_text("text").splitlines()

                print(f"\n--- PDF page {page_number} ---")

                for index, line in enumerate(lines):
                    line = line.strip()
                    if line:
                        print(f"{index:03d}: {line}")

if __name__ == "__main__":
    main()
