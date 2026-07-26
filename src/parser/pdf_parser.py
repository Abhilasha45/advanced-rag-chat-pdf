import fitz
import re


def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        page_text = page.get_text("text")

        page_text = re.sub(r"\n+", "\n", page_text)
        page_text = re.sub(r"[ \t]+", " ", page_text)

        pages.append(
            {
                "page": page_number,
                "text": page_text.strip()
            }
        )

    document.close()

    return pages