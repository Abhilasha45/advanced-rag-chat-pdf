from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " "
        ]
    )

    chunks = []

    for page in pages:

        split_chunks = splitter.split_text(page["text"])

        for chunk in split_chunks:

            chunks.append(
                {
                    "page": page["page"],
                    "text": chunk
                }
            )

    return chunks