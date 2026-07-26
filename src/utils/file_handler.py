from pathlib import Path

UPLOAD_DIR = Path("data/uploads")


def save_uploaded_files(uploaded_files):
    """
    Save uploaded PDFs into data/uploads.
    """

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for file in uploaded_files:
        file_path = UPLOAD_DIR / file.name

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        saved_files.append(file_path)

    return saved_files