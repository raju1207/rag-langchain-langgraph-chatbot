import csv
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from src.config import BASE_DIR, DOCUMENTS_DIR


METADATA_FILE = BASE_DIR / "data" / "metadata.csv"


def safe_filename(text: str) -> str:
    """
    Convert a document ID/title into a safe filename.
    """

    text = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        text
    )

    text = text.strip()

    return text


def get_pdf_url(row):
    """
    Get the best available PDF URL from metadata.
    """

    url = row.get("url", "").strip()

    if url:
        return url

    return None


def download_pdf(
    url: str,
    output_path: Path
):
    """
    Download one PDF file.
    """

    print(f"Downloading: {url}")

    try:

        response = requests.get(
            url,
            timeout=60,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "RAG Research Project"
                )
            }
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        # Save the file
        output_path.write_bytes(
            response.content
        )

        print(
            f"Saved: {output_path.name}"
        )

        return True

    except Exception as error:

        print(
            f"FAILED: {url}"
        )

        print(
            f"Reason: {error}"
        )

        return False


def main():

    print("\n==============================")
    print("DOWNLOAD RAG DOCUMENTS")
    print("==============================\n")

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    print(
        f"Found {len(rows)} metadata records."
    )

    successful = 0
    skipped = 0
    failed = 0

    # --------------------------------------------------
    # For the first version, download only 10 documents.
    # --------------------------------------------------

    MAX_DOCUMENTS = 10

    for row in rows[:MAX_DOCUMENTS]:

        document_id = row.get(
            "id",
            ""
        ).strip()

        title = row.get(
            "title",
            ""
        ).strip()

        url = get_pdf_url(row)

        if not document_id or not url:

            print(
                f"Skipping invalid record: {title}"
            )

            skipped += 1

            continue

        filename = (
            safe_filename(document_id)
            + ".pdf"
        )

        output_path = (
            DOCUMENTS_DIR / filename
        )

        # Don't download again
        if output_path.exists():

            print(
                f"Already exists: {filename}"
            )

            skipped += 1

            continue

        success = download_pdf(
            url,
            output_path
        )

        if success:

            successful += 1

        else:

            failed += 1

        # Be polite to public servers
        time.sleep(1)

    print("\n==============================")
    print("DOWNLOAD SUMMARY")
    print("==============================")

    print(
        f"Successful : {successful}"
    )

    print(
        f"Skipped    : {skipped}"
    )

    print(
        f"Failed     : {failed}"
    )

    print(
        f"\nDocuments directory:"
        f"\n{DOCUMENTS_DIR}"
    )


if __name__ == "__main__":
    main()