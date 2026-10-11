
import csv
import re
import time
from pathlib import Path

import requests

from src.config import BASE_DIR, DOCUMENTS_DIR


METADATA_FILE = BASE_DIR / "data" / "metadata.csv"
MAX_DOCUMENTS = 10


def safe_filename(text: str) -> str:
    """Convert a document ID into a safe filename."""
    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    return text.strip().strip(".")


def get_pdf_url(row: dict) -> str:
    """Return the document URL after trimming whitespace."""
    return (row.get("url") or "").strip()


def download_pdf(url: str, output_path: Path) -> bool:
    """Download a PDF and save it only if its content is valid."""
    temp_path = output_path.with_suffix(".pdf.part")

    print(f"Downloading: {url}")

    try:
        with requests.get(
            url,
            timeout=90,
            allow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 RAG Research Project"
            },
            stream=True,
        ) as response:
            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            with open(temp_path, "wb") as file:
                first_chunk = True

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if not chunk:
                        continue

                    if first_chunk:
                        if not chunk.startswith(b"%PDF-"):
                            raise ValueError(
                                "URL did not return a valid PDF. "
                                f"Content-Type: {content_type}"
                            )
                        first_chunk = False

                    file.write(chunk)

                if first_chunk:
                    raise ValueError("Downloaded file is empty.")

        temp_path.replace(output_path)
        print(f"Saved: {output_path.name}")
        return True

    except Exception as error:
        print(f"FAILED: {url}")
        print(f"Reason: {error}")

        if temp_path.exists():
            temp_path.unlink()

        return False


def main():
    print("\n==============================")
    print("DOWNLOAD RAG DOCUMENTS")
    print("==============================\n")

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # utf-8-sig removes a possible UTF-8 BOM from the first CSV header.
    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("metadata.csv has no header row.")

        # Normalize header names to avoid whitespace issues.
        reader.fieldnames = [
            name.strip() if name else name
            for name in reader.fieldnames
        ]

        required_columns = {"id", "title", "url"}
        missing = required_columns - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"Missing CSV columns: {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        rows = list(reader)

    print(f"Found {len(rows)} metadata records.")
    print(f"Downloading up to {MAX_DOCUMENTS} documents.\n")

    successful = 0
    skipped = 0
    failed = 0

    for row in rows[:MAX_DOCUMENTS]:
        document_id = (row.get("id") or "").strip()
        title = (row.get("title") or "").strip()
        url = get_pdf_url(row)

        if not document_id or not url:
            print(
                f"Skipping invalid record: {title!r} "
                f"(id={document_id!r}, url={url!r})"
            )
            skipped += 1
            continue

        filename = f"{safe_filename(document_id)}.pdf"
        output_path = DOCUMENTS_DIR / filename

        if output_path.exists() and output_path.stat().st_size > 0:
            with open(output_path, "rb") as existing_file:
                if existing_file.read(5) == b"%PDF-":
                    print(f"Already exists: {filename}")
                    skipped += 1
                    continue

            print(f"Existing file is invalid; replacing: {filename}")
            output_path.unlink()

        if download_pdf(url, output_path):
            successful += 1
        else:
            failed += 1

        # Be polite to public servers.
        time.sleep(1)

    print("\n==============================")
    print("DOWNLOAD SUMMARY")
    print("==============================")
    print(f"Successful : {successful}")
    print(f"Skipped    : {skipped}")
    print(f"Failed     : {failed}")
    print(f"Directory  : {DOCUMENTS_DIR}")


if __name__ == "__main__":
    main()