"""
utils/pdf_reader.py — PDF Text Extraction

Decodes the base64-encoded file content delivered by Dash's dcc.Upload
component and extracts all readable text from the PDF using pypdf.

Only text-based PDFs are supported. Scanned PDFs (image-only) will return
an empty string, which is handled gracefully by the upload callback.
"""

import base64
import io

from pypdf import PdfReader


def extract_text_from_b64(b64_content: str) -> tuple[str, int]:
    """Decode a base64 PDF (as delivered by dcc.Upload) and return (text, page_count)."""
    raw = base64.b64decode(b64_content.split(",", 1)[1])
    reader = PdfReader(io.BytesIO(raw))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return text, len(reader.pages)
