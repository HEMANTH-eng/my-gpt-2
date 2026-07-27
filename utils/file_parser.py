import io
import re
from typing import Dict, Union

from utils.logger import get_logger

logger = get_logger("file_parser")


def parse_uploaded_file(filename: str, content_bytes: bytes) -> Dict[str, Union[str, int]]:
    """Parses uploaded PDF, DOCX, or TXT file bytes into clean text content.

    Args:
        filename: Name of the uploaded file.
        content_bytes: Binary contents of the file.

    Returns:
        Dictionary containing 'filename', 'file_type', 'file_size', and 'text'.
    """
    ext = filename.split(".")[-1].lower() if "." in filename else "txt"
    file_size = len(content_bytes)
    extracted_text = ""

    if ext == "txt":
        extracted_text = _parse_txt(content_bytes)
    elif ext == "pdf":
        extracted_text = _parse_pdf(content_bytes)
    elif ext in ("docx", "doc"):
        extracted_text = _parse_docx(content_bytes)
    else:
        extracted_text = _parse_txt(content_bytes)

    # Sanitize & truncate text for LLM context injection if needed
    cleaned_text = re.sub(r"\s+", " ", extracted_text).strip()

    logger.info(f"Parsed document '{filename}' ({ext}): extracted {len(cleaned_text)} characters.")

    return {
        "filename": filename,
        "file_type": ext,
        "file_size": file_size,
        "text": cleaned_text,
    }


def _parse_txt(content_bytes: bytes) -> str:
    """Parses plain text files with UTF-8 / latin-1 fallback decoding."""
    try:
        return content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return content_bytes.decode("latin-1", errors="ignore")


def _parse_pdf(content_bytes: bytes) -> str:
    """Extracts text from PDF bytes using pypdf if available, else regex text fallback."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content_bytes))
        pages_text = [page.extract_text() for page in reader.pages if page.extract_text()]
        if pages_text:
            return "\n".join(pages_text)
    except Exception:
        pass

    # Fallback text extraction using regex stream matching
    text_matches = re.findall(rb"\((.*?)\)\s*Tj", content_bytes)
    if text_matches:
        return " ".join(match.decode("latin-1", errors="ignore") for match in text_matches)
    return _parse_txt(content_bytes)



def _parse_docx(content_bytes: bytes) -> str:
    """Extracts text from DOCX bytes using python-docx if available, else zip XML fallback."""
    try:
        import docx
        doc = docx.Document(io.BytesIO(content_bytes))
        return "\n".join([p.text for p in doc.paragraphs if p.text])
    except ImportError:
        try:
            import zipfile
            from xml.etree import ElementTree
            with zipfile.ZipFile(io.BytesIO(content_bytes)) as zf:
                xml_content = zf.read("word/document.xml")
                tree = ElementTree.fromstring(xml_content)
                text_nodes = tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
                return "".join([node.text for node in text_nodes if node.text])
        except Exception:
            return _parse_txt(content_bytes)
