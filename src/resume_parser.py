"""
Resume Parser Module
Extracts text from PDF and DOCX files uploaded by the user.
"""

import os
import tempfile
from pypdf import PdfReader


def extract_text(uploaded_file):
    """
    Extract text from an uploaded PDF or DOCX file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        str: Extracted text content from the document
    """
    file_extension = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    try:
        if file_extension == "docx":
            text = _extract_from_docx(tmp_path)
        else:
            text = _extract_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)

    return text


def _extract_from_pdf(file_path):
    """Extract text from a PDF file using PyPDF."""
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return "\n".join(pages)


def _extract_from_docx(file_path):
    """Extract text from a DOCX file using python-docx."""
    import docx
    doc = docx.Document(file_path)
    paragraphs = []
    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text)
    return "\n".join(paragraphs)
