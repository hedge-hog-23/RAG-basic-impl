# file_loader.py
import os
from pypdf import PdfReader

def load_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)

def load_text(path):
    return open(path, "r", encoding="utf-8").read()

def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".txt":
        return load_text(path)
    raise ValueError(f"Unsupported file type: {ext}")