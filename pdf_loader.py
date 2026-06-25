import tempfile
import os

try:
    from langchain_community.document_loaders import PyPDFLoader
    _HAS_LANGCHAIN_LOADER = True
except Exception:
    _HAS_LANGCHAIN_LOADER = False

from pypdf import PdfReader


class SimpleDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


def load_pdf(uploaded_file):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name

    try:
        if _HAS_LANGCHAIN_LOADER:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
        else:
            reader = PdfReader(pdf_path)
            documents = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                documents.append(SimpleDoc(text, metadata={"page": i + 1}))
        return documents
    finally:
        try:
            os.remove(pdf_path)
        except Exception:
            pass