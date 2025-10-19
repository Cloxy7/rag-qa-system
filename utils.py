from PyPDF2 import PdfReader
from docx import Document
import re

def count_tokens(text, model="gpt-3.5-turbo"):
    """Estimate tokens (rough approximation: 1 token ≈ 4 characters)"""
    return len(text) // 4

def chunk_text(text, chunk_size=3000, overlap=450):
    """
    Chunk text by characters with overlap
    chunk_size: target characters per chunk (default 3000 ~ 750-1000 tokens)
    overlap: overlap characters between chunks (default 15%)
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

def extract_text_from_pdf(file_path):
    """Extract text from PDF"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX"""
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def clean_text(text):
    """Clean extracted text"""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def process_document(file_path, source_name):
    """
    Process document and return chunks with metadata
    Returns: list of dicts with {text, metadata}
    """
    # Extract text based on file type
    if file_path.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        text = extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    # Clean text
    text = clean_text(text)
    
    # Chunk text (3000 chars ≈ 750-1000 tokens)
    chunks = chunk_text(text, chunk_size=3000, overlap=450)
    
    # Add metadata
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        processed_chunks.append({
            "text": chunk,
            "metadata": {
                "source": source_name,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "tokens": count_tokens(chunk)
            }
        })
    
    return processed_chunks