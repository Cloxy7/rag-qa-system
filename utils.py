import tiktoken
from PyPDF2 import PdfReader
from docx import Document
import re

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count tokens in text"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def chunk_text(text, chunk_size=1000, overlap=150):
    """
    Chunk text with overlap
    chunk_size: target tokens per chunk (default 1000)
    overlap: overlap tokens between chunks (default 15%)
    """
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
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
    
    # Chunk text
    chunks = chunk_text(text, chunk_size=1000, overlap=150)
    
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