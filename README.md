#  RAG Q&A System

** Live Demo**: https://huggingface.co/spaces/be22b047/rag-qa-system

---

##  Overview

A production-ready Retrieval-Augmented Generation (RAG) system that allows users to upload documents or paste text, then ask questions with AI-generated answers backed by inline citations.

##  Features

-  Upload text directly via web interface
-  Intelligent text chunking (3000 chars, 15% overlap)
-  Semantic search with vector embeddings (all-MiniLM-L6-v2, 384 dimensions)
-  Two-stage retrieval: Initial top-10 + Cohere reranking to top-3
-  LLM-generated answers with inline citations [1], [2], [3]
-  Performance metrics displayed (tokens, timing)
-  Clean Gradio web interface

---

##  System Architecture
```
User Input (Text)
    ↓
Text Cleaning & Processing
    ↓
Chunking (3000 chars, 15% overlap)
    ↓
Embedding (all-MiniLM-L6-v2)
    ↓
Vector Database (Pinecone - Serverless)

Query Flow:
User Query
    ↓
Embedding (all-MiniLM-L6-v2)
    ↓
Retrieval (Top 10, Cosine Similarity)
    ↓
Reranking (Cohere Rerank v3 - Top 3)
    ↓
LLM Generation (Groq Llama 3.1 8B)
    ↓
Answer with Citations
```

---

##  Tech Stack

| Component | Technology | Details |
|-----------|------------|---------|
| **Frontend** | Gradio 4.44.0 | Interactive web UI |
| **Backend** | Python 3.10 | Core application |
| **Embeddings** | all-MiniLM-L6-v2 | 384 dimensions, SentenceTransformers |
| **Vector DB** | Pinecone | Serverless, AWS us-east-1 |
| **Reranker** | Cohere Rerank v3 | Precision reranking |
| **LLM** | Groq (Llama 3.1 8B) | Answer generation |
| **Deployment** | Hugging Face Spaces | Free tier, auto-scaling |

---

##  Configuration Details

### Chunking Strategy
- **Size**: 3000 characters (~750-1000 tokens)
- **Overlap**: 450 characters (~15%)
- **Method**: Character-based (deployment-optimized)
- **Rationale**: Balances context preservation with retrieval precision

### Retrieval Pipeline
- **Stage 1**: Retrieve top 10 results from Pinecone (cosine similarity)
- **Stage 2**: Rerank to top 3 using Cohere Rerank v3
- **Why two-stage?**: Maximizes recall in stage 1, precision in stage 2

### LLM Configuration
- **Model**: Llama 3.1 8B Instant (via Groq)
- **Temperature**: 0.3 (focused, deterministic)
- **Max Tokens**: 800
- **Prompt Engineering**: Enforces document grounding and mandatory citations

---

##  Remarks

### Known Limitations & Trade-offs

**API Rate Limits:**
- Groq: ~30 requests/min (free tier)
- Cohere: 1000 rerank requests/month (free tier)
- Pinecone: 100K vectors, 1 index (free tier)

**Technical Constraints:**
- Character-based chunking used instead of token-based (tiktoken) for deployment compatibility
- CPU-only inference (no GPU acceleration)
- Single Pinecone index (no namespace separation)

**Document Support:**
- Text input via paste only (file upload removed for simplicity)
- No OCR support for scanned documents
- Max practical text size: ~50K characters per upload

### Design Decisions

**Why Pinecone?**
- Serverless architecture (no infrastructure management)
- Consistent low-latency queries
- Free tier sufficient for demo

**Why Cohere Rerank?**
- Significantly improves precision over pure vector search
- Cost-effective on free tier
- Better than implementing custom reranking

**Why Groq?**
- Extremely fast inference (~500 tokens/sec)
- Free tier generous for development
- Simple API integration

**Why Character-based Chunking?**
- Avoids tiktoken dependency (requires C++ compiler)
- Reduces deployment image size by ~3GB
- Minimal accuracy impact for chunking purposes

### Future Improvements

- [ ] Add file upload support (PDF, DOCX, TXT)
- [ ] Implement conversation memory for follow-up questions
- [ ] Add OCR support (Tesseract) for scanned documents
- [ ] Support CSV/Excel file analysis
- [ ] Hybrid search (BM25 + semantic)
- [ ] Multi-language document support
- [ ] Batch document processing
- [ ] Custom chunking strategies per document type
- [ ] Add analytics dashboard

---

## 🚀 Local Setup
```bash
# Clone repository
git clone https://github.com/Cloxy7/rag-qa-system.git
cd rag-qa-system

# Install dependencies
pip install -r requirements.txt

# Create .env file with API keys
echo "PINECONE_API_KEY=your_key" >> .env
echo "GROQ_API_KEY=your_key" >> .env
echo "COHERE_API_KEY=your_key" >> .env

# Run locally
python app.py
```

---

##  License

MIT License

---

**Built with ❤️ for efficient document Q&A**
