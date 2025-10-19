# 🤖 RAG Q&A System

**Live Demo**: https://your-deployed-app.com
**GitHub Repo**: https://github.com/your-username/rag-qa-system

**Developer**: [Your Name]
**Resume**: [Link to your resume/LinkedIn]

---

## 📋 Architecture Diagram

[Add a simple flowchart here - see below]

## 🏗️ System Architecture
```
User Input (File/Text)
    ↓
Text Extraction & Cleaning (PyPDF2, python-docx)
    ↓
Chunking (1000 tokens, 15% overlap - tiktoken)
    ↓
Embedding (all-MiniLM-L6-v2, 384 dimensions)
    ↓
Vector Database (Pinecone - Serverless, AWS us-east-1)

Query Flow:
User Query
    ↓
Embedding (all-MiniLM-L6-v2)
    ↓
Retrieval (Top 10, Cosine Similarity)
    ↓
Reranking (Cohere Rerank v3 - Top 3)
    ↓
LLM Generation (Groq Llama 3.1 8B, temp=0.3)
    ↓
Answer with Citations [1], [2], [3]
```

## ⚙️ Configuration Details

### Chunking Strategy
- **Size**: 1000 tokens (~750-850 words)
- **Overlap**: 150 tokens (~15%)
- **Rationale**: Maintains context continuity while keeping chunks semantically coherent

### Retrieval Settings
- **Initial Retrieval**: Top 10 from Pinecone (cosine similarity)
- **Reranker**: Cohere Rerank v3 (top 3)
- **Rationale**: Cast wide net, then precision rerank for best results

### LLM Configuration
- **Model**: Llama 3.1 8B Instant (via Groq)
- **Temperature**: 0.3 (focused, deterministic)
- **Max Tokens**: 800
- **Prompt**: Enforces citation requirement and document grounding

### Providers Used
- **Vector DB**: Pinecone (Serverless, free tier)
- **Embeddings**: SentenceTransformers (local)
- **Reranker**: Cohere (free tier - 1000 reqs/month)
- **LLM**: Groq (free tier with rate limits)

## 📝 Remarks

### Limitations & Trade-offs
- **File Size**: Limited to 16MB per upload
- **API Rate Limits**: 
  - Groq: ~30 requests/min on free tier
  - Cohere: 1000 reranking requests/month
  - Pinecone: 100K vectors on free tier
- **PDF Extraction**: Cannot handle scanned PDFs (OCR not implemented)
- **Context Length**: Using 1000 token chunks limits single-chunk context

### What I'd Do Next (Given More Time)
- [ ] Add conversation memory for follow-up questions
- [ ] Implement OCR for scanned documents
- [ ] Add support for CSV, Excel files
- [ ] Multi-language support
- [ ] Implement hybrid search (keyword + semantic)
- [ ] Add document version control
- [ ] Batch processing for large documents

### Design Decisions
- **Why Pinecone?** Serverless, scales automatically, free tier sufficient
- **Why Cohere Rerank?** Significantly improves precision over pure vector search
- **Why Groq?** Fast inference (much faster than OpenAI), free tier
- **Why 1000 token chunks?** Balance between context and retrieval precision

---

[Rest of your existing README content]
