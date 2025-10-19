import cohere
import os
from dotenv import load_dotenv

load_dotenv()

class Reranker:
    def __init__(self):
        self.co = cohere.Client(os.getenv("COHERE_API_KEY"))
    
    def rerank(self, query, documents, top_k=3):
        """
        Rerank documents using Cohere
        documents: list of dicts with 'text' key
        Returns: reranked documents with relevance scores
        """
        # Extract text for reranking
        texts = [doc['text'] for doc in documents]
        
        # Rerank using Cohere
        results = self.co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=texts,
            top_n=top_k,
            return_documents=True
        )
        
        # Map back to original documents with scores
        reranked = []
        for result in results.results:
            original_doc = documents[result.index]
            reranked.append({
                **original_doc,
                "rerank_score": result.relevance_score
            })
        
        return reranked