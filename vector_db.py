from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

class VectorDB:
    def __init__(self, index_name="rag-system"):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = index_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        
        # Create index if doesn't exist
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"✅ Created index: {index_name}")
        
        self.index = self.pc.Index(index_name)
    
    def add_documents(self, chunks):
        """
        Add documents to Pinecone
        chunks: list of dicts with {text, metadata}
        """
        vectors = []
        for chunk in chunks:
            # Generate unique ID
            chunk_id = str(uuid.uuid4())
            
            # Create embedding
            embedding = self.model.encode(chunk["text"]).tolist()
            
            # Prepare vector
            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    **chunk["metadata"]
                }
            })
        
        # Upload to Pinecone (batch upsert)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"✅ Added {len(vectors)} chunks to Pinecone")
        return len(vectors)
    
    def search(self, query, top_k=10):
        """
        Search for similar chunks
        Returns: list of matches with metadata
        """
        query_embedding = self.model.encode(query).tolist()
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        matches = []
        for match in results['matches']:
            matches.append({
                "text": match['metadata']['text'],
                "source": match['metadata'].get('source', 'unknown'),
                "chunk_id": match['metadata'].get('chunk_id', 0),
                "score": float(match['score'])
            })
        
        return matches
    
    def clear_index(self):
        """Delete all vectors from index"""
        self.index.delete(delete_all=True)
        print("✅ Cleared index")
    
    def get_stats(self):
        """Get index statistics"""
        return self.index.describe_index_stats()