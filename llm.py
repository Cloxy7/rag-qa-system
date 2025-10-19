from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

class LLM:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
    
    def generate_answer(self, query, chunks):
        """
        Generate answer with inline citations
        chunks: list of dicts with 'text' key
        Returns: dict with answer and token usage
        """
        # Build context with citations
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"[{i+1}] {chunk['text']}\n\n"
        
        # Create prompt
        prompt = f"""You are a helpful assistant that answers questions based on provided context.

IMPORTANT RULES:
1. Use ONLY the information from the context below
2. Include inline citations like [1], [2] for every claim
3. If the context doesn't contain enough information, say "I cannot answer this based on the provided documents"
4. Be concise but complete

Context:
{context}

Question: {query}

Answer with citations:"""
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )
        
        answer = response.choices[0].message.content
        
        # Get token usage
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        return {
            "answer": answer,
            "usage": usage
        }
    
    def estimate_cost(self, usage):
        """
        Estimate cost (Groq is free but this is for demonstration)
        """
        # Example pricing (adjust based on actual provider)
        prompt_cost_per_1k = 0.0001  # $0.0001 per 1K tokens
        completion_cost_per_1k = 0.0002
        
        prompt_cost = (usage['prompt_tokens'] / 1000) * prompt_cost_per_1k
        completion_cost = (usage['completion_tokens'] / 1000) * completion_cost_per_1k
        
        return {
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": prompt_cost + completion_cost
        }