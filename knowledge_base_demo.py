#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) demo using embeddings as knowledge base.
This demonstrates how to use embeddings for semantic search and Q&A.
"""
import getpass
import json
from openai import OpenAI
import numpy as np


def get_embedding(text, client, dimensions=512):
    """Get embedding for a single text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        dimensions=dimensions
    )
    return response.data[0].embedding


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return np.dot(a, b) / denom


class KnowledgeBase:
    """Simple vector-based knowledge base using embeddings."""
    
    def __init__(self, client, dimensions=512):
        self.client = client
        self.dimensions = dimensions
        self.documents = []
        self.embeddings = []
    
    def add_document(self, text, metadata=None):
        """Add a document to the knowledge base."""
        embedding = get_embedding(text, self.client, self.dimensions)
        self.documents.append({
            'text': text,
            'metadata': metadata or {},
            'id': len(self.documents)
        })
        self.embeddings.append(embedding)
        print(f"Added document #{len(self.documents)}: {text[:60]}...")
    
    def search(self, query, top_k=3):
        """Search for most relevant documents using semantic similarity."""
        query_embedding = get_embedding(query, self.client, self.dimensions)
        
        similarities = [
            cosine_similarity(query_embedding, doc_emb)
            for doc_emb in self.embeddings
        ]
        
        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in ranked_indices:
            results.append({
                'document': self.documents[idx],
                'similarity': similarities[idx]
            })
        
        return results
    
    def ask(self, question, top_k=3):
        """Answer a question using relevant context from knowledge base."""
        results = self.search(question, top_k)
        
        context = "\n\n".join([
            f"[Context {i+1}]: {r['document']['text']}"
            for i, r in enumerate(results)
        ])

        print("Generated context for question:\n", context)
        
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain relevant information, say so."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return {
            'answer': response.choices[0].message.content,
            'sources': results
        }
    
    def save(self, filepath):
        """Save knowledge base to file."""
        data = {
            'documents': self.documents,
            'embeddings': [emb for emb in self.embeddings],
            'dimensions': self.dimensions
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"Knowledge base saved to {filepath}")
    
    def load(self, filepath):
        """Load knowledge base from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self.dimensions = data['dimensions']
        print(f"Loaded {len(self.documents)} documents from {filepath}")


def main():
    api_key = getpass.getpass("Enter your OpenAI API key: ")
    client = OpenAI(api_key=api_key)
    
    kb = KnowledgeBase(client, dimensions=512)
    
    # Example: Team/Service domain knowledge
    print("\n=== Building Knowledge Base ===\n")
    
    kb.add_document(
        "The Payment Service is a microservice responsible for processing all payment transactions. "
        "It uses Stripe API for credit card processing and supports multiple currencies. "
        "The service is written in Python using FastAPI framework.",
        metadata={"service": "payment", "type": "overview"}
    )
    
    kb.add_document(
        "The Payment Service exposes a POST /api/v1/payments endpoint that accepts amount, currency, "
        "and customer_id as required parameters. It returns a transaction_id on success. "
        "Rate limit is 100 requests per minute per API key.",
        metadata={"service": "payment", "type": "api"}
    )
    
    kb.add_document(
        "For Payment Service deployment, use the deploy.sh script in the /scripts directory. "
        "Required environment variables: STRIPE_API_KEY, DATABASE_URL, REDIS_URL. "
        "The service runs on port 8080 in production.",
        metadata={"service": "payment", "type": "deployment"}
    )
    
    kb.add_document(
        "The User Service manages user authentication and authorization. "
        "It uses JWT tokens with 24-hour expiration. The service integrates with Auth0 for SSO. "
        "User data is stored in PostgreSQL database.",
        metadata={"service": "user", "type": "overview"}
    )
    
    kb.add_document(
        "The Analytics Team is responsible for data pipeline maintenance, dashboard creation, "
        "and reporting. They use Apache Airflow for ETL jobs and Tableau for visualization. "
        "Contact: analytics@company.com, Slack: #analytics",
        metadata={"team": "analytics", "type": "info"}
    )
    
    # Demonstrate semantic search
    print("\n=== Semantic Search Demo ===\n")
    
    query = "How do I process credit cards?"
    print(f"Query: {query}\n")
    results = kb.search(query, top_k=2)
    
    for i, result in enumerate(results, 1):
        print(f"Result {i} (similarity: {result['similarity']:.3f}):")
        print(f"  {result['document']['text'][:100]}...")
        print()
    
    # Demonstrate Q&A
    print("\n=== Q&A Demo ===\n")
    
    questions = [
        "What technology does the Payment Service use?",
        "What port does the payment service run on?",
        "How can I contact the Analytics team?"
    ]
    
    for question in questions:
        print(f"Q: {question}")
        result = kb.ask(question, top_k=2)
        print(f"A: {result['answer']}")
        print(f"Sources: {len(result['sources'])} documents used")
        print()
    
    # Save knowledge base
    kb.save('knowledge_base.json')
    
    print("\n=== Try your own questions ===")
    print("(Press Ctrl+C to exit)\n")
    
    try:
        while True:
            user_question = input("Your question: ").strip()
            if not user_question:
                continue
            
            result = kb.ask(user_question, top_k=3)
            print(f"\nAnswer: {result['answer']}\n")
            print("Relevant sources:")
            for i, src in enumerate(result['sources'], 1):
                print(f"  {i}. (similarity: {src['similarity']:.3f}) {src['document']['text'][:80]}...")
            print()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    main()
