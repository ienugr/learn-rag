#!/usr/bin/env python3
"""
Advanced Knowledge Base with feedback loop and maintenance operations.
Demonstrates how to evolve the knowledge base based on usage patterns.
"""
import getpass
import json
from datetime import datetime
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
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class AdaptiveKnowledgeBase:
    """Knowledge base with feedback loop and maintenance capabilities."""
    
    def __init__(self, client, dimensions=512):
        self.client = client
        self.dimensions = dimensions
        self.documents = []
        self.embeddings = []
        self.unanswered_questions = []  # Track gaps in knowledge
        self.query_log = []  # Track all queries for analysis
    
    def add_document(self, text, metadata=None, doc_id=None):
        """Add or update a document in the knowledge base."""
        embedding = get_embedding(text, self.client, self.dimensions)
        
        # If doc_id provided, update existing document
        if doc_id is not None and 0 <= doc_id < len(self.documents):
            print(f"Updating document #{doc_id}")
            old_text = self.documents[doc_id]['text']
            self.documents[doc_id] = {
                'text': text,
                'metadata': metadata or {},
                'id': doc_id,
                'updated_at': datetime.now().isoformat(),
                'version': self.documents[doc_id].get('version', 1) + 1
            }
            self.embeddings[doc_id] = embedding
            print(f"  Old: {old_text[:60]}...")
            print(f"  New: {text[:60]}...")
        else:
            # Add new document
            doc_id = len(self.documents)
            self.documents.append({
                'text': text,
                'metadata': metadata or {},
                'id': doc_id,
                'created_at': datetime.now().isoformat(),
                'version': 1
            })
            self.embeddings.append(embedding)
            print(f"Added document #{doc_id}: {text[:60]}...")
        
        return doc_id
    
    def delete_document(self, doc_id):
        """Remove a document from the knowledge base."""
        if 0 <= doc_id < len(self.documents):
            deleted_doc = self.documents[doc_id]
            print(f"Deleting document #{doc_id}: {deleted_doc['text'][:60]}...")
            # Mark as deleted instead of removing (preserves IDs)
            self.documents[doc_id] = {
                'text': '[DELETED]',
                'metadata': {'deleted': True, 'deleted_at': datetime.now().isoformat()},
                'id': doc_id
            }
            self.embeddings[doc_id] = [0.0] * self.dimensions  # Zero vector
            return True
        return False
    
    def search(self, query, top_k=3, similarity_threshold=0.5):
        """Search for most relevant documents."""
        query_embedding = get_embedding(query, self.client, self.dimensions)
        
        similarities = []
        for i, doc_emb in enumerate(self.embeddings):
            # Skip deleted documents
            if self.documents[i].get('metadata', {}).get('deleted'):
                similarities.append(-1.0)
            else:
                similarities.append(cosine_similarity(query_embedding, doc_emb))
        
        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in ranked_indices:
            sim = similarities[idx]
            if sim >= similarity_threshold:
                results.append({
                    'document': self.documents[idx],
                    'similarity': sim
                })
        
        return results
    
    def ask(self, question, top_k=3, similarity_threshold=0.5):
        """Answer with strong grounding and feedback tracking."""
        # Log the query
        self.query_log.append({
            'question': question,
            'timestamp': datetime.now().isoformat()
        })
        
        results = self.search(question, top_k, similarity_threshold)
        
        # No relevant results found
        if not results:
            self._log_unanswered(question, reason="no_relevant_documents")
            return {
                'answer': "I don't have information about that in my knowledge base.",
                'sources': [],
                'status': 'unanswered',
                'reason': 'no_relevant_documents',
                'suggestion': 'Consider adding documentation about this topic.'
            }
        
        # Low confidence results
        max_similarity = max(r['similarity'] for r in results)
        if max_similarity < 0.7:
            self._log_unanswered(question, reason="low_confidence", results=results)
        
        context_text = "\n\n".join([
            f"[Source {i+1}]: {r['document']['text']}"
            for i, r in enumerate(results)
        ])
        
        system_prompt = """You are a knowledge base assistant. Follow these rules:
1. Answer ONLY using the provided sources
2. Cite every claim with [Source N]
3. If sources don't fully answer the question, say what's missing
4. Do not use external knowledge
5. Be precise and quote when helpful"""

        user_prompt = f"""{context_text}

Question: {question}

Answer with citations:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        
        answer = response.choices[0].message.content
        
        # Detect if LLM indicated missing information
        uncertain_phrases = [
            "don't have information",
            "not mentioned",
            "doesn't specify",
            "not found in",
            "missing information"
        ]
        
        if any(phrase in answer.lower() for phrase in uncertain_phrases):
            self._log_unanswered(question, reason="partial_answer", results=results)
        
        return {
            'answer': answer,
            'sources': results,
            'status': 'answered' if max_similarity >= 0.7 else 'low_confidence',
            'max_similarity': max_similarity
        }
    
    def _log_unanswered(self, question, reason, results=None):
        """Track questions that couldn't be answered well."""
        self.unanswered_questions.append({
            'question': question,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'similar_docs': [r['document']['id'] for r in results] if results else []
        })
    
    def get_knowledge_gaps(self):
        """Analyze unanswered questions to find knowledge gaps."""
        if not self.unanswered_questions:
            return []
        
        # Group by similarity to find common themes
        print(f"\n=== Knowledge Gaps Analysis ===")
        print(f"Total unanswered questions: {len(self.unanswered_questions)}\n")
        
        gaps = {}
        for uq in self.unanswered_questions:
            reason = uq['reason']
            if reason not in gaps:
                gaps[reason] = []
            gaps[reason].append(uq)
        
        summary = []
        for reason, questions in gaps.items():
            print(f"{reason}: {len(questions)} questions")
            for q in questions[:3]:  # Show top 3
                print(f"  - {q['question']}")
            if len(questions) > 3:
                print(f"  ... and {len(questions) - 3} more")
            print()
            
            summary.append({
                'reason': reason,
                'count': len(questions),
                'sample_questions': [q['question'] for q in questions[:5]]
            })
        
        return summary
    
    def suggest_new_content(self, question):
        """Generate a template for new content based on unanswered question."""
        prompt = f"""A user asked: "{question}"

Our knowledge base couldn't answer this. Generate a template for documentation that would answer this question.

Format:
1. What information is needed
2. Suggested documentation structure
3. Key points to cover"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a documentation expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        
        return response.choices[0].message.content
    
    def find_duplicates(self, similarity_threshold=0.95):
        """Find potentially duplicate or redundant documents."""
        duplicates = []
        
        for i in range(len(self.embeddings)):
            if self.documents[i].get('metadata', {}).get('deleted'):
                continue
                
            for j in range(i + 1, len(self.embeddings)):
                if self.documents[j].get('metadata', {}).get('deleted'):
                    continue
                
                sim = cosine_similarity(self.embeddings[i], self.embeddings[j])
                if sim >= similarity_threshold:
                    duplicates.append({
                        'doc1_id': i,
                        'doc2_id': j,
                        'similarity': sim,
                        'doc1_text': self.documents[i]['text'][:100],
                        'doc2_text': self.documents[j]['text'][:100]
                    })
        
        return duplicates
    
    def save(self, filepath):
        """Save knowledge base with all metadata."""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'dimensions': self.dimensions,
            'unanswered_questions': self.unanswered_questions,
            'query_log': self.query_log,
            'saved_at': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Knowledge base saved to {filepath}")
    
    def load(self, filepath):
        """Load knowledge base with all metadata."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self.dimensions = data['dimensions']
        self.unanswered_questions = data.get('unanswered_questions', [])
        self.query_log = data.get('query_log', [])
        print(f"Loaded {len(self.documents)} documents from {filepath}")


def demo_feedback_loop(client):
    """Demonstrate the feedback loop for evolving knowledge base."""
    kb = AdaptiveKnowledgeBase(client, dimensions=512)
    
    print("\n=== Initial Knowledge Base ===\n")
    
    # Initial knowledge
    kb.add_document(
        "The Payment Service uses Stripe API. It runs on port 8080.",
        metadata={"service": "payment"}
    )
    
    kb.add_document(
        "The User Service handles authentication. It uses JWT tokens.",
        metadata={"service": "user"}
    )
    
    print("\n=== Asking Questions ===\n")
    
    # Question 1: Should be answered
    print("Q1: What port does the Payment Service use?")
    result = kb.ask("What port does the Payment Service use?")
    print(f"Status: {result['status']}")
    print(f"Answer: {result['answer']}\n")
    
    # Question 2: Not in knowledge base
    print("Q2: What database does the Payment Service use?")
    result = kb.ask("What database does the Payment Service use?")
    print(f"Status: {result['status']}")
    print(f"Answer: {result['answer']}\n")
    
    # Question 3: Partially related
    print("Q3: How do I deploy the Payment Service?")
    result = kb.ask("How do I deploy the Payment Service?")
    print(f"Status: {result['status']}")
    print(f"Answer: {result['answer']}\n")
    
    # Analyze gaps
    kb.get_knowledge_gaps()
    
    # Get suggestion for missing content
    print("\n=== Suggested Content for Gap ===\n")
    suggestion = kb.suggest_new_content("What database does the Payment Service use?")
    print(suggestion)
    
    # Add new knowledge based on feedback
    print("\n\n=== Adding Missing Knowledge ===\n")
    kb.add_document(
        "The Payment Service uses PostgreSQL 14 as its primary database. "
        "Connection pooling is handled by PgBouncer. Database backups run daily at 2 AM UTC.",
        metadata={"service": "payment", "type": "database"}
    )
    
    # Ask again
    print("Q4: What database does the Payment Service use? (asking again)")
    result = kb.ask("What database does the Payment Service use?")
    print(f"Status: {result['status']}")
    print(f"Answer: {result['answer']}\n")
    
    # Update existing knowledge (fixing wrong info)
    print("\n=== Fixing Incorrect Information ===\n")
    print("Scenario: Port 8080 was wrong, should be 8090\n")
    kb.add_document(
        "The Payment Service uses Stripe API. It runs on port 8090 in production and port 8091 in staging.",
        metadata={"service": "payment"},
        doc_id=0  # Update first document
    )
    
    print("\nQ5: What port does the Payment Service use? (after fix)")
    result = kb.ask("What port does the Payment Service use?")
    print(f"Answer: {result['answer']}\n")
    
    # Find duplicates
    print("\n=== Checking for Duplicates ===\n")
    duplicates = kb.find_duplicates(similarity_threshold=0.85)
    if duplicates:
        for dup in duplicates:
            print(f"Similar docs (similarity: {dup['similarity']:.2f}):")
            print(f"  Doc {dup['doc1_id']}: {dup['doc1_text']}...")
            print(f"  Doc {dup['doc2_id']}: {dup['doc2_text']}...")
    else:
        print("No duplicates found")
    
    kb.save('adaptive_knowledge_base.json')


def main():
    api_key = getpass.getpass("Enter your OpenAI API key: ")
    client = OpenAI(api_key=api_key)
    
    demo_feedback_loop(client)


if __name__ == "__main__":
    main()
