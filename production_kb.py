#!/usr/bin/env python3
"""
Production-Grade Knowledge Base with Embeddings, Reranking, and Adaptive Learning.

Features:
- Semantic search using embeddings (text-embedding-3-small)
- Reranking with cross-encoder for improved relevance
- Strong grounding to prevent hallucination
- Adaptive learning from unanswered questions
- CRUD operations with versioning
- Gap analysis and duplicate detection
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


class CrossEncoderReranker:
    """
    Reranker using LLM as cross-encoder to score query-document pairs.
    Cross-encoders jointly encode query and document for better relevance scoring.
    """
    
    def __init__(self, client):
        self.client = client
    
    def score_relevance(self, query, document_text):
        """
        Score how relevant a document is to a query using cross-encoding approach.
        Returns score 0-10 and explanation.
        """
        prompt = f"""Rate the relevance of the document to the query on a scale of 0-10.

Query: {query}

Document: {document_text}

Scoring criteria:
- 10: Perfect match, directly answers the query
- 7-9: Highly relevant, contains most needed information
- 4-6: Partially relevant, has some related information
- 1-3: Tangentially related, mentions similar topics
- 0: Not relevant at all

Respond in JSON format:
{{
  "score": <number 0-10>,
  "reason": "<brief explanation>",
  "key_match": "<what part matches the query>"
}}"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a relevance scoring expert. Be strict and accurate."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def rerank(self, query, candidates, top_k=3):
        """
        Rerank candidates using cross-encoder scores.
        
        Args:
            query: Search query
            candidates: List of {document, similarity} dicts from initial retrieval
            top_k: Number of results to return after reranking
        
        Returns:
            Reranked list with cross-encoder scores
        """
        print(f"\n[Reranking] Processing {len(candidates)} candidates...")
        
        scored = []
        for i, candidate in enumerate(candidates):
            doc_text = candidate['document']['text']
            
            # Get cross-encoder score
            ce_score = self.score_relevance(query, doc_text)
            
            scored.append({
                'document': candidate['document'],
                'embedding_similarity': candidate['similarity'],
                'cross_encoder_score': ce_score['score'],
                'cross_encoder_reason': ce_score['reason'],
                'key_match': ce_score.get('key_match', ''),
                'final_score': ce_score['score']  # Use cross-encoder as primary
            })
            
            print(f"  [{i+1}] CE:{ce_score['score']}/10, Emb:{candidate['similarity']:.3f} - {ce_score['reason'][:60]}...")
        
        # Sort by cross-encoder score (more accurate than embedding similarity)
        reranked = sorted(scored, key=lambda x: x['final_score'], reverse=True)[:top_k]
        
        print(f"[Reranking] Selected top {len(reranked)} results")
        return reranked


class ProductionKnowledgeBase:
    """
    Production-grade knowledge base with embeddings, reranking, and adaptive learning.
    """
    
    def __init__(self, client, dimensions=512, use_reranking=True):
        self.client = client
        self.dimensions = dimensions
        self.documents = []
        self.embeddings = []
        self.unanswered_questions = []
        self.query_log = []
        self.use_reranking = use_reranking
        
        if use_reranking:
            self.reranker = CrossEncoderReranker(client)
            print("[Init] Reranking enabled - will use cross-encoder for better relevance")
        else:
            self.reranker = None
            print("[Init] Reranking disabled - using embedding similarity only")
    
    def add_document(self, text, metadata=None, doc_id=None):
        """Add or update a document in the knowledge base."""
        embedding = get_embedding(text, self.client, self.dimensions)
        
        if doc_id is not None and 0 <= doc_id < len(self.documents):
            print(f"[Update] Document #{doc_id}")
            old_text = self.documents[doc_id]['text']
            self.documents[doc_id] = {
                'text': text,
                'metadata': metadata or {},
                'id': doc_id,
                'updated_at': datetime.now().isoformat(),
                'version': self.documents[doc_id].get('version', 1) + 1
            }
            self.embeddings[doc_id] = embedding
            print(f"  Old: {old_text[:50]}...")
            print(f"  New: {text[:50]}...")
        else:
            doc_id = len(self.documents)
            self.documents.append({
                'text': text,
                'metadata': metadata or {},
                'id': doc_id,
                'created_at': datetime.now().isoformat(),
                'version': 1
            })
            self.embeddings.append(embedding)
            print(f"[Add] Document #{doc_id}: {text[:50]}...")
        
        return doc_id
    
    def delete_document(self, doc_id):
        """Soft delete a document from the knowledge base."""
        if 0 <= doc_id < len(self.documents):
            deleted_doc = self.documents[doc_id]
            print(f"[Delete] Document #{doc_id}: {deleted_doc['text'][:50]}...")
            self.documents[doc_id] = {
                'text': '[DELETED]',
                'metadata': {'deleted': True, 'deleted_at': datetime.now().isoformat()},
                'id': doc_id
            }
            self.embeddings[doc_id] = [0.0] * self.dimensions
            return True
        return False
    
    def search(self, query, top_k=10, similarity_threshold=0.3):
        """
        Initial retrieval using embedding similarity.
        Returns more candidates (top_k) for reranking to filter.
        """
        query_embedding = get_embedding(query, self.client, self.dimensions)
        
        similarities = []
        for i, doc_emb in enumerate(self.embeddings):
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
    
    def ask(self, question, top_k=3, rerank_candidates=10, similarity_threshold=0.3):
        """
        Answer question using two-stage retrieval:
        1. Embedding-based initial retrieval (fast, broad)
        2. Cross-encoder reranking (slow, accurate)
        """
        print(f"\n{'='*80}")
        print(f"[Query] {question}")
        print(f"{'='*80}")
        
        # Log query
        self.query_log.append({
            'question': question,
            'timestamp': datetime.now().isoformat()
        })
        
        # Stage 1: Embedding-based retrieval (cast wide net)
        print(f"[Stage 1] Embedding search (retrieving top {rerank_candidates} candidates)...")
        candidates = self.search(query=question, top_k=rerank_candidates, similarity_threshold=similarity_threshold)
        
        if not candidates:
            self._log_unanswered(question, reason="no_relevant_documents")
            return {
                'answer': "I don't have information about that in my knowledge base.",
                'sources': [],
                'status': 'unanswered',
                'reason': 'no_relevant_documents',
                'suggestion': 'Consider adding documentation about this topic.'
            }
        
        print(f"[Stage 1] Found {len(candidates)} candidates (similarity >= {similarity_threshold})")
        for i, c in enumerate(candidates[:5]):
            print(f"  [{i+1}] Similarity: {c['similarity']:.3f} - {c['document']['text'][:60]}...")
        
        # Stage 2: Reranking with cross-encoder (precision filter)
        if self.use_reranking and len(candidates) > 0:
            results = self.reranker.rerank(question, candidates, top_k=top_k)
            
            # Check if best result meets quality threshold
            if results and results[0]['final_score'] < 4.0:
                self._log_unanswered(question, reason="low_quality_match", results=results)
                return {
                    'answer': f"I found some related information, but it doesn't directly answer your question. The closest match only scored {results[0]['final_score']}/10 for relevance.",
                    'sources': results,
                    'status': 'low_confidence',
                    'reason': 'low_quality_match',
                    'suggestion': 'Consider adding more specific documentation about this topic.'
                }
        else:
            # No reranking - use embedding similarity only
            results = candidates[:top_k]
            max_similarity = max(r['similarity'] for r in results)
            
            if max_similarity < 0.5:
                self._log_unanswered(question, reason="low_confidence", results=results)
        
        # Build context with source citations
        context_text = "\n\n".join([
            f"[Source {i+1}] (Relevance: {r.get('cross_encoder_score', r.get('similarity', 0))}/10)\n{r['document']['text']}"
            for i, r in enumerate(results)
        ])
        
        # Generate answer with strong grounding
        system_prompt = """You are a knowledge base assistant. Follow these rules strictly:

1. Answer ONLY using the provided sources
2. Cite every claim with [Source N]
3. If sources don't fully answer the question, explicitly state what's missing
4. Do not use external knowledge or make assumptions
5. Be precise and quote when helpful
6. If relevance scores are low (<7/10), mention uncertainty"""

        user_prompt = f"""{context_text}

Question: {question}

Answer with inline citations:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        
        answer = response.choices[0].message.content
        
        # Detect partial answers
        uncertain_phrases = [
            "don't have information",
            "not mentioned",
            "doesn't specify",
            "not found in",
            "missing information",
            "doesn't say"
        ]
        
        if any(phrase in answer.lower() for phrase in uncertain_phrases):
            self._log_unanswered(question, reason="partial_answer", results=results)
        
        # Calculate confidence
        if self.use_reranking:
            max_score = max(r.get('final_score', 0) for r in results)
            status = 'answered' if max_score >= 7 else 'low_confidence'
        else:
            max_score = max(r.get('similarity', 0) for r in results)
            status = 'answered' if max_score >= 0.7 else 'low_confidence'
        
        print(f"\n[Result] Status: {status}")
        
        return {
            'answer': answer,
            'sources': results,
            'status': status,
            'max_relevance': max_score
        }
    
    def _log_unanswered(self, question, reason, results=None):
        """Track questions that couldn't be answered well."""
        self.unanswered_questions.append({
            'question': question,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'similar_docs': [r['document']['id'] for r in results] if results else []
        })
        print(f"[Gap] Logged unanswered question: {reason}")
    
    def get_knowledge_gaps(self):
        """Analyze unanswered questions to find knowledge gaps."""
        if not self.unanswered_questions:
            print("\n[Gaps] No knowledge gaps found - all questions answered!")
            return []
        
        print(f"\n{'='*80}")
        print(f"[Gaps] Knowledge Gap Analysis")
        print(f"{'='*80}")
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
            for q in questions[:3]:
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
        """Generate documentation template for unanswered question."""
        prompt = f"""A user asked: "{question}"

Our knowledge base couldn't answer this. Generate a documentation template.

Format:
1. What information is needed
2. Suggested structure
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
                        'doc1_text': self.documents[i]['text'][:80],
                        'doc2_text': self.documents[j]['text'][:80]
                    })
        
        if duplicates:
            print(f"\n[Duplicates] Found {len(duplicates)} potential duplicates:")
            for dup in duplicates:
                print(f"  Doc {dup['doc1_id']} <-> Doc {dup['doc2_id']} (similarity: {dup['similarity']:.3f})")
        else:
            print("\n[Duplicates] No duplicates found")
        
        return duplicates
    
    def save(self, filepath):
        """Save knowledge base with all metadata."""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'dimensions': self.dimensions,
            'unanswered_questions': self.unanswered_questions,
            'query_log': self.query_log,
            'use_reranking': self.use_reranking,
            'saved_at': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n[Save] Knowledge base saved to {filepath}")
    
    def load(self, filepath):
        """Load knowledge base with all metadata."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self.dimensions = data['dimensions']
        self.unanswered_questions = data.get('unanswered_questions', [])
        self.query_log = data.get('query_log', [])
        self.use_reranking = data.get('use_reranking', True)
        
        if self.use_reranking and not self.reranker:
            self.reranker = CrossEncoderReranker(self.client)
        
        print(f"[Load] Loaded {len(self.documents)} documents from {filepath}")


def demo_comprehensive():
    """Comprehensive demo showing all features."""
    api_key = getpass.getpass("Enter your OpenAI API key: ")
    client = OpenAI(api_key=api_key)
    
    print("\n" + "="*80)
    print("PRODUCTION KNOWLEDGE BASE DEMO")
    print("Features: Embeddings + Reranking + Adaptive Learning")
    print("="*80)
    
    # Initialize with reranking
    kb = ProductionKnowledgeBase(client, dimensions=512, use_reranking=True)
    
    print("\n=== Building Knowledge Base ===\n")
    
    # Add documents from different domains
    kb.add_document(
        "The Payment Service is a microservice that processes all payment transactions. "
        "It uses Stripe API for credit card processing and supports USD, EUR, and GBP currencies. "
        "The service is written in Python using FastAPI framework and runs on port 8090.",
        metadata={"service": "payment", "domain": "payments"}
    )
    
    kb.add_document(
        "Payment Service deployment requires these environment variables: STRIPE_API_KEY, DATABASE_URL, REDIS_URL. "
        "Use the deploy.sh script in /scripts directory. The service uses PostgreSQL 14 for data storage "
        "and Redis for caching. Daily backups run at 2 AM UTC.",
        metadata={"service": "payment", "type": "deployment", "domain": "payments"}
    )
    
    kb.add_document(
        "The User Service manages user authentication and authorization. "
        "It uses JWT tokens with 24-hour expiration. The service integrates with Auth0 for SSO. "
        "User data is stored in PostgreSQL. Contact: auth-team@company.com",
        metadata={"service": "user", "domain": "authentication"}
    )
    
    kb.add_document(
        "The Analytics Service processes user behavior data and generates reports. "
        "It uses Apache Airflow for ETL jobs and stores data in Snowflake. "
        "Real-time analytics are powered by Apache Kafka. Team: analytics@company.com",
        metadata={"service": "analytics", "domain": "data"}
    )
    
    kb.add_document(
        "For general payment inquiries, contact the Finance team at finance@company.com. "
        "For technical issues with payment processing, contact payment-ops@company.com. "
        "SLA: 4-hour response time for critical issues.",
        metadata={"type": "contact", "domain": "support"}
    )
    
    print("\n=== Demo: Questions with Reranking ===\n")
    
    # Question 1: Should work well
    result = kb.ask("What database does the Payment Service use?", top_k=3)
    print(f"\nAnswer: {result['answer']}\n")
    
    # Question 2: Cross-domain confusion test
    result = kb.ask("What team should I contact about payments?", top_k=3)
    print(f"\nAnswer: {result['answer']}\n")
    
    # Question 3: Missing information
    result = kb.ask("What is the Payment Service API rate limit?", top_k=3)
    print(f"\nAnswer: {result['answer']}\n")
    
    # Question 4: Ambiguous query (reranker should help)
    result = kb.ask("How do I deploy?", top_k=3)
    print(f"\nAnswer: {result['answer']}\n")
    
    # Show knowledge gaps
    kb.get_knowledge_gaps()
    
    # Show how to fix gaps
    print("\n=== Adaptive Learning: Filling Gaps ===\n")
    if kb.unanswered_questions:
        gap_question = kb.unanswered_questions[0]['question']
        print(f"Gap question: {gap_question}\n")
        suggestion = kb.suggest_new_content(gap_question)
        print("Suggested content template:")
        print(suggestion)
    
    # Save
    kb.save('production_kb.json')
    
    print("\n" + "="*80)
    print("Demo complete! Key features demonstrated:")
    print("- Two-stage retrieval (embedding + reranking)")
    print("- Cross-encoder prevents cross-domain confusion")
    print("- Strong grounding with citations")
    print("- Gap detection and content suggestions")
    print("="*80)


def demo_comparison():
    """Compare performance with and without reranking."""
    api_key = getpass.getpass("Enter your OpenAI API key: ")
    client = OpenAI(api_key=api_key)
    
    print("\n" + "="*80)
    print("RERANKING COMPARISON DEMO")
    print("="*80)
    
    # Setup knowledge base
    print("\n=== Building Test Knowledge Base ===\n")
    
    docs = [
        ("Payment Service uses Stripe for credit cards.", {"domain": "payment"}),
        ("User Service handles authentication with JWT tokens.", {"domain": "auth"}),
        ("Payment processing requires STRIPE_API_KEY environment variable.", {"domain": "payment"}),
        ("Authentication service integrates with Auth0 for SSO.", {"domain": "auth"}),
        ("For payment issues, contact finance@company.com.", {"domain": "support"}),
    ]
    
    # Test with reranking OFF
    print("\n--- Test 1: WITHOUT Reranking ---")
    kb_no_rerank = ProductionKnowledgeBase(client, use_reranking=False)
    for text, meta in docs:
        kb_no_rerank.add_document(text, metadata=meta)
    
    print("\nQuery: 'How do I authenticate?'\n")
    result = kb_no_rerank.ask("How do I authenticate?", top_k=2)
    print(f"Answer: {result['answer']}\n")
    
    # Test with reranking ON
    print("\n--- Test 2: WITH Reranking ---")
    kb_rerank = ProductionKnowledgeBase(client, use_reranking=True)
    for text, meta in docs:
        kb_rerank.add_document(text, metadata=meta)
    
    print("\nQuery: 'How do I authenticate?'\n")
    result = kb_rerank.ask("How do I authenticate?", top_k=2)
    print(f"Answer: {result['answer']}\n")
    
    print("="*80)
    print("Notice: Reranking helps select more relevant sources")
    print("="*80)


def main():
    print("\nSelect demo mode:")
    print("1. Comprehensive demo (all features)")
    print("2. Comparison demo (with/without reranking)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        demo_comprehensive()
    elif choice == "2":
        demo_comparison()
    else:
        print("Invalid choice. Running comprehensive demo...")
        demo_comprehensive()


if __name__ == "__main__":
    main()
