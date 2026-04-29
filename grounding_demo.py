#!/usr/bin/env python3
"""
Demonstrates different grounding strategies to keep LLM answers aligned with context.
Shows the difference between weak and strong grounding techniques.
"""
import getpass
from openai import OpenAI
import json


def weak_grounding(client, context, question):
    """Weak grounding - LLM can easily drift from context."""
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content


def strong_grounding(client, context, question):
    """Strong grounding - explicit instructions and lower temperature."""
    system_prompt = """You are a knowledge base assistant. Follow these rules strictly:
1. ONLY use information from the provided context
2. If the context doesn't contain the answer, respond: "I don't have that information in my knowledge base."
3. Quote relevant parts from context when possible
4. Do not use any external knowledge or make assumptions
5. If context is partial, say what you know and what's missing"""

    user_prompt = f"""Context:
{context}

Question: {question}

Provide your answer based ONLY on the context above."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1  # Lower temperature = more deterministic
    )
    return response.choices[0].message.content


def citation_grounding(client, context_items, question):
    """Strongest grounding - requires citations for each claim."""
    context_text = "\n\n".join([
        f"[Source {i+1}]: {item}"
        for i, item in enumerate(context_items)
    ])
    
    system_prompt = """You are a knowledge base assistant that provides cited answers.

Rules:
1. Answer ONLY using the provided sources
2. Cite every claim with [Source N]
3. If information is not in sources, say "Not found in knowledge base"
4. Do not combine information unless explicitly connected in sources
5. Be precise - don't generalize beyond what sources say"""

    user_prompt = f"""{context_text}

Question: {question}

Answer with inline citations:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content


def structured_grounding(client, context, question):
    """Structured output to force validation of answer against context."""
    system_prompt = """You must respond in JSON format with these fields:
- "answer": Your answer to the question
- "confidence": "high" if context fully answers, "medium" if partial, "low" if guessing
- "context_used": List of specific quotes from context you used
- "missing_info": What information is missing to fully answer

Only use information from the provided context."""

    user_prompt = f"""Context:
{context}

Question: {question}

Respond in JSON format:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def main():
    api_key = getpass.getpass("Enter your OpenAI API key: ")
    client = OpenAI(api_key=api_key)
    
    # Test contexts
    context = """The Payment Service uses Stripe API for processing credit cards. 
It supports USD, EUR, and GBP currencies. The service is deployed on port 8080."""
    
    context_items = [
        "The Payment Service uses Stripe API for processing credit cards.",
        "The service supports USD, EUR, and GBP currencies.",
        "The service is deployed on port 8080 in production."
    ]
    
    # Test with question that's partially in context
    question_in_context = "What currencies does the Payment Service support?"
    
    # Test with question outside context
    question_outside_context = "What database does the Payment Service use?"
    
    print("=" * 80)
    print("TEST 1: Question with answer IN context")
    print("=" * 80)
    print(f"\nQuestion: {question_in_context}\n")
    
    print("--- Weak Grounding (basic prompt) ---")
    print(weak_grounding(client, context, question_in_context))
    
    print("\n--- Strong Grounding (strict instructions) ---")
    print(strong_grounding(client, context, question_in_context))
    
    print("\n--- Citation Grounding (requires sources) ---")
    print(citation_grounding(client, context_items, question_in_context))
    
    print("\n--- Structured Grounding (JSON validation) ---")
    result = structured_grounding(client, context, question_in_context)
    print(json.dumps(result, indent=2))
    
    print("\n" + "=" * 80)
    print("TEST 2: Question with answer OUTSIDE context (should refuse)")
    print("=" * 80)
    print(f"\nQuestion: {question_outside_context}\n")
    
    print("--- Weak Grounding (basic prompt) ---")
    print(weak_grounding(client, context, question_outside_context))
    print("\n⚠️  Notice: May hallucinate or guess")
    
    print("\n--- Strong Grounding (strict instructions) ---")
    print(strong_grounding(client, context, question_outside_context))
    print("\n✓ Should refuse to answer")
    
    print("\n--- Citation Grounding (requires sources) ---")
    print(citation_grounding(client, context_items, question_outside_context))
    print("\n✓ Cannot cite non-existent sources")
    
    print("\n--- Structured Grounding (JSON validation) ---")
    result = structured_grounding(client, context, question_outside_context)
    print(json.dumps(result, indent=2))
    print("\n✓ Confidence field reveals uncertainty")


if __name__ == "__main__":
    main()
