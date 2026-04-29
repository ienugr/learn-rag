#!/usr/bin/env python3
"""
OpenAI text embedding demo using text-embedding-3-small with reduced dimensions.
"""
import getpass
from openai import OpenAI


def get_embeddings(text, api_key, dimensions=512):
    """
    Generate embeddings for the given text with reduced dimensions.
    
    Args:
        text: Text to embed (string or list of strings)
        api_key: OpenAI API key
        dimensions: Target dimension size (default 512, original is 1536)
    
    Returns:
        List of embeddings
    """
    client = OpenAI(api_key=api_key)
    
    if isinstance(text, str):
        text = [text]
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        dimensions=dimensions
    )
    
    return [item.embedding for item in response.data]


def main():
    api_key = getpass.getpass("Enter your OpenAI API key: ")
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming artificial intelligence",
        "Python is a versatile programming language"
    ]
    
    print(f"\nGenerating embeddings for {len(sample_texts)} texts...")
    print(f"Using model: text-embedding-3-small with dimensions=512\n")
    
    embeddings = get_embeddings(sample_texts, api_key, dimensions=512)
    
    for i, (text, embedding) in enumerate(zip(sample_texts, embeddings)):
        print(f"Text {i+1}: {text[:50]}...")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        print()


if __name__ == "__main__":
    main()

