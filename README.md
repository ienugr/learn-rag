# OpenAI Embedding Demo

Learn embeddings through practical examples: basic embeddings, RAG knowledge base, grounding techniques, and adaptive knowledge management.

## Setup

```bash
# Set Python version (requires Python 3.7+)
pyenv local 3.12

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

In this demo, we use OpenAI's `text-embedding-3-small` model for generating embeddings. Therefore, you need to provide an OpenAI API key to run the script.

## Examples

### 1. Basic Embeddings (`embedding_demo.py`)

Simple demonstration of generating text embeddings.

```bash
python embedding_demo.py
```

### 2. Knowledge Base / RAG (`knowledge_base_demo.py`)

Build a semantic search knowledge base for team/service documentation.

```bash
python knowledge_base_demo.py
```

### 3. Grounding Techniques (`grounding_demo.py`)

Compare weak vs strong grounding strategies to prevent hallucinations.

```bash
python grounding_demo.py
```

Shows 4 approaches:
- **Weak grounding** - Basic prompt (prone to hallucination)
- **Strong grounding** - Strict rules, low temperature
- **Citation grounding** - Requires source references
- **Structured grounding** - JSON with confidence validation

### 4. Adaptive Knowledge Base (`adaptive_kb_demo.py`)

Production-ready KB with feedback loop and maintenance operations.

```bash
python adaptive_kb_demo.py
```

Features:
- **Gap detection** - Tracks unanswered questions
- **Content suggestions** - AI generates templates for missing docs
- **Update documents** - Fix incorrect information with version tracking
- **Delete documents** - Remove outdated content
- **Duplicate detection** - Find redundant documentation
- **Query logging** - Analyze usage patterns

## Knowledge Base Lifecycle

```
1. Initial Build
   └─> Add documents → Generate embeddings

2. Production Usage
   └─> Users ask questions
       ├─> Answered → Log success
       └─> Unanswered → Log gap

3. Gap Analysis
   └─> Review unanswered questions
       └─> Identify missing topics

4. Content Evolution
   ├─> Add new documents (fill gaps)
   ├─> Update documents (fix errors, re-embed)
   ├─> Delete documents (remove obsolete)
   └─> Merge duplicates

5. Repeat from step 2
```

## How It Works

**RAG Pipeline:**
1. Embed documents → vector store
2. Query → embed → similarity search
3. Retrieve top-k contexts
4. LLM generates grounded answer

**Feedback Loop:**
1. Track unanswered/low-confidence queries
2. Analyze patterns → identify gaps
3. Add/update documentation
4. Re-embed changed content
5. Monitor improvement

## Maintenance Operations

**Adding:** `kb.add_document(text, metadata)`
**Updating:** `kb.add_document(text, metadata, doc_id=5)` (re-embeds)
**Deleting:** `kb.delete_document(doc_id=5)` (soft delete)
**Finding duplicates:** `kb.find_duplicates(threshold=0.85)`
**Gap analysis:** `kb.get_knowledge_gaps()`

## Features

- Uses `text-embedding-3-small` (1536→512 dimensions)
- Cosine similarity for semantic matching
- Strong grounding with citations
- Version tracking for updates
- Persistent storage (JSON)
