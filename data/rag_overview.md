# Retrieval-Augmented Generation (RAG): An Overview

## What is RAG?
Retrieval-Augmented Generation (RAG) is an architecture that combines a
retrieval system with a large language model (LLM). Instead of relying only on
the parametric knowledge stored in an LLM's weights, RAG retrieves relevant
documents from an external knowledge base at query time and supplies them to the
model as context. This grounds the model's answer in source material and reduces
hallucination.

## Why use RAG?
1. **Up-to-date knowledge** – The knowledge base can be updated without
   retraining the model. New documents are simply re-indexed.
2. **Source attribution** – Because answers are grounded in retrieved chunks,
   the system can cite the exact passages it used.
3. **Cost efficiency** – Fine-tuning a large model is expensive. RAG lets a
   general-purpose model specialize on private data cheaply.
4. **Reduced hallucination** – When the model is told to answer only from the
   provided context, it is far less likely to fabricate facts.

## The RAG pipeline
A typical RAG pipeline has two phases: indexing (offline) and querying (online).

### Indexing phase
1. **Load** documents from sources such as PDFs, web pages, or databases.
2. **Split** the documents into smaller overlapping chunks so that each chunk
   fits comfortably inside the embedding model's context window.
3. **Embed** each chunk into a dense vector using an embedding model.
4. **Store** the vectors in a vector database (for example FAISS or Chroma)
   together with the original text and metadata.

### Querying phase
1. **Embed the user question** using the same embedding model.
2. **Retrieve** the top-k most similar chunks using vector similarity search
   (commonly cosine similarity).
3. **Augment** the prompt by inserting the retrieved chunks as context.
4. **Generate** an answer with the LLM, instructing it to use only the context.

## Key components
- **Embedding model**: Converts text into vectors. OpenAI's
  `text-embedding-3-small` produces 1536-dimensional embeddings and is a strong,
  inexpensive default.
- **Vector store**: FAISS (Facebook AI Similarity Search) is an in-memory
  library for fast nearest-neighbour search and is ideal for prototypes.
- **Chunking strategy**: A chunk size of roughly 800–1200 characters with a
  100–200 character overlap is a common starting point. Overlap preserves
  context that would otherwise be cut at chunk boundaries.
- **Top-k**: The number of chunks retrieved per query. Typical values are 3–6.
  Too few risks missing context; too many adds noise and cost.
