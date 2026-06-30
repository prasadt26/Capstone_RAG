# Vector Databases and Similarity Search

## What is a vector database?
A vector database stores high-dimensional embedding vectors and supports fast
**approximate nearest-neighbour (ANN)** search. Given a query vector it returns
the stored vectors that are most similar, typically ranked by cosine similarity
or Euclidean distance. Vector databases are the storage layer of a RAG system.

## FAISS
FAISS (Facebook AI Similarity Search) is an open-source library for efficient
similarity search. It runs in-memory, has no server to manage, and can be
persisted to disk by saving its index files. FAISS is an excellent choice for
prototypes and small-to-medium corpora because of its simplicity and speed.
In LangChain, a FAISS index can be built directly from a list of documents and
an embedding model, then saved locally and reloaded later.

## Chroma
Chroma is an open-source, developer-friendly vector database that persists data
to disk by default and offers a simple Python API. It is a common alternative to
FAISS when persistence and metadata filtering are needed without standing up a
heavy service.

## Managed vector databases
For production workloads at scale, managed services such as Pinecone, Weaviate,
Milvus, and pgvector (a PostgreSQL extension) provide durability, horizontal
scaling, metadata filtering, and access control. They trade the simplicity of an
in-memory library for operational robustness.

## Distance metrics
- **Cosine similarity** measures the angle between two vectors and ignores their
  magnitude. It is the most common metric for text embeddings.
- **Euclidean (L2) distance** measures straight-line distance and is sensitive
  to magnitude.
- **Dot product** is fast and works well when embeddings are normalized.

## Indexing strategies
A flat index compares the query against every stored vector and is exact but
slow for large datasets. ANN indexes such as HNSW (Hierarchical Navigable Small
World) graphs and IVF (Inverted File) indexes trade a small amount of accuracy
for large speed-ups, which is essential once a corpus grows beyond millions of
vectors.
