"""
rag_pipeline.py
================
Reusable Retrieval-Augmented Generation (RAG) pipeline built on LangChain +
OpenAI + FAISS.

This module is the single source of truth for the RAG logic. Both the Jupyter
notebook (project.ipynb) and the Streamlit deployment app (app.py) import from
here, so the demo and the documented experiment run the *exact* same code.

Design goals
------------
* **Self-contained**  : loads `.md`, `.txt` and `.pdf` documents from a folder.
* **Reproducible**    : temperature 0 for the LLM, fixed chunking parameters.
* **Inspectable**     : the query method returns the answer *and* the source
                        chunks used, so callers can show citations.
* **Persistable**     : the FAISS index can be saved to / loaded from disk.

Providers
---------
The pipeline supports two LLM providers and three embedding backends, selected
from environment variables (see ``.env.example``):

* **Chat LLM** – Azure OpenAI is used automatically when ``AZURE_OPENAI_ENDPOINT``
  is set; otherwise standard OpenAI (``OPENAI_API_KEY``) is used.
* **Embeddings** – controlled by ``EMBEDDINGS_BACKEND``:
    - ``local``  : free offline HuggingFace model (no API key needed).
    - ``azure``  : Azure OpenAI embeddings deployment.
    - ``openai`` : standard OpenAI embeddings.

Environment
-----------
Azure chat:   AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
              AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION
OpenAI chat:  OPENAI_API_KEY
Embeddings:   EMBEDDINGS_BACKEND (local|azure|openai), and either
              LOCAL_EMBEDDING_MODEL or AZURE_OPENAI_EMBEDDING_DEPLOYMENT.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# LangChain imports (LangChain >= 0.2 / 0.3 module layout)
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # read OPENAI_API_KEY from a local .env if present

# Default file types we know how to load.
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


@dataclass
class RAGConfig:
    """Tunable configuration for the RAG pipeline.

    LLM and embedding providers are resolved from environment variables at call
    time (see the module docstring), so the same config object works whether you
    point it at Azure OpenAI, standard OpenAI, or local embeddings.
    """

    data_dir: str = "data"
    index_dir: str = "faiss_index"
    # Embeddings
    embedding_model: str = "text-embedding-3-small"  # used when backend == openai
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Chat (standard OpenAI). For Azure, the deployment comes from env.
    chat_model: str = "gpt-4o-mini"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    temperature: float = 0.0


# The system prompt is the heart of a faithful RAG system: it forces the model
# to answer strictly from retrieved context and to abstain otherwise.
SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the user's question using ONLY the context provided below.
If the answer is not contained in the context, reply exactly:
"I don't know based on the provided documents."
Be concise and, where helpful, quote the relevant phrasing from the context.

Context:
{context}
"""


def _format_docs(docs: List[Document]) -> str:
    """Join retrieved chunks into a single context string for the prompt."""
    return "\n\n---\n\n".join(d.page_content for d in docs)


@dataclass
class RAGPipeline:
    """End-to-end RAG pipeline.

    Typical usage::

        rag = RAGPipeline()
        rag.load_documents()      # read + chunk files from data_dir
        rag.build_index()         # embed chunks into a FAISS vector store
        result = rag.query("What is RAG?")
        print(result["answer"])
        for src in result["sources"]:
            print(src.metadata["source"])
    """

    config: RAGConfig = field(default_factory=RAGConfig)

    # Populated as the pipeline runs.
    documents: List[Document] = field(default_factory=list, init=False)
    chunks: List[Document] = field(default_factory=list, init=False)
    vector_store: Optional[FAISS] = field(default=None, init=False)

    # ------------------------------------------------------------------ #
    # 1. Loading
    # ------------------------------------------------------------------ #
    def load_documents(self, data_dir: Optional[str] = None) -> List[Document]:
        """Load every supported file from ``data_dir`` into LangChain Documents."""
        directory = Path(data_dir or self.config.data_dir)
        if not directory.exists():
            raise FileNotFoundError(f"Data directory not found: {directory}")

        docs: List[Document] = []
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
            else:
                loader = TextLoader(str(path), encoding="utf-8")
            loaded = loader.load()
            # Normalize the source metadata to just the filename for clean citations.
            for d in loaded:
                d.metadata["source"] = path.name
            docs.extend(loaded)

        if not docs:
            raise ValueError(
                f"No supported documents ({SUPPORTED_EXTENSIONS}) found in {directory}"
            )
        self.documents = docs
        return docs

    def add_uploaded_files(self, file_paths: List[str]) -> List[Document]:
        """Load an explicit list of file paths (used by the Streamlit uploader)."""
        docs: List[Document] = []
        for fp in file_paths:
            path = Path(fp)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
            else:
                loader = TextLoader(str(path), encoding="utf-8")
            loaded = loader.load()
            for d in loaded:
                d.metadata["source"] = path.name
            docs.extend(loaded)
        self.documents.extend(docs)
        return docs

    # ------------------------------------------------------------------ #
    # 2. Chunking
    # ------------------------------------------------------------------ #
    def split_documents(self) -> List[Document]:
        """Split loaded documents into overlapping chunks."""
        if not self.documents:
            raise RuntimeError("Call load_documents() before split_documents().")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunks = splitter.split_documents(self.documents)
        return self.chunks

    # ------------------------------------------------------------------ #
    # 3. Embedding + indexing
    # ------------------------------------------------------------------ #
    def _embeddings(self):
        """Return an embeddings object based on EMBEDDINGS_BACKEND.

        - ``local``  : HuggingFace sentence-transformers model (no API key).
        - ``azure``  : Azure OpenAI embeddings deployment.
        - ``openai`` : standard OpenAI embeddings.
        """
        backend = os.getenv("EMBEDDINGS_BACKEND", "local").lower()
        if backend == "local":
            # Imported lazily so the heavy torch dependency is only required
            # when local embeddings are actually used.
            from langchain_huggingface import HuggingFaceEmbeddings

            model_name = os.getenv(
                "LOCAL_EMBEDDING_MODEL", self.config.local_embedding_model
            )
            return HuggingFaceEmbeddings(model_name=model_name)
        if backend == "azure":
            return AzureOpenAIEmbeddings(
                azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            )
        return OpenAIEmbeddings(model=self.config.embedding_model)

    def _llm(self):
        """Return a chat model. Uses Azure OpenAI if configured, else OpenAI.

        Some newer Azure deployments (e.g. the gpt-5 chat family) only accept the
        default temperature and reject an explicit ``temperature=0``. We therefore
        omit the parameter on Azure unless ``AZURE_SEND_TEMPERATURE=1`` is set,
        letting the model use its default. Standard OpenAI still honours
        ``config.temperature`` (0 by default) for deterministic, factual answers.
        """
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            kwargs = dict(
                azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            )
            if os.getenv("AZURE_SEND_TEMPERATURE") == "1":
                kwargs["temperature"] = self.config.temperature
            return AzureChatOpenAI(**kwargs)
        return ChatOpenAI(
            model=self.config.chat_model, temperature=self.config.temperature
        )

    def build_index(self) -> FAISS:
        """Embed chunks and build an in-memory FAISS vector store."""
        if not self.chunks:
            self.split_documents()
        self.vector_store = FAISS.from_documents(self.chunks, self._embeddings())
        return self.vector_store

    def save_index(self, index_dir: Optional[str] = None) -> str:
        """Persist the FAISS index to disk."""
        if self.vector_store is None:
            raise RuntimeError("No index to save. Call build_index() first.")
        target = index_dir or self.config.index_dir
        self.vector_store.save_local(target)
        return target

    def load_index(self, index_dir: Optional[str] = None) -> FAISS:
        """Load a previously saved FAISS index from disk."""
        target = index_dir or self.config.index_dir
        # allow_dangerous_deserialization is required to load a local pickle we created.
        self.vector_store = FAISS.load_local(
            target, self._embeddings(), allow_dangerous_deserialization=True
        )
        return self.vector_store

    # ------------------------------------------------------------------ #
    # 4. Retrieval + generation
    # ------------------------------------------------------------------ #
    def _build_chain(self):
        """Construct the LCEL RAG chain (retrieve -> prompt -> LLM -> parse)."""
        if self.vector_store is None:
            raise RuntimeError("Build or load an index before querying.")

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.top_k}
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{question}")]
        )
        llm = self._llm()
        chain = (
            {
                "context": retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain, retriever

    def query(self, question: str) -> dict:
        """Answer a question and return the answer plus the source chunks.

        Returns
        -------
        dict with keys:
            ``answer``  : str          – the grounded answer.
            ``sources`` : List[Document] – the retrieved chunks used as context.
        """
        chain, retriever = self._build_chain()
        # Retrieve once for citation display; the chain retrieves again internally,
        # but FAISS retrieval is fast and deterministic so results match.
        source_docs = retriever.invoke(question)
        answer = chain.invoke(question)
        return {"answer": answer, "sources": source_docs}

    def retrieve_only(self, question: str) -> List[Document]:
        """Return retrieved chunks without calling the LLM (useful for eval)."""
        if self.vector_store is None:
            raise RuntimeError("Build or load an index before retrieving.")
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.top_k}
        )
        return retriever.invoke(question)


def build_default_pipeline(data_dir: str = "data") -> RAGPipeline:
    """Convenience constructor: load + chunk + index in one call."""
    rag = RAGPipeline(RAGConfig(data_dir=data_dir))
    rag.load_documents()
    rag.split_documents()
    rag.build_index()
    return rag


if __name__ == "__main__":
    # Smoke test from the command line: python src/rag_pipeline.py "your question"
    import sys

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_ENDPOINT")):
        raise SystemExit(
            "Set OPENAI_API_KEY or Azure OpenAI env vars before running this test."
        )

    question = sys.argv[1] if len(sys.argv) > 1 else "What is RAG and why use it?"
    pipeline = build_default_pipeline()
    out = pipeline.query(question)
    print("\nQ:", question)
    print("\nA:", out["answer"])
    print("\nSources:", sorted({d.metadata["source"] for d in out["sources"]}))
