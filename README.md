# 🔎 RAG-Powered Search Engine — Q&A with Retrieval-Augmented Generation

**Final Capstone Project · AI Application Development**

A complete Retrieval-Augmented Generation (RAG) system that answers
natural-language questions about a custom document collection. Answers are
**grounded** in retrieved passages, **cite their sources**, and the system
**abstains** ("I don't know…") when the documents don't contain the answer —
directly addressing the two big LLM weaknesses: private/stale knowledge and
hallucination.

> **Project option #2** from the capstone brief. Built with
> **LangChain · Azure OpenAI (`gpt-5-chat`) · local MiniLM embeddings · FAISS ·
> Streamlit**. The LLM and embedding providers are swappable via environment
> variables (Azure OpenAI, standard OpenAI, or fully-local embeddings).

---

## ✨ Features

- Loads `.md`, `.txt`, and `.pdf` documents from the `data/` folder.
- Chunk → embed → index into a **FAISS** vector store.
- LangChain **LCEL** chain: retrieve → prompt → LLM → grounded answer.
- **Source citations** for every answer.
- **Anti-hallucination** system prompt (answers only from retrieved context).
- Two-axis **evaluation**: retrieval `hit@k` + generation faithfulness/abstention.
- Interactive **Streamlit** UI with document upload and tunable retrieval.

---

## 📁 Project Structure

```
FinalCapstoneIITG/
├── project.ipynb         # End-to-end notebook (problem → code → eval → results)
├── app.py                # Streamlit deployment app
├── src/
│   └── rag_pipeline.py   # Shared RAG pipeline (used by notebook AND app)
├── data/                 # Sample knowledge base (RAG / LLM / vector DB docs)
│   ├── rag_overview.md
│   ├── llm_concepts.md
│   └── vector_databases.md
├── report.pptx           # Editable slide report (add your screenshots here)
├── report.pdf            # Same report as PDF (overview, architecture, results…)
├── requirements.txt      # Pinned dependencies
├── .env.example          # Template for your OpenAI API key
├── build_notebook.py     # Build-time helper that regenerates project.ipynb
└── README.md             # This file
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure your provider (`.env`)
Copy `.env.example` to `.env` and fill in your credentials.

**Azure OpenAI (default in this project):**
```
AZURE_OPENAI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=gpt-5-chat
AZURE_OPENAI_API_VERSION=2025-01-01-preview
EMBEDDINGS_BACKEND=local          # offline MiniLM embeddings, no embeddings deployment needed
```

**Embeddings options** (`EMBEDDINGS_BACKEND`):
- `local` — free offline HuggingFace model `all-MiniLM-L6-v2` (default; no API key).
- `azure` — set `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` to your Azure embeddings deployment.
- `openai` — set `OPENAI_API_KEY` and uses `text-embedding-3-small`.

**Standard OpenAI instead of Azure:** simply omit the `AZURE_*` vars and set
`OPENAI_API_KEY`; the pipeline auto-detects the provider.

### 3a. Run the notebook
```bash
jupyter notebook project.ipynb
```
Run the cells top to bottom to reproduce the full pipeline and evaluation.

### 3b. Run the web app
```bash
streamlit run app.py
```
Then open the URL shown (usually http://localhost:8501). With Azure configured in
`.env`, the sidebar shows the active provider — just click **Build / Rebuild
knowledge base** and start asking questions.

### Command-line smoke test
```bash
python src/rag_pipeline.py "What is RAG and why is it useful?"
```

---

## 🧠 How It Works

**Indexing (offline)**
```
documents → load → split into overlapping chunks → embed → FAISS vector store
```

**Querying (online)**
```
question → embed → retrieve top-k chunks → build prompt (context + question)
        → LLM generates grounded answer → answer + cited sources
```

Key defaults (see `RAGConfig` in `src/rag_pipeline.py`): `chunk_size=1000`,
`chunk_overlap=150`, `top_k=4`, `temperature=0.0`.

---

## 📊 Evaluation

The notebook measures:
- **Retrieval — `hit@k`**: does the expected source appear in the top-k results?
- **Generation — faithfulness & abstention**: correct in-corpus answers and
  correct refusal on out-of-corpus questions.
- **Latency**: end-to-end wall-clock per query.

---

## ⚠️ Notes & Limitations

- Chat runs on Azure OpenAI (`gpt-5-chat`); embeddings run **locally** by default
  (`all-MiniLM-L6-v2`), so only the chat completion incurs API cost. The first run
  downloads the embedding model (~90 MB) once.
- FAISS is in-memory and ideal for prototypes; for production scale, swap in a
  managed vector DB (Pinecone, pgvector, Weaviate).
- **Security:** never commit your real `.env`. The provided key should be rotated
  if it was shared anywhere public.

## 🔮 Future Improvements

- Cross-encoder **re-ranking** after vector retrieval.
- **Hybrid search** (BM25 + dense vectors).
- **Conversational memory** for follow-up questions and response streaming.
- Automated faithfulness scoring with **RAGAS**.
