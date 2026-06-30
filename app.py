"""
app.py — Streamlit deployment for the RAG-Powered Search Engine
================================================================
Interactive web UI for the RAG pipeline defined in ``src/rag_pipeline.py``.

Run locally::

    streamlit run app.py

Features
--------
* Build a knowledge base from the bundled ``data/`` corpus, or upload your own
  ``.pdf`` / ``.txt`` / ``.md`` files.
* Ask natural-language questions and get answers grounded in your documents.
* See the exact source passages used to answer each question (citations).
* Tune retrieval (top-k) and model settings from the sidebar.

The OpenAI API key can be entered in the sidebar or provided via the
``OPENAI_API_KEY`` environment variable / ``.env`` file.
"""

import os
import sys
import tempfile

import streamlit as st
from dotenv import load_dotenv

# Make the shared pipeline importable whether run from repo root or elsewhere.
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from rag_pipeline import RAGConfig, RAGPipeline  # noqa: E402

load_dotenv()

st.set_page_config(page_title="RAG Search Engine", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _save_uploads_to_tempdir(uploaded_files) -> list[str]:
    """Write Streamlit UploadedFile objects to a temp dir, return their paths."""
    paths = []
    tmpdir = tempfile.mkdtemp(prefix="rag_uploads_")
    for uf in uploaded_files:
        dest = os.path.join(tmpdir, uf.name)
        with open(dest, "wb") as fh:
            fh.write(uf.getbuffer())
        paths.append(dest)
    return paths


def build_pipeline(config: RAGConfig, uploaded_files) -> RAGPipeline:
    """Construct and index a RAGPipeline from the corpus and/or uploads."""
    rag = RAGPipeline(config)

    # Load bundled corpus if the data dir exists.
    if os.path.isdir(config.data_dir) and any(
        f.lower().endswith((".md", ".txt", ".pdf"))
        for f in os.listdir(config.data_dir)
    ):
        rag.load_documents()

    # Add any user uploads.
    if uploaded_files:
        paths = _save_uploads_to_tempdir(uploaded_files)
        rag.add_uploaded_files(paths)

    if not rag.documents:
        raise ValueError(
            "No documents to index. Add files to the data/ folder or upload some."
        )

    rag.split_documents()
    rag.build_index()
    return rag


# --------------------------------------------------------------------------- #
# Sidebar — configuration
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Configuration")

# Detect which LLM provider is configured (Azure takes precedence).
USING_AZURE = bool(os.getenv("AZURE_OPENAI_ENDPOINT"))
EMB_BACKEND = os.getenv("EMBEDDINGS_BACKEND", "local").lower()

if USING_AZURE:
    st.sidebar.success(
        f"Provider: **Azure OpenAI**\n\n"
        f"Deployment: `{os.getenv('AZURE_OPENAI_DEPLOYMENT', '?')}`\n\n"
        f"Embeddings: `{EMB_BACKEND}`"
    )
    chat_model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-chat")
else:
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value="",
        help="Leave blank to use the OPENAI_API_KEY environment variable / .env file.",
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
    chat_model = st.sidebar.selectbox("Chat model", ["gpt-4o-mini", "gpt-4o"], index=0)
top_k = st.sidebar.slider("Chunks to retrieve (top-k)", 1, 10, 4)
chunk_size = st.sidebar.slider("Chunk size (chars)", 400, 2000, 1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk overlap (chars)", 0, 400, 150, step=50)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, step=0.1)

st.sidebar.markdown("---")
uploaded_files = st.sidebar.file_uploader(
    "Add your own documents",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    help="Optional. Uploaded files are indexed alongside the bundled corpus.",
)

build_clicked = st.sidebar.button("🔧 Build / Rebuild knowledge base", type="primary")


# --------------------------------------------------------------------------- #
# Main panel
# --------------------------------------------------------------------------- #
st.title("🔎 RAG-Powered Search Engine")
st.caption(
    "Ask questions about your documents. Answers are grounded in retrieved "
    "passages and cite their sources. Built with LangChain · OpenAI · FAISS."
)

if "rag" not in st.session_state:
    st.session_state.rag = None
if "history" not in st.session_state:
    st.session_state.history = []

# Build the index when requested.
if build_clicked:
    if not (USING_AZURE or os.getenv("OPENAI_API_KEY")):
        st.error("Please provide an OpenAI API key in the sidebar first.")
    else:
        config = RAGConfig(
            data_dir="data",
            chat_model=chat_model,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            temperature=temperature,
        )
        try:
            with st.spinner("Loading, chunking, embedding and indexing documents…"):
                st.session_state.rag = build_pipeline(config, uploaded_files)
            n_vectors = st.session_state.rag.vector_store.index.ntotal
            st.success(
                f"Knowledge base ready — {len(st.session_state.rag.chunks)} chunks "
                f"({n_vectors} vectors) indexed."
            )
        except Exception as exc:  # surface any setup/API errors to the user
            st.session_state.rag = None
            st.error(f"Failed to build knowledge base: {exc}")

# Status banner.
if st.session_state.rag is None:
    hint = (
        "👈 Click **Build / Rebuild knowledge base** to get started."
        if USING_AZURE
        else "👈 Enter your OpenAI API key and click **Build / Rebuild knowledge base**."
    )
    st.info(hint + " The bundled `data/` corpus is indexed by default.")

# Question box.
question = st.text_input(
    "Your question",
    placeholder="e.g. What is RAG and why does it reduce hallucination?",
    disabled=st.session_state.rag is None,
)
ask = st.button("Ask", disabled=st.session_state.rag is None)

if ask and question:
    try:
        with st.spinner("Retrieving and generating answer…"):
            result = st.session_state.rag.query(question)
        st.session_state.history.insert(0, (question, result))
    except Exception as exc:
        st.error(f"Query failed: {exc}")

# Render conversation history (most recent first).
for q, result in st.session_state.history:
    st.markdown(f"### ❓ {q}")
    st.markdown(result["answer"])
    sources = sorted({d.metadata.get("source", "?") for d in result["sources"]})
    st.caption("📚 Sources: " + ", ".join(sources))
    with st.expander("Show retrieved passages"):
        for i, d in enumerate(result["sources"], 1):
            st.markdown(f"**[{i}] {d.metadata.get('source', '?')}**")
            st.write(d.page_content)
    st.markdown("---")
