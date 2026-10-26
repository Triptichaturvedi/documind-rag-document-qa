
import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from src.rag_engine import (
    load_uploaded_file,
    split_documents,
    build_vector_store,
    open_vector_store,
    answer_question,
    DEFAULT_OPENAI_MODEL,
    OPENAI_MODELS,
    ModelProviderError,
)
# --------------------------------------------------
# 1. APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind | RAG Q&A",
    page_icon="📚",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
# Keep the legacy index intact; current Chroma versions use a separate runtime
# directory to avoid opening incompatible native index files.
# Keep legacy and previously generated indexes intact. Chroma versions can
# change their on-disk schema, so current sessions use a fresh database path.
DB_DIR = str((PROJECT_DIR / "chroma_db_runtime_current").resolve())
DATASET_PATH = PROJECT_DIR / "data" / "processed" / "dataset_cleaned.csv"

# Each indexing run gets its own ChromaDB collection.
COLLECTION_PREFIX = "course_docs_"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def configured_openai_api_key() -> str:
    """Read a deployment secret without requiring a local secrets file."""
    try:
        return str(st.secrets.get("OPENAI_API_KEY", ""))
    except FileNotFoundError:
        return os.getenv("OPENAI_API_KEY", "")


# --------------------------------------------------
# 2. SESSION STATE
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "active_collection" not in st.session_state:
    st.session_state.active_collection = None

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False


# --------------------------------------------------
# 3. STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #14213d, #2457a7);
        color: white;
        margin-bottom: 1rem;
    }

    .hero h1 {
        color: white;
        margin-bottom: .3rem;
    }

    .hero p {
        color: #e6edff;
        margin: 0;
    }

    [data-testid="stMetric"] {
        background: #f5f7fb;
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>📚 DocuMind — RAG Document Q&A</h1>
        <p>
            Ask questions about your documents and inspect the evidence
            behind every answer.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 4. SIDEBAR CONFIGURATION
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    configured_key = configured_openai_api_key()
    provider = st.radio(
        "Language model provider",
        ["OpenAI", "Ollama"],
        index=0,
        help=(
            "OpenAI uses a hosted API. Ollama runs a model locally "
            "without an API key if installed and downloaded."
        ),
    )

    if provider == "OpenAI":
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
            value=configured_key,
            help="Never commit this key to Git.",
        )

        model_name = st.selectbox(
            "Model",
            list(OPENAI_MODELS),
            index=list(OPENAI_MODELS).index(DEFAULT_OPENAI_MODEL),
        )

        ollama_url = "http://localhost:11434"

    else:
        api_key = ""

        model_name = st.text_input(
            "Ollama model name",
            value=DEFAULT_OLLAMA_MODEL,
        )

        ollama_url = st.text_input(
            "Ollama URL",
            value=DEFAULT_OLLAMA_URL,
        )

    if provider == "OpenAI" and not configured_key:
        st.warning(
            "OpenAI is the cloud-compatible default, but OPENAI_API_KEY is missing. "
            "Add it to Streamlit Secrets or choose Ollama for local use."
        )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=8,
        value=4,
    )

    st.caption(
        "Embedding model: sentence-transformers/all-MiniLM-L6-v2 "
        "(CPU-compatible; first run downloads model files)."
    )

    if st.session_state.index_ready:
        st.success("Document index is ready for this session.")
    else:
        st.info("Upload documents and build an index to get started.")


# --------------------------------------------------
# 5. DOCUMENT UPLOAD AND INDEXING
# --------------------------------------------------

left, right = st.columns([1.1, 1])

with left:
    st.subheader("1. Upload source documents")

    uploads = st.file_uploader(
        "Upload PDF, DOCX, TXT, Markdown or HTML files",
        type=["pdf", "docx", "txt", "md", "html", "htm"],
        accept_multiple_files=True,
    )

    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=300,
        max_value=1600,
        value=900,
        step=100,
    )

    overlap = st.slider(
        "Chunk overlap (characters)",
        min_value=0,
        max_value=300,
        value=150,
        step=25,
    )

    build_clicked = st.button(
        "Build / replace document index",
        type="primary",
        use_container_width=True,
    )

    if build_clicked:
        if not uploads:
            st.warning("Upload at least one document first.")

        else:
            try:
                documents = []

                with st.spinner("Reading and splitting documents..."):
                    for uploaded in uploads:
                        loaded_docs = load_uploaded_file(
                            uploaded.name,
                            uploaded.getvalue(),
                        )
                        documents.extend(loaded_docs)

                    chunks = split_documents(
                        documents,
                        chunk_size=chunk_size,
                        chunk_overlap=min(overlap, chunk_size // 2),
                    )

                if not chunks:
                    st.error(
                        "No text chunks were created. "
                        "Check whether the uploaded documents contain text."
                    )

                else:
                    # Create a unique collection for this indexing run.
                    # Do not delete the database directory while ChromaDB
                    # may have an open connection to it.
                    new_collection = (
                        COLLECTION_PREFIX + uuid.uuid4().hex[:12]
                    )

                    Path(DB_DIR).mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    with st.spinner(
                        f"Embedding {len(chunks)} chunks and "
                        "building the ChromaDB index..."
                    ):
                        build_vector_store(
                            chunks,
                            persist_directory=DB_DIR,
                            collection_name=new_collection,
                        )

                    # Only mark the new index active after successful
                    # completion of the build operation.
                    st.session_state.active_collection = new_collection
                    st.session_state.index_ready = True
                    st.session_state.chat_history = []

                    st.success(
                        f"Successfully indexed {len(documents)} "
                        f"document pages/sections into {len(chunks)} chunks."
                    )

                    st.caption(
                        f"Active collection: {new_collection}"
                    )

            except Exception as error:
                st.error(f"Indexing failed: {error}")
                st.exception(error)


# --------------------------------------------------
# 6. QUESTION ANSWERING
# --------------------------------------------------

with right:
    st.subheader("2. Ask a question")

    st.info(
        "Answers are generated from retrieved document chunks. "
        "Inspect the source evidence to verify each response."
    )

    question = st.text_area(
        "Your question",
        placeholder="What does the document say about...?",
        height=110,
    )

    ask_clicked = st.button(
        "Get answer",
        use_container_width=True,
    )

    if ask_clicked:
        if not question.strip():
            st.warning("Enter a question first.")

        elif (
            not st.session_state.index_ready
            or not st.session_state.active_collection
        ):
            st.warning(
                "Build the document index before asking questions."
            )

        else:
            try:
                with st.spinner(
                    "Retrieving evidence and generating answer..."
                ):
                    store = open_vector_store(
                        persist_directory=DB_DIR,
                        collection_name=st.session_state.active_collection,
                    )

                    answer, sources = answer_question(
                        question=question,
                        vector_store=store,
                        provider=provider,
                        model_name=model_name,
                        top_k=top_k,
                        openai_api_key=api_key,
                        ollama_base_url=ollama_url,
                    )

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "sources": sources,
                    }
                )

            except Exception as error:
                st.error(f"Could not answer the question: {error}")
                if not isinstance(error, ModelProviderError):
                    st.exception(error)


# --------------------------------------------------
# 7. ANSWERS AND SOURCE EVIDENCE
# --------------------------------------------------

if st.session_state.chat_history:
    st.divider()
    st.subheader("3. Answers and evidence")

    for item in reversed(st.session_state.chat_history):
        with st.container(border=True):
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(item["answer"])

            sources = item["sources"]

            with st.expander(
                f"Evidence used ({len(sources)} chunks)",
                expanded=True,
            ):
                if not sources:
                    st.write(
                        "No sufficiently relevant source chunks were found."
                    )

                for document, score in sources:
                    source_name = document.metadata.get(
                        "source", "Unknown"
                    )
                    page = document.metadata.get(
                        "page_or_section", "N/A"
                    )
                    chunk_id = document.metadata.get(
                        "chunk_id", "N/A"
                    )

                    st.markdown(
                        f"**Source:** `{source_name}`  \n"
                        f"**Page/section:** {page}  \n"
                        f"**Chunk:** `{chunk_id}`  \n"
                        f"**Relevance score:** {score:.3f}"
                    )

                    st.write(document.page_content)
                    st.divider()


# --------------------------------------------------
# 8. PROVIDED DATASET ANALYTICS
# --------------------------------------------------

st.divider()
st.subheader("Dataset analytics (provided course CSV)")

try:
    analytics = pd.read_csv(DATASET_PATH)

    required_columns = [
        "retrieval_score",
        "answer_quality_score",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in analytics.columns
    ]

    if missing_columns:
        st.warning(
            "The cleaned dataset is missing these columns: "
            + ", ".join(missing_columns)
        )

    else:
        col_a, col_b, col_c = st.columns(3)

        col_a.metric(
            "Rows",
            f"{len(analytics):,}",
        )

        col_b.metric(
            "Mean retrieval score",
            f"{analytics['retrieval_score'].mean():.3f}",
        )

        col_c.metric(
            "Mean answer quality",
            f"{analytics['answer_quality_score'].mean():.3f}",
        )

        st.caption(
            "These are descriptive statistics from the supplied evaluation "
            "dataset, not live performance measurements of uploaded documents."
        )

        with st.expander("View cleaned evaluation dataset"):
            st.dataframe(
                analytics,
                use_container_width=True,
            )

except FileNotFoundError:
    st.caption(
        "The cleaned evaluation CSV was not found. "
        "Run the EDA script to create the processed dataset."
    )

except Exception as error:
    st.warning(f"Could not load dataset analytics: {error}")
