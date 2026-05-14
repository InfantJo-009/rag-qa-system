"""
=============================================================================
  RAG Question Answering System
  ─────────────────────────────
  A beginner-friendly Context-Based Question Answering app powered by
  Retrieval-Augmented Generation (RAG).

  How it works (high level):
    1. You upload a PDF document.
    2. The app extracts plain text from the PDF.
    3. The text is split into small, overlapping "chunks".
    4. Each chunk is converted into a numerical vector (embedding).
    5. The embeddings are stored in a FAISS vector database.
    6. When you ask a question, the app finds the most relevant chunks.
    7. Those chunks are sent to an LLM (GPT) as context to generate an answer.

  Author : Your Name
  Date   : 2026-04-18
=============================================================================
"""

# ── Standard library imports ────────────────────────────────────────────────
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

# ── Third-party imports ─────────────────────────────────────────────────────
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIGURATION
#    Sets browser tab title, icon, and layout width.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Question Answering System",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CUSTOM CSS
#    Adds gradient backgrounds, card styling, and polished typography
#    so the app looks professional out of the box.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ─────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main container background ──────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
}

/* ── Header banner ──────────────────────────────────────────────────────── */
.header-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.35);
}
.header-banner h1 {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.header-banner p {
    color: rgba(255,255,255,0.85);
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 300;
}

/* ── Glass card ─────────────────────────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(102,126,234,0.18);
}
.glass-card h3 {
    color: #a5b4fc;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* ── Answer box ─────────────────────────────────────────────────────────── */
.answer-box {
    background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.15) 100%);
    border-left: 4px solid #667eea;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
    color: #e2e8f0;
    font-size: 1.05rem;
    line-height: 1.7;
}

/* ── Context box ────────────────────────────────────────────────────────── */
.context-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 1.2rem;
    margin-top: 0.5rem;
    color: #94a3b8;
    font-size: 0.92rem;
    line-height: 1.6;
    max-height: 300px;
    overflow-y: auto;
}

/* ── Status badges ──────────────────────────────────────────────────────── */
.status-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
}
.badge-success {
    background: rgba(16,185,129,0.15);
    color: #34d399;
    border: 1px solid rgba(16,185,129,0.3);
}
.badge-info {
    background: rgba(102,126,234,0.15);
    color: #a5b4fc;
    border: 1px solid rgba(102,126,234,0.3);
}
.badge-warning {
    background: rgba(245,158,11,0.15);
    color: #fbbf24;
    border: 1px solid rgba(245,158,11,0.3);
}

/* ── Streamlit overrides ────────────────────────────────────────────────── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
}
.stFileUploader {
    background: rgba(255,255,255,0.04);
    border: 2px dashed rgba(102,126,234,0.4);
    border-radius: 14px;
    padding: 1rem;
}
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(102,126,234,0.4) !important;
}
.stSpinner > div {
    border-color: #667eea !important;
}

/* ── Sidebar styling ────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.95);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a5b4fc;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>RAG Question Answering System</h1>
    <p>Upload a PDF, ask questions, get answers grounded in your document</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SIDEBAR – API KEY & SETTINGS
#    The OpenAI API key is entered via the sidebar so it stays out of
#    the main content area. It is stored in Streamlit's session state.
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    # ---------- API Key Loading ----------
    # Securely load the API key from Streamlit Secrets or Environment Variables.
    # This prevents the key from being shown in the UI.
    try:
        api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    except Exception:
        api_key = os.environ.get("GROQ_API_KEY", "")

    st.markdown("---")

    # ---------- Advanced Settings ----------
    st.markdown("### Advanced Settings")

    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=200,
        max_value=2000,
        value=1000,
        step=100,
        help="Size of each text chunk. Smaller = more precise, larger = more context.",
    )

    chunk_overlap = st.slider(
        "Chunk overlap (characters)",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
        help="Overlap between consecutive chunks to preserve context across boundaries.",
    )

    top_k = st.slider(
        "Top-K chunks to retrieve",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of most-relevant chunks to feed to the LLM.",
    )

    model_name = st.selectbox(
        "LLM Model",
        options=["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0,  # default to llama-3.1-8b-instant
        help="Which Groq model to use for generating answers.",
    )

    show_context = st.checkbox(
        "Show retrieved context",
        value=True,
        help="If checked, the raw chunks used to answer your question are shown.",
    )

    st.markdown("---")
    st.markdown(
        "<p style='color:#64748b;font-size:0.8rem;text-align:center;'>"
        "Built with LangChain · FAISS · Groq · HF</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf(uploaded_file) -> list:
    """
    Save the uploaded PDF to a temp file, then use LangChain's PyPDFLoader
    to extract text page-by-page.

    Returns:
        A list of LangChain Document objects (one per page).
    """
    # Streamlit's UploadedFile is in-memory; PyPDFLoader needs a file path,
    # so we write it to a temporary file first.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    # Clean up the temp file
    os.unlink(tmp_path)

    return pages


def split_documents(pages: list, chunk_size: int, chunk_overlap: int) -> list:
    """
    Split the extracted pages into smaller overlapping chunks.

    Why overlap? So that important information sitting at the boundary of
    two chunks isn't lost.

    Returns:
        A list of LangChain Document objects (chunks).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,           # count characters
        separators=["\n\n", "\n", ". ", " ", ""],  # prefer splitting at paragraphs
    )
    chunks = splitter.split_documents(pages)
    return chunks


def create_vector_store(chunks: list, api_key: str):
    """
    Convert each text chunk into an embedding vector using HuggingFace's
    local embedding model, then index them in a FAISS vector store for fast
    similarity search. (No API key needed for HF embeddings)

    Returns:
        A FAISS vector store object.
    """
    with st.spinner("Loading embedding model..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def build_qa_chain(vector_store, api_key: str, model_name: str, top_k: int):
    """
    Build a RetrievalQA chain that:
      1. Retrieves the top-K most relevant chunks from the FAISS store.
      2. Injects them as context into a prompt.
      3. Sends the prompt to the LLM to generate a grounded answer.

    The custom prompt instructs the model to ONLY use the provided context,
    preventing hallucination.

    Returns:
        A LangChain RetrievalQA chain.
    """
    # ── Custom prompt template ──────────────────────────────────────────
    prompt_template = """You are a helpful assistant that answers questions 
based ONLY on the provided context. If the answer cannot be found in the 
context, say "I couldn't find the answer in the uploaded document."

Context:
{context}

Question: {question}

Answer (be concise and accurate):"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    # ── Initialize the LLM ─────────────────────────────────────────────
    llm = ChatGroq(
        model_name=model_name,
        groq_api_key=api_key,
        temperature=0.2,           # low temperature = more factual
    )

    # ── Build the retrieval chain ──────────────────────────────────────
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",       # "stuff" = concatenate all chunks into one prompt
        retriever=retriever,
        return_source_documents=True,  # so we can display the retrieved chunks
        chain_type_kwargs={"prompt": PROMPT},
    )

    return qa_chain


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

# Create two columns: left for upload, right for Q&A
col_upload, col_qa = st.columns([1, 1.5], gap="large")

# ─── LEFT COLUMN: Document Upload ───────────────────────────────────────────
with col_upload:
    st.markdown('<div class="glass-card"><h3>Upload Document</h3>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload the PDF you want to ask questions about.",
    )

    # Process button — only enabled when a file is uploaded
    process_btn = st.button("Process Document", disabled=(uploaded_file is None))

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Processing pipeline ─────────────────────────────────────────────
    if process_btn and uploaded_file is not None:
        # --- Validate API key ---
        if not api_key:
            st.error("Groq API key not found. Please add it to your Streamlit Secrets.")
            st.stop()

        try:
            with st.spinner("Reading PDF..."):
                pages = load_pdf(uploaded_file)
                if not pages:
                    st.error("The PDF appears to be empty or unreadable.")
                    st.stop()

            with st.spinner("Splitting text into chunks..."):
                chunks = split_documents(pages, chunk_size, chunk_overlap)

            with st.spinner("Creating embeddings & vector store..."):
                vector_store = create_vector_store(chunks, api_key)

            # Store results in session state so they persist across reruns
            st.session_state["vector_store"] = vector_store
            st.session_state["chunks"] = chunks
            st.session_state["num_pages"] = len(pages)
            st.session_state["file_name"] = uploaded_file.name

            st.success("Document processed successfully!")

        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

    # ── Show document stats ─────────────────────────────────────────────
    if "chunks" in st.session_state:
        st.markdown('<div class="glass-card"><h3>Document Stats</h3>', unsafe_allow_html=True)
        stat_cols = st.columns(3)
        with stat_cols[0]:
            st.metric("Pages", st.session_state["num_pages"])
        with stat_cols[1]:
            st.metric("Chunks", len(st.session_state["chunks"]))
        with stat_cols[2]:
            st.metric("File", st.session_state["file_name"][:15])
        st.markdown(
            '<span class="status-badge badge-success">Ready for questions</span>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ─── RIGHT COLUMN: Question & Answer ────────────────────────────────────────
with col_qa:
    st.markdown('<div class="glass-card"><h3>Ask a Question</h3>', unsafe_allow_html=True)

    question = st.text_input(
        "Type your question below:",
        placeholder="e.g., What is the main topic of this document?",
    )

    ask_btn = st.button("Get Answer")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Generate answer ─────────────────────────────────────────────────
    if ask_btn:
        # --- Validations ---
        if not api_key:
            st.error("Groq API key not found. Please add it to your Streamlit Secrets.")
        elif "vector_store" not in st.session_state:
            st.warning("Please upload and process a PDF first.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Thinking..."):
                    # Build the QA chain
                    qa_chain = build_qa_chain(
                        st.session_state["vector_store"],
                        api_key,
                        model_name,
                        top_k,
                    )

                    # Run the chain
                    result = qa_chain.invoke({"query": question})

                # ── Display the answer ──────────────────────────────────
                st.markdown('<div class="glass-card"><h3>Answer</h3>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="answer-box">{result["result"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

                # ── Optionally show retrieved context chunks ────────────
                if show_context and "source_documents" in result:
                    st.markdown(
                        '<div class="glass-card"><h3>Retrieved Context</h3>',
                        unsafe_allow_html=True,
                    )
                    for i, doc in enumerate(result["source_documents"], 1):
                        page_num = doc.metadata.get("page", "?")
                        st.markdown(
                            f"**Chunk {i}** · Page {int(page_num) + 1}",
                        )
                        st.markdown(
                            f'<div class="context-box">{doc.page_content}</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. FOOTER – HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="glass-card" style="text-align:center;">
    <h3>How RAG Works — In 4 Steps</h3>
    <div style="display:flex; justify-content:center; gap:2rem; flex-wrap:wrap; margin-top:1rem;">
        <div style="flex:1; min-width:180px;">
            <p style="color:#a5b4fc; font-weight:600;">1. Extract</p>
            <p style="color:#94a3b8; font-size:0.9rem;">Text is extracted from your PDF document</p>
        </div>
        <div style="flex:1; min-width:180px;">
            <p style="color:#a5b4fc; font-weight:600;">2. Chunk</p>
            <p style="color:#94a3b8; font-size:0.9rem;">Text is split into small, meaningful pieces</p>
        </div>
        <div style="flex:1; min-width:180px;">
            <p style="color:#a5b4fc; font-weight:600;">3. Embed & Store</p>
            <p style="color:#94a3b8; font-size:0.9rem;">Chunks become vectors in a FAISS database</p>
        </div>
        <div style="flex:1; min-width:180px;">
            <p style="color:#a5b4fc; font-weight:600;">4. Retrieve & Answer</p>
            <p style="color:#94a3b8; font-size:0.9rem;">Relevant chunks are sent to the LLM for answering</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
