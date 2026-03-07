"""
RAG Chatbot with Chat History
Stack: LangChain + Groq (Llama3) + HuggingFace + Pinecone + Cohere Rerank

Run:
    streamlit run app.py

.env file needs:
    GROQ_API_KEY
    PINECONE_API_KEY
    COHERE_API_KEYY
"""

import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── LangChain imports ──────────────────────────────────────────────────────
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_cohere import CohereRerank
from langchain_pinecone import PineconeVectorStore
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# ── Pinecone ───────────────────────────────────────────────────────────────
from pinecone import Pinecone, ServerlessSpec

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

INDEX_NAME = "rag-chat-history"
DIMENSION  = 384          # matches all-MiniLM-L6-v2
METRIC     = "cosine"     # cosine for pure dense search


# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCES  (loaded once, reused on every Streamlit rerun)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name   = "all-MiniLM-L6-v2",
        model_kwargs = {"device": "cpu"},
        encode_kwargs= {"normalize_embeddings": True}
    )

@st.cache_resource
def load_llm(api_key: str):
    return ChatGroq(
        groq_api_key = api_key,
        model_name   = "llama-3.3-70b-versatile",
        temperature  = 0,
        max_tokens   = 1024,
    )

@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name      = INDEX_NAME,
            dimension = DIMENSION,
            metric    = METRIC,
            spec      = ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc


# ─────────────────────────────────────────────────────────────────────────────
# INGEST  —  PDF → chunks → embeddings → Pinecone
# ─────────────────────────────────────────────────────────────────────────────

def get_loader(file_path: str, file_name: str):
    """Pick the right loader based on file extension."""
    ext = Path(file_name).suffix.lower()
    if ext == ".pdf":
        return PyMuPDFLoader(file_path)
    elif ext == ".docx":
        return Docx2txtLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def ingest_pdfs(uploaded_files: list) -> int:
    all_chunks = []

    for uf in uploaded_files:
        # write to temp file so PyMuPDFLoader can open it
        suffix = Path(uf.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name

        docs = get_loader(tmp_path, uf.name).load()
        os.remove(tmp_path)

        for doc in docs:
            doc.metadata["source"] = uf.name   # tag with original filename

        chunks = RecursiveCharacterTextSplitter(
            chunk_size    = 1000,
            chunk_overlap = 150,
            separators    = ["\n\n", "\n", ". ", " ", ""],
        ).split_documents(docs)

        all_chunks.extend(chunks)

    # embed + upsert all chunks in one call
    PineconeVectorStore.from_documents(
        documents  = all_chunks,
        embedding  = load_embeddings(),
        index_name = INDEX_NAME,
    )

    return len(all_chunks)


# ─────────────────────────────────────────────────────────────────────────────
# RAG CHAIN  —  retriever → reranker → history-aware → LLM
# ─────────────────────────────────────────────────────────────────────────────

def build_chain(llm):
    # 1. Vector retriever  (top-10 by cosine similarity)
    vectorstore = PineconeVectorStore(
        index_name = INDEX_NAME,
        embedding  = load_embeddings(),
    )
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 2. Cohere reranker  (top-10 → best-5)
    reranker = CohereRerank(
        cohere_api_key = os.environ["COHERE_API_KEY"],
        model          = "rerank-english-v3.0",
        top_n          = 5,
    )
    compressed_retriever = ContextualCompressionRetriever(
        base_compressor = reranker,
        base_retriever  = base_retriever,
    )

    # 3. Reformulate follow-up questions using chat history
    #    e.g. "tell me more" → "tell me more about leave policy"
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "rewrite the question to be fully self-contained. "
         "Do NOT answer it — just rewrite if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, compressed_retriever, rephrase_prompt
    )

    # 4. Answer prompt
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. "
         "Answer using ONLY the context provided below. "
         "If the answer is not in the context say: "
         "'I don't have enough information to answer this.' "
         "Always cite the source at the end: [Source: filename, Page: X]\n\n"
         "{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # 5. Combine into retrieval chain
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        create_stuff_documents_chain(llm, answer_prompt),
    )

    # 6. Wrap with per-session chat history
    def get_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.chat_store:
            st.session_state.chat_store[session_id] = ChatMessageHistory()
        return st.session_state.chat_store[session_id]

    return RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key   = "input",
        history_messages_key = "chat_history",
        output_messages_key  = "answer",
    )


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# session state defaults
for k, v in {
    "messages":   [],
    "chat_store": {},
    "ingested":   False,
    "doc_names":  [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    groq_key   = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    session_id = st.text_input(
        "Session ID", value="session_1",
        help="Change this to start a fresh independent conversation"
    )

    st.divider()
    st.header("📁 Upload Documents")

    uploaded = st.file_uploader(
        "Choose files (PDF / DOCX / TXT)",
        type                  = ["pdf", "docx", "txt"],
        accept_multiple_files = True,
    )


    if uploaded and groq_key:
        if st.button("⚡ Ingest", use_container_width=True, type="primary"):
            with st.spinner("Reading, chunking and indexing..."):
                try:
                    init_pinecone()
                    t0    = time.time()
                    count = ingest_pdfs(uploaded)
                    elapsed = round(time.time() - t0, 1)
                    st.session_state.ingested  = True
                    st.session_state.doc_names = [f.name for f in uploaded]
                    st.success(f"✅ {count} chunks indexed in {elapsed}s")
                except Exception as e:
                    st.error(f"❌ Ingestion error: {e}")

    if st.session_state.doc_names:
        st.divider()
        st.caption("📂 **Indexed files**")
        for name in st.session_state.doc_names:
            st.caption(f"  📄 {name}")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages   = []
        st.session_state.chat_store = {}
        st.rerun()

    st.caption("Llama 3.3 70B · HuggingFace Embeddings · Pinecone · Cohere Rerank")


# ── MAIN CHAT ──────────────────────────────────────────────────────────────
st.title("🤖 RAG Chatbot with Chat History")
st.caption("Upload PDFs → Ask questions → Follow-up questions remember context")

# guard: need api key
if not groq_key:
    st.info("👈 Enter your Groq API key in the sidebar to get started.")
    st.stop()

# guard: need ingested docs
if not st.session_state.ingested:
    st.info("👈 Upload one or more PDFs and click Ingest.")
    st.stop()

# render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.caption(f"📄 `{s['source']}` — Page {s['page']}")

# chat input
if question := st.chat_input("Ask a question about your documents..."):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                llm   = load_llm(groq_key)
                chain = build_chain(llm)

                t0       = time.time()
                response = chain.invoke(
                    {"input": question},
                    config={"configurable": {"session_id": session_id}},
                )
                elapsed = round(time.time() - t0, 1)

                answer  = response["answer"]
                sources = [
                    {
                        "source": d.metadata.get("source", "unknown"),
                        "page":   d.metadata.get("page",   "N/A"),
                    }
                    for d in response.get("context", [])
                ]

                st.markdown(answer)
                if sources:
                    with st.expander("📚 Sources"):
                        for s in sources:
                            st.caption(f"📄 `{s['source']}` — Page {s['page']}")
                st.caption(f"⏱️ {elapsed}s · Llama 3.3 70B via Groq · Cohere Rerank")

                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

                
