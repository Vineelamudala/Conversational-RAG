"""
RAG Chatbot — FastAPI Production Backend
Supports: PDF, DOCX, TXT

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Then open:
    http://localhost:8000/docs   ← interactive API docs (free from FastAPI)

.env file needs:
    GROQ_API_KEY
    PINECONE_API_KEY
    COHERE_API_KEY
"""

import os
import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ── LangChain ─────────────────────────────────────────────────────────────
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

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

INDEX_NAME      = "rag-fastapi"
DIMENSION       = 384
SUPPORTED_TYPES = {".pdf", ".docx", ".txt"}


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZE ONCE AT STARTUP
# ─────────────────────────────────────────────────────────────────────────────

# Embeddings — HuggingFace, runs locally
embeddings = HuggingFaceEmbeddings(
    model_name   = "all-MiniLM-L6-v2",
    model_kwargs = {"device": "cpu"},
    encode_kwargs= {"normalize_embeddings": True}
)

# LLM — Groq Llama 3
llm = ChatGroq(
    groq_api_key = os.environ["GROQ_API_KEY"],
    model_name   = "llama-3.3-70b-versatile",
    temperature  = 0,
    max_tokens   = 1024,
)

# Pinecone index
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name      = INDEX_NAME,
        dimension = DIMENSION,
        metric    = "cosine",
        spec      = ServerlessSpec(cloud="aws", region="us-east-1"),
    )
pinecone_index = pc.Index(INDEX_NAME)

# In-memory chat history store  { session_id: ChatMessageHistory }
chat_store: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_loader(file_path: str, file_name: str):
    """Pick the right LangChain loader based on file extension."""
    ext = Path(file_name).suffix.lower()
    if ext == ".pdf":
        return PyMuPDFLoader(file_path)
    elif ext == ".docx":
        return Docx2txtLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def get_rag_chain():
    """
    Builds full RAG chain:
    Pinecone vector search (top 10)
    → Cohere reranker (best 5)
    → History-aware query reformulation
    → Llama 3 answer
    """
    # Vector retriever
    vectorstore    = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Cohere reranker
    reranker = CohereRerank(
        cohere_api_key = os.environ["COHERE_API_KEY"],
        model          = "rerank-english-v3.0",
        top_n          = 5,
    )
    compressed_retriever = ContextualCompressionRetriever(
        base_compressor = reranker,
        base_retriever  = base_retriever,
    )

    # Reformulate follow-up questions using chat history
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and latest question, rewrite the question "
         "to be fully self-contained. Do NOT answer — just rewrite if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, compressed_retriever, rephrase_prompt
    )

    # Answer prompt
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer using ONLY the context below. "
         "If the answer is not in the context say: "
         "'I don't have enough information to answer this.' "
         "Always cite: [Source: filename, Page: X]\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Full chain
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        create_stuff_documents_chain(llm, answer_prompt),
    )

    # Wrap with session-based chat history
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in chat_store:
            chat_store[session_id] = ChatMessageHistory()
        return chat_store[session_id]

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key   = "input",
        history_messages_key = "chat_history",
        output_messages_key  = "answer",
    )


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "RAG Chatbot API",
    description = "Production RAG backend — PDF/DOCX/TXT + Reranking + Chat History",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # restrict to your frontend domain in real production
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Request / Response models ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str
    session_id: str = "default"

class QueryResponse(BaseModel):
    answer:     str
    sources:    list
    session_id: str


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — DevOps pings this to confirm service is alive."""
    return {
        "status":  "ok",
        "version": "1.0.0",
        "model":   "llama-3.3-70b-versatile",
        "index":   INDEX_NAME,
        "supported_file_types": list(SUPPORTED_TYPES),
    }


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Upload a document (PDF / DOCX / TXT) → chunk → embed → store in Pinecone.
    React frontend calls this when user uploads a file.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code = 400,
            detail      = f"Unsupported file type: '{ext}'. Allowed: {SUPPORTED_TYPES}"
        )

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Load
        docs = get_loader(tmp_path, file.filename).load()
        for doc in docs:
            doc.metadata["source"] = file.filename

        # Chunk
        chunks = RecursiveCharacterTextSplitter(
            chunk_size    = 1000,
            chunk_overlap = 150,
            separators    = ["\n\n", "\n", ". ", " ", ""],
        ).split_documents(docs)

        # Embed + store in Pinecone
        PineconeVectorStore.from_documents(
            documents  = chunks,
            embedding  = embeddings,
            index_name = INDEX_NAME,
        )

        return {
            "status":   "success",
            "filename": file.filename,
            "type":     ext,
            "pages":    len(docs),
            "chunks":   len(chunks),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.remove(tmp_path)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question → vector search → rerank → LLM → answer with sources.
    React frontend calls this when user sends a chat message.
    Chat history maintained per session_id.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        chain    = get_rag_chain()
        response = chain.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": request.session_id}},
        )

        sources = [
            {
                "source": d.metadata.get("source", "unknown"),
                "page":   d.metadata.get("page",   "N/A"),
            }
            for d in response.get("context", [])
        ]

        return QueryResponse(
            answer     = response["answer"],
            sources    = sources,
            session_id = request.session_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/index")
def clear_index():
    """Clear all vectors from Pinecone. Use when re-ingesting updated docs."""
    try:
        pinecone_index.delete(delete_all=True)
        chat_store.clear()
        return {"status": "cleared", "index": INDEX_NAME}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
def list_sessions():
    """List all active chat sessions."""
    return {"sessions": list(chat_store.keys()), "count": len(chat_store)}


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    """Clear chat history for a specific session."""
    if session_id not in chat_store:
        raise HTTPException(status_code=404, detail="Session not found.")
    del chat_store[session_id]
    return {"status": "cleared", "session_id": session_id}


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
