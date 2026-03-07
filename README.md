---
title: Conversational RAG
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: "3.10"
app_file: app.py 
pinned: false
---

# 🤖 Conversational RAG Chatbot

> ⚠️ **Note:** This is a **demo version** of a RAG system I built and worked on in a production environment.
> Due to confidentiality, the actual production code cannot be shared publicly.
> This repository replicates the **core architecture and functionality** of the production system
> using the same tech stack, designed to demonstrate the concepts and implementation..

---

## 🏭 Production Context

In the production system, this RAG pipeline was used to:
- Process and index large volumes of enterprise documents
- Enable employees to query internal knowledge bases in natural language
- Return accurate answers with source citations from the original documents
- Maintain conversation context across multi-turn interactions

The production architecture used **FastAPI** as the backend service, consumed by a **React frontend**, containerized with **Docker**, and deployed on cloud infrastructure. This demo replicates the same AI pipeline with a **Streamlit UI** for quick visualization.

---

## 🔗 Live Demo
👉 [Try it here](https://huggingface.co/spaces/Vineel0508/Conversational-RAG)

---

## 🏗️ Architecture

```
Documents (PDF/DOCX/TXT)
        ↓
Load → Chunk → Embed (HuggingFace)
        ↓
Store in Pinecone Vector DB
        ↓
User Question → Vector Search (top 10)
        ↓
Cohere Reranker (best 5)
        ↓
Llama 3.3 70B (Groq) → Answer + Sources
```

---

## ✨ Features
- Upload PDF, DOCX, TXT files
- Ask questions and get answers with source citations
- Chat history preserved per session (follow-up questions work!)
- Powered by Llama 3.3 70B via Groq
- HuggingFace Embeddings (free, runs locally)
- Pinecone vector search
- Cohere reranking (top 10 → best 5)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector DB | Pinecone |
| Reranker | Cohere rerank-english-v3.0 |
| Framework | LangChain |
| UI | Streamlit (demo) / React (production) |
| Backend | FastAPI |
| Deployment | Hugging Face Spaces (demo) / Docker + Cloud (production) |

---

## 🚀 Run Locally

```bash
# Create conda environment
conda create -n rag-env python=3.11 -y
conda activate rag-env

# Install dependencies
pip install -r requirements.txt

# Add your API keys
cp .env.example .env
# Fill in your keys in .env

# Run Streamlit demo
streamlit run app.py

# Run FastAPI backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔑 Environment Variables

```
GROQ_API_KEY       → groq.com (free tier)
PINECONE_API_KEY   → pinecone.io (free tier)
COHERE_API_KEY     → cohere.com (free tier)
```

---

## 📡 API Endpoints (FastAPI)

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check |
| POST | /ingest | Upload document |
| POST | /query | Ask a question |
| DELETE | /index | Clear all vectors |
| GET | /sessions | List chat sessions |
| DELETE | /sessions/{id} | Clear a session |

---

## 👨‍💻 Author
**Vineel** — [GitHub](https://github.com/Vineel0508) · [LinkedIn](https://linkedin.com/in/vineel-amudala)
