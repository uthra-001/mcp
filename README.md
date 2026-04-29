# Construction Project Risk Predictor

> An AI-powered chatbot that analyzes construction project documents and delivers structured risk analysis using **Retrieval-Augmented Generation (RAG)** and **Model Context Protocol (MCP)** tools.

---

## 📌 1. Project Overview

The **Construction Project Risk Predictor** is a conversational AI system that allows users to upload construction-related documents (reports, budgets, timelines) and query them using natural language.

### 🔑 Key Features

* Natural language Q&A over documents
* Automatic **Delay Risk Detection** (Low / Medium / High)
* Automatic **Cost Risk Detection** (Low / Medium / High)
* **Contradiction detection** across documents
* **Evidence-based answers** with source references
* Fast and interactive chatbot interface

---

## 🏗️ 2. Architecture

```
┌──────────────────────────────────────────────┐
│         Streamlit Frontend (app.py)          │
│   Chat UI · File Upload · Sidebar Controls   │
└──────────────────────┬───────────────────────┘
                       │ process_query(query)
                       ▼
┌──────────────────────────────────────────────┐
│           MCP Pipeline (backend.py)          │
│                                              │
│  [1] query_processor                         │
│  [2] retrieve_context (FAISS)                │
│  [3] relevance_filter                        │
│  [4] timeline_risk_analyzer                  │
│  [5] budget_risk_analyzer                    │
│  [6] contradiction_detector                  │
│  [7] answer_generator (LLM / fallback)       │
│  [8] evidence_validator                      │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│              RAG Engine (rag.py)             │
│  Load → Chunk → Embed → Index → Retrieve     │
└──────────────────────────────────────────────┘
```

---

## 📁 3. Project Structure

```
construction-risk-predictor/
│
├── app.py              # Streamlit frontend
├── backend.py          # MCP tools + pipeline
├── rag.py              # RAG implementation
│
├── requirements.txt    # Dependencies
├── .env                # API key (not committed)
│
├── data/               # Input documents (.pdf / .txt)
│
└── logs/               # Optional debug logs (if enabled)
```

---

## ⚙️ 4. Tech Stack

* **Frontend:** Streamlit
* **LLM:** Groq API (llama-3.1-8b-instant)
* **Embeddings:** sentence-transformers (MiniLM-L6-v2)
* **Vector DB:** FAISS (cosine similarity)
* **PDF Processing:** pdfplumber
* **Environment:** python-dotenv

---

## 🧩 5. MCP Tools

All tools are implemented in `backend.py` and executed via `process_query()`.

### 🔹 Core Tools

1. **Query Processor**
   Cleans and normalizes user input

2. **Retriever (RAG)**
   Fetches relevant chunks using FAISS

3. **Relevance Filter**
   Removes low-quality chunks

4. **Timeline Risk Analyzer**
   Detects delay-related signals

5. **Budget Risk Analyzer**
   Detects cost overruns

6. **Contradiction Detector**
   Identifies conflicting information

7. **Answer Generator**
   Generates response using LLM or fallback

8. **Evidence Validator**
   Returns supporting document snippets

---

### 📤 Output Format

```python
{
    "answer": str,
    "delay_risk": "Low/Medium/High",
    "cost_risk": "Low/Medium/High",
    "issues": list,
    "evidence": list,
    "chunks_used": int,
    "llm_mode": str
}
```

---

## 🔍 6. RAG Pipeline

### Step 1 — Document Loading

Loads `.pdf` and `.txt` files from `data/`

### Step 2 — Chunking

Splits text into small chunks (~100 words)

### Step 3 — Embedding

Converts text into vector form using MiniLM

### Step 4 — Indexing

Stores vectors in FAISS for fast search

### Step 5 — Retrieval

Fetches top relevant chunks based on query

---

## ⚡ 7. Setup & Installation

### Prerequisites

* Python 3.10+

---

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 2 — Add API key

Create `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

---

### Step 3 — Add documents (optional)

Place files in:

```
data/
```

---

## ▶️ 8. Run the Application

```bash
streamlit run app.py
```

Open:
👉 http://localhost:8501

---

## 💬 9. Usage Guide

* Enter question in chat input
* Upload documents via sidebar
* View answers with evidence
* Ask different queries (delay, budget, risks)

---

## ❓ 10. Sample Questions

* What are the current project delays?
* Is the project over budget?
* What are the main risks?
* Are there any contradictions?
* What is the completion status?

---

## 🎯 11. Conclusion

This project demonstrates how **RAG + MCP architecture** can be used to build an intelligent, explainable, and practical risk analysis system for real-world construction scenarios.

---
