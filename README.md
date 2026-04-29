# 🏗️ Construction Project Risk Predictor

> An AI-powered chatbot that analyzes construction project documents and delivers structured risk analysis using **Retrieval-Augmented Generation (RAG)** and **Model Context Protocol (MCP)** tools.
>
> **Status:** ✅ **Groq API Connected & Configured**

---

## 📌 Project Overview

The **Construction Project Risk Predictor** is a conversational AI system powered by **Groq's LLM** that allows users to upload construction-related documents (reports, budgets, timelines) and query them using natural language.

### 🔑 Key Features

- ✅ **Groq API Integration** - Ultra-fast LLM inference (llama-3.1-8b-instant)
- 🤖 Natural language Q&A over documents
- ⏱️ Automatic **Delay Risk Detection** (Low / Medium / High)
- 💰 Automatic **Cost Risk Detection** (Low / Medium / High)
- 🔍 **Contradiction detection** across documents
- 📋 **Evidence-based answers** with source references
- ⚡ Fast and interactive chatbot interface

---

## �️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              🎨 Streamlit Frontend (app.py)                  │
│         Chat UI · File Upload · Sidebar Controls             │
└──────────────────────────────────┬───────────────────────────┘
                                   │ process_query(query)
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│            🔗 MCP Pipeline (backend.py)                      │
│                                                              │
│  [1] Query Processor          → Clean & normalize input      │
│  [2] Retrieve Context (FAISS) → Find relevant chunks         │
│  [3] Relevance Filter         → Quality assurance            │
│  [4] Timeline Risk Analyzer   → Detect delays                │
│  [5] Budget Risk Analyzer     → Detect cost risks            │
│  [6] Contradiction Detector   → Find conflicts               │
│  [7] Answer Generator (Groq)  → Generate response ✅         │
│  [8] Evidence Validator       → Return sources               │
│                                                              │
└──────────────────────────────────┬───────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│            📊 RAG Engine (rag.py)                            │
│     Load → Chunk → Embed → Index → Retrieve                  │
│                                                              │
│  Vector DB: FAISS (Cosine Similarity)                        │
│  Embeddings: MiniLM-L6-v2                                    │
└──────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  🚀 Groq API ✅          │
                    │  Model: llama-3.1-8b     │
                    │  Status: CONNECTED       │
                    └──────────────────────────┘
```

---

## 📁 Project Structure

```
mcp/
│
├── 🎨 app.py                   # Streamlit frontend & UI
├── 🔗 backend.py               # MCP tools & processing pipeline
├── 📊 rag.py                   # RAG engine implementation
│
├── 📋 requirements.txt         # Python dependencies
├── .env                        # API keys (Groq - ✅ CONFIGURED)
│
├── 📂 data/                    # Input documents (.pdf / .txt)
├── 📂 logs/                    # Debug logs & metrics
├── 📂 tests/                   # Unit tests
│
├── README.md                   # This file


---

## 🛠️ Tech Stack

| Component | Technology | Status |
|-----------|-----------|--------|
| Frontend | Streamlit | ✅ Active |
| LLM | Groq API (llama-3.1-8b-instant) | ✅ CONNECTED |
| Embeddings | sentence-transformers (MiniLM-L6-v2) | ✅ Active |
| Vector Database | FAISS (cosine similarity) | ✅ Active |
| PDF Processing | pdfplumber | ✅ Active |
| Environment | python-dotenv | ✅ CONFIGURED |
| Framework | Python 3.10+ | ✅ Active |

---

## 🧩 MCP Tools & Pipeline

All tools are implemented in [backend.py](backend.py) and executed via the `process_query()` function.

### 🔹 Processing Pipeline

| Step | Tool | Purpose |
|------|------|---------|
| 1 | **Query Processor** | Cleans and normalizes user input |
| 2 | **Retriever (RAG)** | Fetches relevant chunks from FAISS |
| 3 | **Relevance Filter** | Removes low-quality/irrelevant chunks |
| 4 | **Timeline Risk Analyzer** | Detects project delay signals |
| 5 | **Budget Risk Analyzer** | Identifies cost overrun indicators |
| 6 | **Contradiction Detector** | Finds conflicting information |
| 7 | **Answer Generator (Groq)** | Generates responses via LLM ✅ |
| 8 | **Evidence Validator** | Returns supporting document snippets |

### 📤 Response Format

```json
{
    "answer": "string - Main response from LLM",
    "delay_risk": "Low/Medium/High",
    "cost_risk": "Low/Medium/High",
    "issues": ["list of identified issues"],
    "evidence": ["supporting document excerpts"],
    "chunks_used": 5,
    "llm_mode": "groq/fallback"
}
```

---

## � RAG (Retrieval-Augmented Generation) Pipeline

The RAG pipeline enables the system to search and retrieve relevant information from uploaded documents.

| Phase | Description | Implementation |
|-------|-------------|-----------------|
| **1. Load** | Load `.pdf` and `.txt` files from `data/` folder | pdfplumber |
| **2. Chunk** | Split text into small chunks (~100 words) | Text splitting with overlap |
| **3. Embed** | Convert text into vector embeddings | MiniLM-L6-v2 model |
| **4. Index** | Store vectors for fast retrieval | FAISS (Cosine Similarity) |
| **5. Retrieve** | Fetch top relevant chunks based on query | Semantic search |

---

## 🚀 Quick Start

### ✅ Prerequisites Check

- ✅ Python 3.10 or higher
- ✅ Groq API Key (already configured in `.env`)
- ✅ All dependencies listed in `requirements.txt`

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Groq API Configuration

Check that `.env` file contains your Groq API key:

```bash
# On Windows (PowerShell)
Get-Content .env1 | Select-String "GROQ_API_KEY"

# On Linux/Mac
grep GROQ_API_KEY .env1
```

**Output should show:**
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
```

✅ If key is present, Groq API is **ready to use**!

### Step 3: Prepare Documents (Optional)

Place your construction documents in the `data/` folder:

```
data/
  ├── project_report.pdf
  ├── budget_analysis.txt
  └── timeline.pdf
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

**Access the app:**
- 🌐 Local: http://localhost:8501
- 🌍 Network: http://<your-ip>:8501

---

## ⚡ Detailed Installation & Configuration

### Option 1: Fresh Setup

```bash
# 1. Clone or download the repository
cd mcp_kiro

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/Scripts/activate  # On Windows
# source venv/bin/activate    # On Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure Groq API (if not already done)
# Create or edit .env file:
# GROQ_API_KEY=your_groq_api_key_here

# 5. Start the app
streamlit run app.py
```

### Option 2: Get Free Groq API Key

1. Visit: https://console.groq.com
2. Sign up (no credit card required)
3. Generate API key
4. Add to `.env` file as `GROQ_API_KEY=your_key`

---

## 💻 Usage Guide

### Basic Workflow

1. **Upload Documents** (via sidebar)
   - Supported formats: `.pdf`, `.txt`
   - Files are automatically processed and indexed

2. **Ask Questions** (in chat input)
   - Natural language queries about your documents
   - System automatically detects risk types

3. **View Results**
   - Answer with risk assessments
   - Evidence from source documents
   - Reasoning behind the assessment

### Example Questions

- 📅 "What are the current project delays?"
- 💰 "Is the project over budget?"
- ⚠️ "What are the main risks?"
- 🔍 "Are there any contradictions?"
- ✅ "What is the project completion status?"
- 📊 "Summarize the project timeline"

---

## 🔧 API Reference

### Groq API Integration

**Model Used:** `llama-3.1-8b-instant`

**Features:**
- ⚡ Ultra-fast inference (~100ms per response)
- 🎯 Accurate context understanding
- 💻 Low computational requirements
- 🌍 High availability & reliability

**Configuration:**
- API Key: Stored in `.env` file
- Environment Variable: `GROQ_API_KEY`
- Status: ✅ **Currently Active**

**Example Usage:**
```python
from backend import process_query

response = process_query(
    query="What are the project risks?",
    llm_mode="groq"  # Uses Groq API
)
print(response["answer"])
```

---

## 🧪 Testing & Validation

### Run Tests

```bash
# Test MCP tools
pytest tests/test_mcp_tools.py -v

# Test RAG pipeline
pytest tests/test_rag.py -v

# Test all fixes
pytest test_fixes.py -v
```

### Performance Metrics

Check performance logs:
```bash
tail -f logs/metrics.jsonl
```

---


## 🎯 Project Roadmap

- ✅ Groq API Integration
- ✅ RAG Pipeline
- ✅ Risk Detection
- ⏳ Multi-language support
- ⏳ Advanced analytics dashboard
- ⏳ Document comparison tools

---
