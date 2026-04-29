🏗️ Construction Project Risk Predictor

AI-powered system for analyzing construction documents and detecting risks using RAG (Retrieval-Augmented Generation) and MCP (Model Context Protocol).

🚀 Status: Groq API Integrated & Fully Functional

📌 Overview

This project is a conversational AI tool that allows users to upload construction-related documents and extract meaningful insights through natural language queries.

It combines:

🔍 Semantic search (RAG)
🤖 LLM reasoning (Groq)
🧠 Custom risk analysis tools (MCP)

to deliver accurate, evidence-based risk predictions.

✨ Key Features
⚡ Fast responses using Groq LLM (llama-3.1-8b-instant)
📄 Document-based Q&A (PDF / TXT)
⏱️ Delay risk detection (Low / Medium / High)
💰 Cost risk detection (Low / Medium / High)
⚠️ Contradiction detection across documents
📋 Evidence-backed answers
💬 Interactive chatbot UI (Streamlit)
🏗️ System Architecture
User Query
   │
   ▼
🎨 Streamlit Frontend (app.py)
   │
   ▼
🔗 MCP Pipeline (backend.py)
   ├── Query Processing
   ├── Context Retrieval (FAISS)
   ├── Risk Analysis
   ├── Contradiction Detection
   └── Answer Generation (Groq)
   │
   ▼
📊 RAG Engine (rag.py)
   │
   ▼
🚀 Groq API (LLM)
📁 Project Structure
mcp_kiro/
│
├── app.py                # Streamlit UI
├── backend.py            # MCP pipeline
├── rag.py                # RAG engine
│
├── data/                 # Input documents
├── logs/                 # Logs & metrics
├── tests/                # Unit tests
│
├── requirements.txt
├── .env                  # API keys
│
├── README.md
├── ACCURACY_FIXES.md
└── PERFORMANCE_IMPROVEMENTS.md
🛠️ Tech Stack
Frontend: Streamlit
LLM: Groq (llama-3.1-8b-instant)
Embeddings: sentence-transformers (MiniLM-L6-v2)
Vector DB: FAISS
PDF Processing: pdfplumber
Language: Python 3.10+
🔄 MCP Processing Pipeline
Query Cleaning
Context Retrieval (FAISS)
Relevance Filtering
Delay Risk Detection
Cost Risk Detection
Contradiction Detection
LLM Response Generation (Groq)
Evidence Extraction
📤 Response Format
{
  "answer": "Generated response",
  "delay_risk": "Low/Medium/High",
  "cost_risk": "Low/Medium/High",
  "issues": [],
  "evidence": [],
  "chunks_used": 5,
  "llm_mode": "groq"
}
🔍 RAG Pipeline
Load: Read PDF/TXT files
Chunk: Split into smaller text segments
Embed: Convert text into vectors
Index: Store in FAISS
Retrieve: Fetch relevant content
🚀 Quick Start
1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Add API Key

Create .env file:

GROQ_API_KEY=your_api_key_here
3️⃣ Add Documents
data/
 ├── report.pdf
 ├── budget.txt
 └── timeline.pdf
4️⃣ Run App
streamlit run app.py

👉 Open: http://localhost:8501

💻 Usage
Upload documents
Ask questions
View:
Risk levels
Insights
Evidence
Example Queries
“What are the project risks?”
“Is the project delayed?”
“Are there cost overruns?”
“Find contradictions in reports”
🧪 Testing
pytest tests/ -v
⚡ Performance Highlights
⚡ Fast inference using Groq (~100ms)
📊 Efficient retrieval using FAISS
🎯 High accuracy via RAG + MCP
