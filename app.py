"""
Construction Project Risk Predictor — Streamlit Frontend
Run with: streamlit run app.py
"""

import streamlit as st
import os
from dotenv import load_dotenv
import backend

load_dotenv()
from backend import process_query

st.set_page_config(
    page_title="Construction Risk Predictor",
    page_icon="🏗️",
    layout="wide",
)

# ── Initialize backend once (cached) ──────────────────────────────────────────
@st.cache_resource
def init_backend():
    """Initialize RAG pipeline once at startup."""
    backend._ensure_initialized()
    return True

# Call during initial load
init_backend()

@st.cache_data
def process_query_cached(query: str):
    """Cache query results to avoid re-processing identical queries."""
    return process_query(query)

st.set_page_config(
    page_title="Construction Risk Predictor",
    page_icon="🏗️",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f0f4ff 0%, #fafbff 100%);
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
/* Make chat input stick to bottom */
[data-testid="stChatInput"] {
    border-top: 1px solid #e2e8f0;
    background: #ffffff;
    padding: 10px 0;
}
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 4px 8px;
    margin-bottom: 6px;
    background: #ffffff;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.risk-card {
    border-radius: 12px;
    padding: 14px 18px;
    margin: 5px 0;
    border-left: 5px solid;
}
.risk-high   { background: #fff5f5; border-color: #e53e3e; }
.risk-medium { background: #fffaf0; border-color: #dd6b20; }
.risk-low    { background: #f0fff4; border-color: #38a169; }
.pill {
    display: inline-block;
    background: #edf2f7;
    color: #4a5568;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 2px 3px;
    border: 1px solid #e2e8f0;
}
.issue-item {
    background: #fffbeb;
    border-left: 3px solid #dd6b20;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.88rem;
    color: #7b341e;
}
.answer-box {
    background: #ffffff;
    border: 1px solid #bee3f8;
    border-radius: 10px;
    padding: 14px 18px;
    color: #2d3748;
    line-height: 1.7;
    font-size: 0.95rem;
    box-shadow: 0 1px 6px rgba(49,130,206,0.08);
}
.evidence-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 5px 0;
    font-size: 0.83rem;
    color: #4a5568;
}
.evidence-source { font-size: 0.73rem; color: #3182ce; margin-top: 4px; }
.metric-chip {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.8rem;
}
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #276749, #38a169) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
RISK_CONFIG = {
    "High":   {"color": "#e53e3e", "bg": "risk-high",   "icon": "🔴", "emoji": "🚨"},
    "Medium": {"color": "#dd6b20", "bg": "risk-medium", "icon": "🟡", "emoji": "⚠️"},
    "Low":    {"color": "#38a169", "bg": "risk-low",    "icon": "🟢", "emoji": "✅"},
}

def risk_card(label, level, signals):
    cfg = RISK_CONFIG.get(level, {"color": "#888", "bg": "", "icon": "⚪", "emoji": "ℹ️"})
    pills = "".join(f'<span class="pill">{s}</span>' for s in signals) if signals else ""
    return f"""<div class="risk-card {cfg['bg']}">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
            <span style="font-size:1.2rem">{cfg['icon']}</span>
            <span style="color:{cfg['color']};font-weight:700;">{label}</span>
            <span class="metric-chip" style="background:{cfg['color']}22;color:{cfg['color']};border:1px solid {cfg['color']}55;">
                {cfg['emoji']} {level}
            </span>
        </div>
        <div>{pills}</div>
    </div>"""

def render_result(result):
    # Answer
    st.markdown(
        f'<div class="answer-box">💬 <strong>Answer</strong><br><br>{result["answer"]}</div>',
        unsafe_allow_html=True,
    )
    
    # If insufficient data, don't show risk analysis
    if "Insufficient data" in result["answer"]:
        return
    
    llm = result.get("mode", "Fallback")
    llm_color = "#38a169" if "LLM" in llm else "#dd6b20"
    st.markdown(
        f"<small style='color:#a0aec0;'>Analyzed {result.get('chunks_used','?')} chunks &nbsp;·&nbsp; "
        f"<span style='color:{llm_color};font-weight:600;'>{llm}</span></small>",
        unsafe_allow_html=True,
    )

    # Risk cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(risk_card("Delay Risk", result["delay_risk"], result.get("delay_signals", [])), unsafe_allow_html=True)
    with col2:
        st.markdown(risk_card("Cost Risk", result["cost_risk"], result.get("cost_signals", [])), unsafe_allow_html=True)

    # Issues
    if result["issues"]:
        st.markdown("<br>", unsafe_allow_html=True)
        for issue in result["issues"]:
            st.markdown(f'<div class="issue-item">⚡ {issue}</div>', unsafe_allow_html=True)
    else:
        st.markdown("<small style='color:#38a169;'>✅ No contradictions detected.</small>", unsafe_allow_html=True)

    # Evidence
    with st.expander("📄 Evidence snippets", expanded=False):
        for i, ev in enumerate(result.get("evidence", []), 1):
            score_pct = f"{ev.get('score',0):.0%}" if ev.get("score", 0) <= 1 else ""
            st.markdown(
                f'<div class="evidence-card"><strong style="color:#3182ce;">[{i}]</strong> {ev["text"]}'
                f'<div class="evidence-source">📁 {ev["source"]}'
                f'{"&nbsp;·&nbsp;relevance: " + score_pct if score_pct else ""}</div></div>',
                unsafe_allow_html=True,
            )

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str|dict}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏗️ Risk Predictor")
    st.markdown("<small style='color:#888;'>Construction AI · RAG + MCP</small>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### LLM Status")
    if getattr(backend, "_groq_client", None):
        st.success("✅ Groq LLM connected")
    else:
        st.warning("No API key — extractive fallback")
        st.caption("Add GROQ_API_KEY to .env and restart.")

    st.divider()
    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload .txt or .pdf files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        from pathlib import Path
        import rag
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        saved = []
        for f in uploaded_files:
            (data_dir / f.name).write_bytes(f.read())
            saved.append(f.name)
        if saved:
            with st.spinner("Rebuilding index..."):
                n = rag.initialize()
            st.success(f"✅ {n} chunks indexed from {len(saved)} file(s)")

    st.divider()
    st.markdown("### 💡 Quick questions")
    suggestions = [
        "What are the current delays?",
        "Is the project over budget?",
        "What are the main risks?",
        "Any safety issues?",
        "What is the completion date?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"s_{s[:15]}"):
            st.session_state["prefill"] = s
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:16px 0 6px;">
    <h1 style="font-size:2rem;font-weight:800;
               background:linear-gradient(90deg,#2b6cb0,#3182ce,#38a169);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:2px;">
        🏗️ Construction Project Risk Predictor
    </h1>
    <p style="color:#718096;font-size:0.9rem;margin:0;">
        RAG-powered risk analysis · Ask anything about your project
    </p>
</div>
""", unsafe_allow_html=True)

# ── Render full chat history ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👷"):
            st.markdown(f"**{msg['content']}**")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            render_result(msg["content"])

# ── Chat input (always at bottom, native Streamlit) ───────────────────────────
prefill = st.session_state.pop("prefill", "")

user_input = st.chat_input(
    placeholder="Ask about delays, budget, risks, safety...",
    key="chat_input",
)

# Handle sidebar suggestion prefill
if prefill and not user_input:
    user_input = prefill

if user_input and user_input.strip():
    query = user_input.strip()

    # Show user message immediately
    with st.chat_message("user", avatar="👷"):
        st.markdown(f"**{query}**")

    # Stream the thinking indicator then get result
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing..."):
            result = process_query_cached(query)
        render_result(result)

    # Save to history
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": result})
