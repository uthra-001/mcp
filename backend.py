import re
import os
from dotenv import load_dotenv
import rag

load_dotenv()

# ─────────────────────────────────────────────
# LLM SETUP
# ─────────────────────────────────────────────

_groq_client = None

def _init_groq():
    global _groq_client
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("[LLM] No GROQ key → fallback mode")
        return
    try:
        from groq import Groq
        _groq_client = Groq(api_key=api_key)
        print("[LLM] Groq connected")
    except:
        print("[LLM] Groq failed → fallback")

_init_groq()

# ─────────────────────────────────────────────────────────────
# SAMPLE DATA (used if no files in /data)
# ─────────────────────────────────────────────────────────────

SAMPLE_DATA = """
PROJECT 1: Smart City Bridge Construction
Location: Chennai, India
Start Date: January 2025
Planned Completion: June 2025
Current Status: 50% completed
Timeline Delay: Currently 3 weeks behind schedule. Expected delay of 4-6 additional weeks due to steel supply chain disruptions and monsoon weather conditions.
Budget Status: 60% of budget spent for 50% work completion. Projected cost overrun: 8-12%.
Risk Level: High risk of schedule delay, moderate cost overrun.

PROJECT 2: Green Commercial Complex
Location: Bangalore, India
Start Date: March 2024
Planned Completion: March 2026
Current Status: 35% completed
Timeline: On track with minor 5-day delays due to design changes. Expected to complete 2 weeks ahead.
Budget: 40% of budget spent for 35% work. No cost overrun expected.
Risk Level: Low risk.

PROJECT 3: Residential Housing Development
Location: Mumbai, India
Start Date: June 2024
Planned Completion: December 2025
Current Status: 72% completed
Timeline Delay: 2 weeks behind schedule. Labor shortage caused 3-week delay in foundation work. Expected completion in January 2026 (1 month delay).
Budget Status: 85% of budget spent for 72% work. Cost overrun of 5-7% likely.
Risk Level: Medium risk for schedule, low risk for cost.

PROJECT 4: Industrial Warehouse Expansion
Location: Delhi, India
Start Date: September 2024
Planned Completion: August 2025
Current Status: 25% completed
Timeline: On schedule. No delays reported.
Budget: 22% of budget spent for 25% work. Tracking well.
Risk Level: Low risk.
"""
"""
backend.py - MCP Tools + RAG Pipeline
Construction Project Risk Predictor
"""

# ─────────────────────────────────────────────
# INIT RAG
# ─────────────────────────────────────────────

_initialized = False

def _ensure_initialized():
    global _initialized
    if not _initialized:
        rag.initialize()
        _initialized = True


# ─────────────────────────────────────────────
# MCP TOOL 1 - Query Processor
# ─────────────────────────────────────────────

def query_processor(query: str) -> str:
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r"[^\w\s]", "", query)
    return query.lower()


# ─────────────────────────────────────────────
# INTENT DETECTION - What is user asking about?
# ─────────────────────────────────────────────

def detect_query_intent(query: str) -> str:
    """
    Detect if query is about: 'cost', 'time', 'risk', or 'general'
    Priority: time > cost > risk (time always takes priority if present)
    """
    query_lower = query.lower()
    
    time_kw = ['delay', 'timeline', 'schedule', 'deadline', 'completion', 'time', 'duration', 'behind', 'ahead']
    cost_kw = ['cost', 'budget', 'expense', 'spending', 'amount', 'spent', 'allocat', 'price', 'overrun']
    risk_kw = ['risk', 'issue', 'problem', 'danger', 'challenge']
    
    time_match = any(kw in query_lower for kw in time_kw)
    cost_match = any(kw in query_lower for kw in cost_kw)
    risk_match = any(kw in query_lower for kw in risk_kw)
    
    # Priority: time > cost > risk
    if time_match:
        return 'time'
    elif cost_match:
        return 'cost'
    elif risk_match:
        return 'risk'
    return 'general'


# ─────────────────────────────────────────────
# MCP TOOL 2 - Context Retriever
# ─────────────────────────────────────────────

def retrieve_context(query: str, top_k: int = 3):
    return rag.retrieve(query, top_k=top_k)


# ─────────────────────────────────────────────
# MCP TOOL 3 - Relevance Filter
# ─────────────────────────────────────────────

def relevance_filter(results, min_score=0.05):
    return [r for r in results if r["score"] >= min_score]


# ─────────────────────────────────────────────
# MCP TOOL 4 - Timeline Risk Analyzer
# ─────────────────────────────────────────────

def timeline_risk_analyzer(chunks):
    text = " ".join(chunks).lower()

    if "weeks behind" in text or "delay" in text:
        return "High", ["Project delay detected"]
    return "Low", []


# ─────────────────────────────────────────────
# MCP TOOL 5 - Budget Risk Analyzer
# ─────────────────────────────────────────────

def budget_risk_analyzer(chunks):
    text = " ".join(chunks).lower()

    percentages = [int(x) for x in re.findall(r"(\d+)%", text)]

    if any(p >= 90 for p in percentages):
        return "High", ["Budget above 90%"]
    elif any(p >= 75 for p in percentages):
        return "Medium", ["Budget above 75%"]
    return "Low", []


# ─────────────────────────────────────────────
# MCP TOOL 6 - Contradiction Detector
# ─────────────────────────────────────────────

def contradiction_detector(chunks):
    text = " ".join(chunks).lower()
    issues = []

    if "on schedule" in text and "delay" in text:
        issues.append("Conflicting schedule info")

    return issues


# ─────────────────────────────────────────────
# MCP TOOL 7 - Answer Generator
# ─────────────────────────────────────────────

def _extractive_answer(context, query):
    sentences = re.split(r"(?<=[.!?])\s+", context)
    query_terms = set(query.lower().split())

    scored = []
    for s in sentences:
        words = set(s.lower().split())
        score = len(words & query_terms)
        if score > 0:
            scored.append((score, s))

    if not scored:
        return "No relevant information found."

    scored.sort(reverse=True)
    return " ".join(s for _, s in scored[:2])


def _llm_answer(context, query, intent='general'):
    """Generate LLM answer with intent-specific filtering"""
    
    # Build intent-specific instructions
    if intent == 'cost':
        filter_instruction = """
CRITICAL - COST QUERY:
- ONLY extract: Budget amounts, spending, cost status, budget utilization percentages
- EXCLUDE: Timeline, deadlines, delays, completion dates
- Format: List the cost information concisely with exact numbers
- DO NOT include: Any timeline or delay information
"""
    elif intent == 'time':
        filter_instruction = """
CRITICAL - TIME/DELAY QUERY:
- ONLY extract: Delays, timelines, deadlines, completion dates, schedule status
- EXCLUDE: Budget amounts, costs, spending information
- Format: List the timeline information concisely with exact dates
- DO NOT include: Any budget or cost information
"""
    elif intent == 'risk':
        filter_instruction = """
CRITICAL - RISK QUERY:
- ONLY extract: Risk levels, risk factors, indicators
- Format: List the risk information concisely
- DO NOT include: Detailed timeline or budget unless critical to risk
"""
    else:
        filter_instruction = """
- Answer the question directly
- Use exact numbers and specific information
- Keep answer concise and focused
"""
    
    prompt = f"""{filter_instruction}

Context:
{context}

Question:
{query}

Answer (factual, no conclusions):"""

    try:
        res = _groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        return res.choices[0].message.content.strip()
    except:
        return _extractive_answer(context, query)


def answer_generator(context, query, intent='general'):
    if _groq_client:
        return _llm_answer(context, query, intent)
    return _extractive_answer(context, query)


# ─────────────────────────────────────────────
# POST-PROCESSING - Clean answer based on intent
# ─────────────────────────────────────────────

def clean_answer_by_intent(answer: str, intent: str) -> str:
    """Remove unrelated information from answer based on query intent"""
    
    if intent == 'cost':
        # Remove timeline/delay sentences from cost answer
        lines = answer.split('\n')
        filtered = []
        timeline_keywords = ['deadline', 'schedule', 'timeline', 'delay', 'completion', 'behind', 'ahead', 'expected completion']
        
        for line in lines:
            line_lower = line.lower()
            # Skip if line is ONLY about timeline
            if any(kw in line_lower for kw in timeline_keywords):
                # Check if line also mentions budget/cost
                if not any(kw in line_lower for kw in ['budget', 'cost', 'spent', 'amount', 'inr']):
                    continue
            filtered.append(line)
        
        answer = '\n'.join(filtered).strip()
        
    elif intent == 'time':
        # Remove cost/budget sentences from time answer
        lines = answer.split('\n')
        filtered = []
        cost_keywords = ['budget', 'cost', 'spent', 'amount', 'allocat', 'inr', 'overrun', 'spending']
        
        for line in lines:
            line_lower = line.lower()
            # Skip if line is ONLY about cost
            if any(kw in line_lower for kw in cost_keywords):
                # Check if line also mentions timeline/delay
                if not any(kw in line_lower for kw in ['delay', 'schedule', 'timeline', 'completion', 'deadline']):
                    continue
            filtered.append(line)
        
        answer = '\n'.join(filtered).strip()
    
    return answer


# ─────────────────────────────────────────────
# MCP TOOL 8 - Evidence Validator
# ─────────────────────────────────────────────

def evidence_validator(results):
    return [
        {
            "text": r["chunk"],
            "source": r["source"],
            "score": r["score"],
        }
        for r in results[:2]
    ]


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def process_query(query: str):
    _ensure_initialized()

    clean_query = query_processor(query)
    
    # Detect what user is asking about
    intent = detect_query_intent(query)

    raw_results = retrieve_context(clean_query)
    results = relevance_filter(raw_results)

    if not results:
        return {
            "answer": "No relevant data found.",
            "delay_risk": "Unknown",
            "cost_risk": "Unknown",
            "issues": [],
            "evidence": [],
        }

    chunks = [r["chunk"] for r in results[:2]]
    context = "\n".join(chunks)

    delay_risk, _ = timeline_risk_analyzer(chunks)
    cost_risk, _ = budget_risk_analyzer(chunks)
    issues = contradiction_detector(chunks)

    # Generate answer with intent awareness
    answer = answer_generator(context, query, intent)
    
    # Clean up answer - remove unrelated info based on intent
    answer = clean_answer_by_intent(answer, intent)
    
    evidence = evidence_validator(results)

    return {
        "answer": answer,
        "delay_risk": delay_risk,
        "cost_risk": cost_risk,
        "issues": issues,
        "evidence": evidence,
    }