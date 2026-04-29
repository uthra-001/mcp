"""
Quick test script to verify the chatbot fixes
Run: python test_fixes.py
"""

import sys
sys.path.insert(0, '.')

# Test just the intent detection first
from backend import detect_query_intent

print("\n" + "="*70)
print("TESTING QUERY INTENT DETECTION")
print("="*70)

test_queries = [
    ("what are the time risk", "time"),
    ("what is the time delay of this project", "time"),
    ("what is the project cost", "cost"),
    ("what is the budget", "cost"),
    ("what is the risk level", "risk"),
    ("show me the delay information", "time"),
]

for query, expected_intent in test_queries:
    detected_intent = detect_query_intent(query)
    status = "✅" if detected_intent == expected_intent else "❌"
    print(f"{status} Query: '{query}'")
    print(f"   Expected: {expected_intent}, Got: {detected_intent}\n")

print("✅ Intent detection tests complete!")
print("="*70)

