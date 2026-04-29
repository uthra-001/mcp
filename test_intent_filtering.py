"""
Quick test to verify cost/time filtering works correctly
"""

from backend import detect_query_intent, clean_answer_by_intent

print("\n" + "="*70)
print("TESTING INTENT DETECTION")
print("="*70)

test_queries = [
    "what is the cost required to complete this project?",
    "what is the time delay?",
    "what is the risk level?",
    "show me the budget",
    "how many weeks behind schedule?",
]

for q in test_queries:
    intent = detect_query_intent(q)
    print(f"Query: '{q}'")
    print(f"Intent: {intent}\n")

print("="*70)
print("TESTING ANSWER CLEANING")
print("="*70)

# Test cost answer cleanup
cost_answer = """Total Budget Allocated: 10,00,000 INR
Amount Spent: 8,50,000 INR
Remaining Budget: 1,50,000 INR
Original Deadline: June 2025
Updated Expected Completion: August 2025
Budget utilization exceeds 80%
Timeline mismatch detected
Project delay expected"""

print("\nBefore cleaning (cost query):")
print(cost_answer)

cleaned = clean_answer_by_intent(cost_answer, 'cost')
print("\nAfter cleaning (SHOULD REMOVE TIMELINE):")
print(cleaned)

# Test time answer cleanup
time_answer = """Currently 3 weeks behind schedule
Timeline Delay: Currently 3 weeks behind
Total Budget Spent: 8,50,000 INR
Budget Status: 60% spent
Expected completion: August 2025"""

print("\n" + "-"*70)
print("\nBefore cleaning (time query):")
print(time_answer)

cleaned = clean_answer_by_intent(time_answer, 'time')
print("\nAfter cleaning (SHOULD REMOVE BUDGET/COST):")
print(cleaned)

print("\n" + "="*70)
print("✅ Tests complete!")
print("="*70)
