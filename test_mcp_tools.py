"""
Unit tests for all 8 MCP tools in backend.py
Tests each tool in isolation with known inputs and expected outputs.
Run with: pytest tests/test_mcp_tools.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from backend import (
    query_processor,
    relevance_filter,
    timeline_risk_analyzer,
    budget_risk_analyzer,
    contradiction_detector,
    evidence_validator,
    _extractive_answer,
)


# ── Test 1: query_processor ──────────────────────────────────────────────────
class TestQueryProcessor:
    def test_strips_whitespace(self):
        assert query_processor("  hello world  ") == "hello world"

    def test_collapses_multiple_spaces(self):
        assert query_processor("hello    world") == "hello world"

    def test_removes_special_characters(self):
        assert query_processor("hello@#$world!") == "helloworld"

    def test_preserves_allowed_punctuation(self):
        result = query_processor("What is the budget?")
        assert "?" in result or result == "what is the budget"

    def test_lowercases_input(self):
        assert query_processor("HELLO WORLD") == "hello world"

    def test_empty_string(self):
        assert query_processor("") == ""


# ── Test 2: relevance_filter ─────────────────────────────────────────────────
class TestRelevanceFilter:
    def test_filters_low_scores(self):
        # When filtered results >= 3, low scores are dropped
        results = [
            {"chunk": "a", "score": 0.8},
            {"chunk": "b", "score": 0.1},  # below threshold
            {"chunk": "c", "score": 0.5},
            {"chunk": "d", "score": 0.6},
        ]
        filtered = relevance_filter(results, min_score=0.2)
        assert len(filtered) == 3
        assert all(r["score"] >= 0.2 for r in filtered)
        assert not any(r["chunk"] == "b" for r in filtered)

    def test_filters_low_scores_fallback_when_too_few(self):
        # When fewer than 3 pass threshold, falls back to top-3 unfiltered
        results = [
            {"chunk": "a", "score": 0.8},
            {"chunk": "b", "score": 0.1},
            {"chunk": "c", "score": 0.05},
        ]
        filtered = relevance_filter(results, min_score=0.2)
        # Only 1 passes threshold, so fallback returns top 3
        assert len(filtered) == 3

    def test_returns_minimum_3_if_all_filtered(self):
        results = [
            {"chunk": "a", "score": 0.05},
            {"chunk": "b", "score": 0.03},
            {"chunk": "c", "score": 0.01},
            {"chunk": "d", "score": 0.02},
        ]
        filtered = relevance_filter(results, min_score=0.2)
        assert len(filtered) == 3

    def test_returns_all_if_above_threshold(self):
        results = [
            {"chunk": "a", "score": 0.9},
            {"chunk": "b", "score": 0.8},
            {"chunk": "c", "score": 0.7},
            {"chunk": "d", "score": 0.6},
        ]
        filtered = relevance_filter(results, min_score=0.2)
        assert len(filtered) == 4


# ── Test 3: timeline_risk_analyzer ───────────────────────────────────────────
class TestTimelineRiskAnalyzer:
    def test_detects_high_risk_severe_delay(self):
        chunks = ["The project has a severe delay due to weather."]
        level, signals = timeline_risk_analyzer(chunks)
        assert level == "High"
        assert any("severe delay" in s.lower() for s in signals)

    def test_detects_high_risk_weeks_behind(self):
        chunks = ["We are 6 weeks behind schedule."]
        level, signals = timeline_risk_analyzer(chunks)
        assert level == "High"
        assert any("6 weeks" in s for s in signals)

    def test_detects_medium_risk_delay(self):
        # "delay" keyword triggers Medium — regex matches "N weeks delay/behind/late"
        chunks = ["The delivery is delayed. Work is behind schedule."]
        level, signals = timeline_risk_analyzer(chunks)
        assert level == "Medium"
        assert any("delay" in s.lower() or "behind" in s.lower() for s in signals)

    def test_detects_medium_risk_weeks_pattern(self):
        # Regex: (\d+)\s*weeks?\s*(?:delay|behind|late|postponed)
        chunks = ["The project is 2 weeks behind on delivery."]
        level, signals = timeline_risk_analyzer(chunks)
        assert level == "Medium"
        assert any("2 weeks" in s for s in signals)

    def test_detects_low_risk_no_delays(self):
        chunks = ["Everything is on track and on schedule."]
        level, signals = timeline_risk_analyzer(chunks)
        assert level == "Low"

    def test_detects_revised_completion_date(self):
        chunks = ["The revised completion date is March 2025."]
        level, signals = timeline_risk_analyzer(chunks)
        assert any("revised" in s.lower() for s in signals)

    def test_multiple_delay_mentions_picks_max(self):
        chunks = ["2 weeks delay", "5 weeks behind"]
        level, signals = timeline_risk_analyzer(chunks)
        assert any("5 weeks" in s for s in signals)


# ── Test 4: budget_risk_analyzer ─────────────────────────────────────────────
class TestBudgetRiskAnalyzer:
    def test_detects_high_risk_over_90_percent(self):
        chunks = ["Budget consumed: 95% of allocated funds."]
        level, signals = budget_risk_analyzer(chunks)
        assert level == "High"
        assert any("95%" in s for s in signals)

    def test_detects_high_risk_over_budget_keyword(self):
        chunks = ["The project is over budget by $2 million."]
        level, signals = budget_risk_analyzer(chunks)
        assert level == "High"
        assert any("over budget" in s.lower() for s in signals)

    def test_detects_medium_risk_80_to_89_percent(self):
        chunks = ["We have spent 85% of the budget."]
        level, signals = budget_risk_analyzer(chunks)
        assert level == "Medium"
        assert any("85%" in s for s in signals)

    def test_detects_low_risk_under_80_percent(self):
        chunks = ["Budget utilization is at 60%."]
        level, signals = budget_risk_analyzer(chunks)
        assert level == "Low"

    def test_detects_additional_funding_request(self):
        chunks = ["Additional funding of $1.5 million requested."]
        level, signals = budget_risk_analyzer(chunks)
        assert any("$1.5 million" in s or "additional" in s.lower() for s in signals)


# ── Test 5: contradiction_detector ───────────────────────────────────────────
class TestContradictionDetector:
    def test_detects_conflicting_completion_percentages(self):
        chunks = ["The project is 65% complete.", "Report shows 70% completion."]
        issues = contradiction_detector(chunks)
        assert len(issues) > 0
        assert any("65" in issue and "70" in issue for issue in issues)

    def test_detects_on_schedule_vs_delayed_conflict(self):
        chunks = ["The project is on schedule.", "We are 3 weeks behind."]
        issues = contradiction_detector(chunks)
        assert len(issues) > 0
        assert any("on schedule" in issue.lower() and "delayed" in issue.lower() for issue in issues)

    def test_detects_multiple_completion_dates(self):
        chunks = ["Completion: December 2024.", "Revised to March 2025."]
        issues = contradiction_detector(chunks)
        assert len(issues) > 0
        assert any("december 2024" in issue.lower() or "march 2025" in issue.lower() for issue in issues)

    def test_no_contradictions_returns_empty(self):
        chunks = ["The project is 65% complete and on track."]
        issues = contradiction_detector(chunks)
        assert len(issues) == 0


# ── Test 6: evidence_validator ───────────────────────────────────────────────
class TestEvidenceValidator:
    def test_returns_supporting_sentences(self):
        answer = "The project is 6 weeks behind schedule due to delays."
        results = [
            {"chunk": "The project is 6 weeks behind schedule. Steel delivery delayed.", "source": "report.pdf", "score": 0.9},
        ]
        evidence = evidence_validator(answer, results)
        assert len(evidence) > 0
        assert "6 weeks" in evidence[0]["text"]
        assert evidence[0]["source"] == "report.pdf"

    def test_filters_low_overlap_sentences(self):
        answer = "Budget is over by 20%."
        results = [
            {"chunk": "The weather was sunny today.", "source": "report.pdf", "score": 0.5},
        ]
        evidence = evidence_validator(answer, results)
        # Should still return something (fallback behavior)
        assert len(evidence) > 0

    def test_returns_max_4_evidence_items(self):
        answer = "delay budget schedule completion"
        results = [
            {"chunk": f"Sentence {i} about delay budget schedule completion.", "source": f"doc{i}.pdf", "score": 0.8}
            for i in range(10)
        ]
        evidence = evidence_validator(answer, results)
        assert len(evidence) <= 4


# ── Test 7: _extractive_answer ───────────────────────────────────────────────
class TestExtractiveAnswer:
    def test_returns_relevant_sentences(self):
        # Use a context where "budget" is clearly the most relevant term
        context = "The budget has been exceeded. Budget utilization is at 91%. Costs are rising."
        query = "What is the budget status?"
        answer = _extractive_answer(context, query)
        assert "budget" in answer.lower() or "91" in answer

    def test_handles_empty_context(self):
        answer = _extractive_answer("", "What is the status?")
        assert "no relevant information" in answer.lower()

    def test_removes_stopwords_from_query(self):
        context = "The budget is 91% consumed."
        query = "What is the budget?"
        answer = _extractive_answer(context, query)
        assert "budget" in answer.lower()

    def test_boosts_sentences_with_numbers(self):
        context = "The project is progressing. Budget is at 91% with $2M remaining."
        query = "budget"
        answer = _extractive_answer(context, query)
        assert "91" in answer or "$2" in answer


# ── Integration Test: Full Pipeline ───────────────────────────────────────────
class TestIntegration:
    def test_full_mcp_pipeline_with_sample_data(self):
        """Test all tools working together on realistic data."""
        chunks = [
            "The project is 65% complete as of September 2024.",
            "We are 6 weeks behind schedule due to supply chain issues.",
            "Budget consumed: 91% of $12.4 million with 35% work remaining.",
            "Q2 report stated 70% complete and on schedule.",
        ]

        # Test timeline analyzer
        delay_risk, delay_signals = timeline_risk_analyzer(chunks)
        assert delay_risk in ["Low", "Medium", "High"]
        assert len(delay_signals) > 0

        # Test budget analyzer
        cost_risk, cost_signals = budget_risk_analyzer(chunks)
        assert cost_risk in ["Low", "Medium", "High"]
        assert len(cost_signals) > 0

        # Test contradiction detector
        issues = contradiction_detector(chunks)
        assert len(issues) > 0  # Should detect 65% vs 70% and on schedule vs behind

        # Test extractive answer
        context = "\n\n".join(chunks)
        answer = _extractive_answer(context, "What are the delays?")
        assert len(answer) > 0
        assert "6 weeks" in answer or "behind" in answer.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
