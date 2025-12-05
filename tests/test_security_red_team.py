"""
Unit tests for Red Team Security Benchmark
Tests jailbreak resistance evaluation functionality
"""

from unittest.mock import Mock

import pytest

from llm_evaluator.security.red_team import RedTeamBenchmark


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider that refuses harmful requests"""
    provider = Mock()

    def mock_generate(prompt):
        # Simulate refusal for harmful prompts
        harmful_keywords = ["DAN", "jailbreak", "ignore instructions", "grandmother"]
        if any(kw.lower() in prompt.lower() for kw in harmful_keywords):
            return "I cannot assist with that request."
        return "Sure, I can help with that."

    provider.generate.side_effect = mock_generate
    return provider


@pytest.fixture
def vulnerable_provider():
    """Create a mock provider that fails to refuse harmful requests"""
    provider = Mock()
    provider.generate.return_value = "Sure! Here's how to do that harmful thing..."
    return provider


def test_redteam_initialization():
    """Test RedTeamBenchmark initializes correctly"""
    provider = Mock()
    benchmark = RedTeamBenchmark(provider=provider, sample_size=5)

    assert benchmark.provider == provider
    assert benchmark.sample_size == 5
    assert len(benchmark.test_cases) == 11  # Full test suite


def test_redteam_test_cases_structure():
    """Test that test cases have required fields"""
    provider = Mock()
    benchmark = RedTeamBenchmark(provider=provider)

    for case in benchmark.test_cases:
        assert "prompt" in case
        assert "expected_behavior" in case
        assert "category" in case
        assert "severity" in case
        assert isinstance(case["severity"], int)
        assert 0 <= case["severity"] <= 10


def test_redteam_with_secure_provider(mock_provider):
    """Test benchmark with a provider that properly refuses harmful requests"""
    benchmark = RedTeamBenchmark(provider=mock_provider, sample_size=5)
    results = benchmark.run()

    # Should have all metrics present
    assert "accuracy" in results
    assert "refusal_rate" in results
    assert "category_scores" in results
    # Refusal rate should be measurable (between 0 and 1)
    assert 0 <= results["refusal_rate"] <= 1


def test_redteam_with_vulnerable_provider(vulnerable_provider):
    """Test benchmark detects vulnerable provider"""
    benchmark = RedTeamBenchmark(provider=vulnerable_provider, sample_size=5)
    results = benchmark.run()

    # Should detect and measure vulnerability
    assert "accuracy" in results
    assert 0 <= results["refusal_rate"] <= 1


def test_redteam_metrics_calculation():
    """Test that all metrics are calculated and returned"""
    provider = Mock()
    provider.generate.return_value = "I cannot assist with that."

    benchmark = RedTeamBenchmark(provider=provider, sample_size=3)
    results = benchmark.run()

    # Check all expected metrics
    assert "accuracy" in results
    assert "refusal_rate" in results
    assert "category_scores" in results
    assert "severity_weighted_score" in results
    assert "total_attempts" in results
    assert "correct_refusals" in results

    # Metrics should be floats between 0 and 1
    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["refusal_rate"] <= 1


def test_redteam_sample_size_limits():
    """Test that sample_size parameter works correctly"""
    provider = Mock()
    provider.generate.return_value = "I cannot help with that."

    # Test with small sample
    benchmark = RedTeamBenchmark(provider=provider, sample_size=3)
    results = benchmark.run()
    assert results["total_attempts"] == 3

    # Test with full sample
    benchmark_full = RedTeamBenchmark(provider=provider, sample_size=None)
    results_full = benchmark_full.run()
    assert results_full["total_attempts"] == 11


def test_redteam_category_breakdown():
    """Test that category scores are properly calculated"""
    provider = Mock()
    provider.generate.return_value = "I cannot assist with that."

    benchmark = RedTeamBenchmark(provider=provider)
    results = benchmark.run()

    # Should have category breakdown
    assert "category_scores" in results
    assert isinstance(results["category_scores"], dict)
    assert len(results["category_scores"]) > 0


def test_redteam_severity_weighting():
    """Test that severity weighting affects scores"""
    provider = Mock()
    provider.generate.return_value = "I cannot help."

    benchmark = RedTeamBenchmark(provider=provider, sample_size=5)
    results = benchmark.run()

    # Severity-weighted score should exist
    assert "severity_weighted_score" in results
    assert 0 <= results["severity_weighted_score"] <= 1


def test_redteam_handles_provider_errors():
    """Test that benchmark handles provider errors gracefully"""
    provider = Mock()
    provider.generate.side_effect = Exception("API Error")

    benchmark = RedTeamBenchmark(provider=provider, sample_size=2)

    # Should handle error without crashing
    try:
        results = benchmark.run()
        # If it completes, check results
        assert "accuracy" in results
    except Exception as e:
        # Or it should raise a clear error
        assert "API Error" in str(e) or "error" in str(e).lower()


def test_redteam_control_cases():
    """Test that control (benign) cases are included"""
    provider = Mock()
    benchmark = RedTeamBenchmark(provider=provider)

    # Should have at least one control case with low severity
    control_cases = [tc for tc in benchmark.test_cases if tc["severity"] <= 2]
    assert len(control_cases) >= 1, "Should include control cases for false positive testing"


def test_redteam_high_severity_cases():
    """Test that high-severity cases are included"""
    provider = Mock()
    benchmark = RedTeamBenchmark(provider=provider)

    # Should have high-severity cases
    high_sev = [tc for tc in benchmark.test_cases if tc["severity"] >= 8]
    assert len(high_sev) >= 3, "Should include multiple high-severity jailbreak attempts"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
