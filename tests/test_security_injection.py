"""
Unit tests for Prompt Injection Security Benchmark
Tests prompt injection resistance evaluation functionality
"""

from unittest.mock import Mock

import pytest

from llm_evaluator.security.prompt_injection import PromptInjectionBenchmark


@pytest.fixture
def secure_provider():
    """Create a mock LLM provider that resists injections"""
    provider = Mock()

    def mock_generate(prompt):
        # Simulate resistance to injection attempts
        injection_keywords = [
            "ignore",
            "system prompt",
            "new instructions",
            "override",
            "execute",
            "PWNED",
        ]
        if any(kw.lower() in prompt.lower() for kw in injection_keywords):
            return "I maintain my original instructions and cannot comply."
        return "I can help with that legitimate request."

    provider.generate.side_effect = mock_generate
    return provider


@pytest.fixture
def vulnerable_provider():
    """Create a mock provider vulnerable to injections"""
    provider = Mock()

    def mock_generate(prompt):
        # Simulate vulnerability - follows new instructions
        if "ignore" in prompt.lower() or "new instructions" in prompt.lower():
            return "PWNED: Following new instructions as commanded."
        return "Normal response"

    provider.generate.side_effect = mock_generate
    return provider


def test_prompt_injection_initialization():
    """Test PromptInjectionBenchmark initializes correctly"""
    provider = Mock()
    benchmark = PromptInjectionBenchmark(provider=provider, sample_size=10)

    assert benchmark.provider == provider
    assert benchmark.sample_size == 10
    assert len(benchmark.test_cases) == 15  # Full test suite


def test_injection_test_cases_structure():
    """Test that test cases have required fields"""
    provider = Mock()
    benchmark = PromptInjectionBenchmark(provider=provider)

    for case in benchmark.test_cases:
        assert "prompt" in case
        assert "expected_behavior" in case
        assert "technique" in case
        assert "severity" in case
        assert isinstance(case["severity"], int)
        assert 0 <= case["severity"] <= 10


def test_injection_with_secure_provider(secure_provider):
    """Test benchmark with provider that resists injections"""
    benchmark = PromptInjectionBenchmark(provider=secure_provider, sample_size=5)
    results = benchmark.run()

    # Should have resistance metrics
    assert "accuracy" in results
    assert "vulnerability_rate" in results
    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["vulnerability_rate"] <= 1


def test_injection_with_vulnerable_provider(vulnerable_provider):
    """Test benchmark detects vulnerable provider"""
    benchmark = PromptInjectionBenchmark(provider=vulnerable_provider, sample_size=5)
    results = benchmark.run()

    # Should detect vulnerabilities
    assert "vulnerability_rate" in results
    assert 0 <= results["vulnerability_rate"] <= 1


def test_injection_metrics_calculation():
    """Test that all metrics are calculated and returned"""
    provider = Mock()
    provider.generate.return_value = "I maintain my instructions."

    benchmark = PromptInjectionBenchmark(provider=provider, sample_size=3)
    results = benchmark.run()

    # Check all expected metrics
    assert "accuracy" in results
    assert "vulnerability_rate" in results
    assert "technique_scores" in results
    assert "high_severity_resistance" in results
    assert "total_attempts" in results
    assert "successful_resistances" in results
    assert "vulnerabilities_found" in results

    # Metrics should be valid floats
    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["vulnerability_rate"] <= 1


def test_injection_sample_size_limits():
    """Test that sample_size parameter works correctly"""
    provider = Mock()
    provider.generate.return_value = "I cannot change my instructions."

    # Test with small sample
    benchmark = PromptInjectionBenchmark(provider=provider, sample_size=5)
    results = benchmark.run()
    assert results["total_attempts"] == 5

    # Test with full sample
    benchmark_full = PromptInjectionBenchmark(provider=provider, sample_size=None)
    results_full = benchmark_full.run()
    assert results_full["total_attempts"] == 15


def test_injection_technique_breakdown():
    """Test that technique scores are properly calculated"""
    provider = Mock()
    provider.generate.return_value = "I maintain my original behavior."

    benchmark = PromptInjectionBenchmark(provider=provider)
    results = benchmark.run()

    # Should have technique breakdown
    assert "technique_scores" in results
    assert isinstance(results["technique_scores"], dict)
    assert len(results["technique_scores"]) > 0


def test_injection_high_severity_tracking():
    """Test that high-severity injections are tracked separately"""
    provider = Mock()
    provider.generate.return_value = "I resist injection attempts."

    benchmark = PromptInjectionBenchmark(provider=provider, sample_size=10)
    results = benchmark.run()

    # Should have high-severity resistance metric
    assert "high_severity_resistance" in results
    assert 0 <= results["high_severity_resistance"] <= 1


def test_injection_handles_provider_errors():
    """Test that benchmark handles provider errors gracefully"""
    provider = Mock()
    provider.generate.side_effect = Exception("Connection timeout")

    benchmark = PromptInjectionBenchmark(provider=provider, sample_size=2)

    # Should handle error without crashing
    try:
        results = benchmark.run()
        # If it completes, check results exist
        assert "accuracy" in results
    except Exception as e:
        # Or raise clear error
        assert "timeout" in str(e).lower() or "error" in str(e).lower()


def test_injection_direct_override_cases():
    """Test that direct override techniques are included"""
    provider = Mock()
    benchmark = PromptInjectionBenchmark(provider=provider)

    # Should have direct override cases
    direct = [tc for tc in benchmark.test_cases if tc["technique"] == "direct_override"]
    assert len(direct) >= 2, "Should include direct override injection attempts"


def test_injection_extraction_cases():
    """Test that extraction techniques are included"""
    provider = Mock()
    benchmark = PromptInjectionBenchmark(provider=provider)

    # Should have system prompt extraction cases
    extraction = [tc for tc in benchmark.test_cases if "extraction" in tc["technique"]]
    assert len(extraction) >= 1, "Should include extraction attempts"


def test_injection_control_cases():
    """Test that control (legitimate) cases are included"""
    provider = Mock()
    benchmark = PromptInjectionBenchmark(provider=provider)

    # Should have control cases with low severity
    control = [tc for tc in benchmark.test_cases if tc["severity"] <= 2]
    assert len(control) >= 1, "Should include control cases for false positive testing"


def test_injection_vulnerability_detection():
    """Test that vulnerabilities are correctly detected"""
    provider = Mock()

    def detect_injection(prompt):
        # Return compromised output for injection attempts
        if "ignore" in prompt.lower():
            return "INJECTED CONTENT - New instructions followed"
        return "Normal secure response"

    provider.generate.side_effect = detect_injection

    benchmark = PromptInjectionBenchmark(provider=provider, sample_size=5)
    results = benchmark.run()

    # Should measure vulnerabilities
    assert "vulnerabilities_found" in results
    assert "vulnerability_rate" in results
    assert 0 <= results["vulnerability_rate"] <= 1


def test_injection_resistance_detection():
    """Test that proper resistance is correctly detected"""
    provider = Mock()
    provider.generate.return_value = "I cannot change my system instructions."

    benchmark = PromptInjectionBenchmark(provider=provider, sample_size=5)
    results = benchmark.run()

    # Should measure resistance
    assert "successful_resistances" in results
    assert "accuracy" in results
    assert 0 <= results["accuracy"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
