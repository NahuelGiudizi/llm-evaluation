"""Unit tests for metrics module"""

import pytest
from llm_evaluator.metrics import PerformanceMetrics, QualityMetrics


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics class"""

    def test_calculate_avg_response_time(self):
        """Test average response time calculation"""
        times = [1.0, 1.5, 2.0, 1.2, 1.8]
        avg = PerformanceMetrics.calculate_avg_response_time(times)
        assert avg == pytest.approx(1.5, rel=0.01)

    def test_calculate_avg_response_time_empty(self):
        """Test with empty list"""
        with pytest.raises(ValueError, match="No response times provided"):
            PerformanceMetrics.calculate_avg_response_time([])

    def test_calculate_tokens_per_second(self):
        """Test tokens per second calculation"""
        tokens = 100
        time_seconds = 2.0
        tps = PerformanceMetrics.calculate_tokens_per_second(tokens, time_seconds)
        assert tps == 50.0

    def test_calculate_tokens_per_second_zero_time(self):
        """Test with zero time (edge case)"""
        with pytest.raises(ValueError, match="Time must be greater than 0"):
            PerformanceMetrics.calculate_tokens_per_second(100, 0)

    def test_calculate_throughput(self):
        """Test throughput calculation"""
        requests = 100
        time_seconds = 60.0
        throughput = PerformanceMetrics.calculate_throughput(requests, time_seconds)
        assert throughput == pytest.approx(1.67, rel=0.01)


class TestQualityMetrics:
    """Test suite for QualityMetrics class"""

    def test_calculate_accuracy(self):
        """Test accuracy calculation"""
        correct = 85
        total = 100
        accuracy = QualityMetrics.calculate_accuracy(correct, total)
        assert accuracy == 0.85

    def test_calculate_accuracy_zero_total(self):
        """Test with zero total (edge case)"""
        with pytest.raises(ValueError, match="Total must be greater than 0"):
            QualityMetrics.calculate_accuracy(10, 0)

    def test_calculate_coherence_score(self):
        """Test coherence score calculation"""
        # Mock responses with varying coherence
        responses = [
            "This is a coherent response about the topic.",
            "Another well-structured answer.",
            "A clear and logical explanation.",
        ]
        score = QualityMetrics.calculate_coherence_score(responses)
        assert 0.0 <= score <= 1.0

    def test_calculate_hallucination_rate(self):
        """Test hallucination rate calculation"""
        hallucinations = 5
        total = 100
        rate = QualityMetrics.calculate_hallucination_rate(hallucinations, total)
        assert rate == 0.05

    def test_calculate_hallucination_rate_bounds(self):
        """Test hallucination rate is within bounds"""
        rate = QualityMetrics.calculate_hallucination_rate(10, 50)
        assert 0.0 <= rate <= 1.0
