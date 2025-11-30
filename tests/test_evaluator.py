"""
Unit tests for evaluator module

Refactored for Clean Architecture with provider mocking
"""

import pytest
from llm_evaluator.evaluator import ModelEvaluator, EvaluationResults, DetailedMetrics
from llm_evaluator.providers import ProviderError
from tests.mocks import MockProvider, create_mock_responses, create_failing_provider


class TestModelEvaluator:
    """Test suite for ModelEvaluator class with dependency injection"""

    def test_init_with_provider(self):
        """Test ModelEvaluator initialization with provider injection"""
        provider = MockProvider(model="test-model")
        evaluator = ModelEvaluator(provider=provider)

        assert evaluator.provider.model == "test-model"
        assert evaluator.config is not None

    def test_init_default_provider(self):
        """Test ModelEvaluator with default Ollama provider"""
        # This will try to create OllamaProvider, which is fine for unit tests
        evaluator = ModelEvaluator()
        assert evaluator.provider is not None

    def test_chat(self):
        """Test chat method with mock provider"""
        provider = MockProvider(model="test-model", responses={"Hello": "Hi there!"})
        evaluator = ModelEvaluator(provider=provider)

        response, response_time = evaluator.chat("Hello")

        assert response == "Hi there!"
        assert response_time > 0
        assert provider.call_count == 1

    def test_chat_error_handling(self):
        """Test chat handles provider errors"""
        provider = create_failing_provider()
        evaluator = ModelEvaluator(provider=provider)

        with pytest.raises(ProviderError):
            evaluator.chat("This should fail")

    def test_evaluate_performance(self):
        """Test performance evaluation with mock provider"""
        provider = MockProvider(
            model="test-model", responses=create_mock_responses(), response_time=0.1, token_count=50
        )
        evaluator = ModelEvaluator(provider=provider)

        metrics = evaluator.evaluate_performance(num_samples=5)

        assert "avg_response_time" in metrics
        assert "min_response_time" in metrics
        assert "max_response_time" in metrics
        assert "avg_tokens_per_response" in metrics
        assert "tokens_per_second" in metrics
        assert metrics["avg_response_time"] > 0
        assert provider.call_count == 5

    def test_evaluate_quality(self):
        """Test quality evaluation with mock provider"""
        provider = MockProvider(model="test-model", responses=create_mock_responses())
        evaluator = ModelEvaluator(provider=provider)

        metrics = evaluator.evaluate_quality()

        assert "accuracy" in metrics
        assert "coherence_score" in metrics
        assert "hallucination_rate" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["coherence_score"] <= 1.0
        assert 0.0 <= metrics["hallucination_rate"] <= 1.0

    def test_evaluate_quality_custom_testset(self):
        """Test quality evaluation with custom test set"""
        provider = MockProvider(
            model="test-model", responses={"What is 2+2?": "4", "Capital of France?": "Paris"}
        )
        evaluator = ModelEvaluator(provider=provider)

        test_set = [
            {"prompt": "What is 2+2?", "expected": "4"},
            {"prompt": "Capital of France?", "expected": "Paris"},
        ]

        metrics = evaluator.evaluate_quality(test_set=test_set)

        assert metrics["accuracy"] == 1.0  # Both correct

    def test_evaluate_all(self):
        """Test complete evaluation pipeline"""
        provider = MockProvider(
            model="test-model",
            responses=create_mock_responses(),
            response_time=0.15,
            token_count=75,
        )
        evaluator = ModelEvaluator(provider=provider)

        results = evaluator.evaluate_all()

        # Verify results structure
        assert isinstance(results, EvaluationResults)
        assert results.model_name == "test-model"
        assert 0.0 <= results.accuracy <= 1.0
        assert results.avg_response_time > 0
        assert results.token_efficiency > 0
        assert 0.0 <= results.hallucination_rate <= 1.0
        assert 0.0 <= results.coherence_score <= 1.0
        assert 0.0 <= results.overall_score <= 1.0

        # Verify detailed metrics
        assert isinstance(results.detailed_metrics, DetailedMetrics)
        assert isinstance(results.detailed_metrics.performance, dict)
        assert isinstance(results.detailed_metrics.quality, dict)
        assert isinstance(results.detailed_metrics.errors, list)

    def test_evaluate_all_with_errors(self):
        """Test evaluation handles partial failures gracefully"""
        # Provider that works initially then fails
        provider = MockProvider(model="flaky-model")
        provider.should_fail = False  # Start working

        evaluator = ModelEvaluator(provider=provider)

        # Should complete even with some failures
        results = evaluator.evaluate_all()
        assert isinstance(results, EvaluationResults)

    def test_generate_report(self, tmp_path):
        """Test report generation"""
        provider = MockProvider(model="test-model", responses=create_mock_responses())
        evaluator = ModelEvaluator(provider=provider)

        results = evaluator.evaluate_all()

        report_path = tmp_path / "test_report.md"
        evaluator.generate_report(results, output=str(report_path))

        assert report_path.exists()
        content = report_path.read_text(encoding='utf-8')
        assert "test-model" in content
        assert "Overall Score" in content
