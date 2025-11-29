"""Unit tests for evaluator module"""

import pytest
from unittest.mock import Mock, patch
from llm_evaluator.evaluator import ModelEvaluator


class TestModelEvaluator:
    """Test suite for ModelEvaluator class"""

    @patch("llm_evaluator.evaluator.ollama")
    def test_init(self, mock_ollama):
        """Test ModelEvaluator initialization"""
        evaluator = ModelEvaluator(model="test-model")
        assert evaluator.model == "test-model"
        assert evaluator.results == {}

    @patch("llm_evaluator.evaluator.ollama")
    def test_evaluate_response_time(self, mock_ollama, mock_ollama_response):
        """Test response time evaluation"""
        mock_ollama.Client.return_value = mock_ollama_response

        evaluator = ModelEvaluator(model="test-model")
        metrics = evaluator.evaluate_response_time(num_requests=5)

        assert "avg_response_time" in metrics
        assert "min_response_time" in metrics
        assert "max_response_time" in metrics
        assert metrics["total_requests"] == 5

    @patch("llm_evaluator.evaluator.ollama")
    def test_evaluate_token_efficiency(self, mock_ollama, mock_ollama_response):
        """Test token efficiency evaluation"""
        mock_ollama.Client.return_value = mock_ollama_response

        evaluator = ModelEvaluator(model="test-model")
        metrics = evaluator.evaluate_token_efficiency(num_requests=3)

        assert "tokens_per_second" in metrics
        assert "avg_tokens_per_response" in metrics
        assert metrics["tokens_per_second"] > 0

    @patch("llm_evaluator.evaluator.ollama")
    def test_evaluate_accuracy(self, mock_ollama):
        """Test accuracy evaluation with known Q&A pairs"""
        mock_client = Mock()
        mock_client.generate.return_value = {
            "response": "Paris",
            "total_duration": 1000000000,
            "eval_count": 10,
        }
        mock_ollama.Client.return_value = mock_client

        evaluator = ModelEvaluator(model="test-model")
        qa_pairs = [
            ("What is the capital of France?", "Paris"),
            ("What is 2+2?", "4"),
        ]
        metrics = evaluator.evaluate_accuracy(qa_pairs)

        assert "accuracy" in metrics
        assert "total_questions" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    @patch("llm_evaluator.evaluator.ollama")
    def test_evaluate_coherence(self, mock_ollama, mock_ollama_response):
        """Test coherence evaluation"""
        mock_ollama.Client.return_value = mock_ollama_response

        evaluator = ModelEvaluator(model="test-model")
        prompts = ["Explain AI", "What is ML?", "Define deep learning"]
        metrics = evaluator.evaluate_coherence(prompts)

        assert "coherence_score" in metrics
        assert 0.0 <= metrics["coherence_score"] <= 1.0

    @patch("llm_evaluator.evaluator.ollama")
    def test_evaluate_all(self, mock_ollama, mock_ollama_response):
        """Test complete evaluation pipeline"""
        mock_ollama.Client.return_value = mock_ollama_response

        evaluator = ModelEvaluator(model="test-model")
        results = evaluator.evaluate_all()

        assert "model" in results
        assert "performance" in results
        assert "quality" in results
        assert results["model"] == "test-model"

    @patch("llm_evaluator.evaluator.ollama")
    def test_generate_report(self, mock_ollama, sample_evaluation_results):
        """Test report generation"""
        evaluator = ModelEvaluator(model="test-model")
        evaluator.results = sample_evaluation_results

        report = evaluator.generate_report()

        assert isinstance(report, str)
        assert "test-model" in report
        assert "Performance Metrics" in report or "performance" in report.lower()

    @patch("llm_evaluator.evaluator.ollama")
    def test_error_handling_connection(self, mock_ollama):
        """Test error handling for connection issues"""
        mock_ollama.Client.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            evaluator = ModelEvaluator(model="test-model")
            evaluator.evaluate_response_time(num_requests=1)

    @patch("llm_evaluator.evaluator.ollama")
    def test_empty_results(self, mock_ollama):
        """Test handling of empty results"""
        evaluator = ModelEvaluator(model="test-model")

        # Should not raise exception with empty results
        report = evaluator.generate_report()
        assert isinstance(report, str)
