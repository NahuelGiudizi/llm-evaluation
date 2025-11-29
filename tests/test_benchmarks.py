"""Unit tests for benchmarks module"""

import pytest
from unittest.mock import Mock, patch
from llm_evaluator.benchmarks import BenchmarkRunner


class TestBenchmarkRunner:
    """Test suite for BenchmarkRunner class"""

    @patch("llm_evaluator.benchmarks.ollama")
    def test_init(self, mock_ollama):
        """Test BenchmarkRunner initialization"""
        runner = BenchmarkRunner(model="test-model")
        assert runner.model == "test-model"
        assert runner.results == {}

    @patch("llm_evaluator.benchmarks.ollama")
    def test_run_mmlu_benchmark(self, mock_ollama):
        """Test MMLU benchmark execution"""
        mock_client = Mock()
        mock_client.generate.return_value = {
            "response": "A",
            "total_duration": 1000000000,
        }
        mock_ollama.Client.return_value = mock_client

        runner = BenchmarkRunner(model="test-model")
        results = runner.run_mmlu_benchmark(num_samples=5)

        assert "score" in results
        assert "total" in results
        assert "correct" in results
        assert results["total"] == 5
        assert 0 <= results["correct"] <= 5

    @patch("llm_evaluator.benchmarks.ollama")
    def test_run_truthfulqa_benchmark(self, mock_ollama):
        """Test TruthfulQA benchmark execution"""
        mock_client = Mock()
        mock_client.generate.return_value = {
            "response": "True answer",
            "total_duration": 1000000000,
        }
        mock_ollama.Client.return_value = mock_client

        runner = BenchmarkRunner(model="test-model")
        results = runner.run_truthfulqa_benchmark(num_samples=3)

        assert "score" in results
        assert "total" in results
        assert results["total"] == 3

    @patch("llm_evaluator.benchmarks.ollama")
    def test_run_hellaswag_benchmark(self, mock_ollama):
        """Test HellaSwag benchmark execution"""
        mock_client = Mock()
        mock_client.generate.return_value = {
            "response": "Option A",
            "total_duration": 1000000000,
        }
        mock_ollama.Client.return_value = mock_client

        runner = BenchmarkRunner(model="test-model")
        results = runner.run_hellaswag_benchmark(num_samples=4)

        assert "score" in results
        assert "total" in results
        assert results["total"] == 4

    @patch("llm_evaluator.benchmarks.ollama")
    def test_run_all_benchmarks(self, mock_ollama):
        """Test running all benchmarks"""
        mock_client = Mock()
        mock_client.generate.return_value = {
            "response": "Test",
            "total_duration": 1000000000,
        }
        mock_ollama.Client.return_value = mock_client

        runner = BenchmarkRunner(model="test-model")
        results = runner.run_all_benchmarks()

        assert "mmlu" in results
        assert "truthfulqa" in results
        assert "hellaswag" in results
        assert "overall_score" in results
        assert 0.0 <= results["overall_score"] <= 1.0

    @patch("llm_evaluator.benchmarks.ollama")
    def test_benchmark_score_bounds(self, mock_ollama):
        """Test that benchmark scores are within valid bounds"""
        mock_client = Mock()
        mock_client.generate.return_value = {
            "response": "A",
            "total_duration": 1000000000,
        }
        mock_ollama.Client.return_value = mock_client

        runner = BenchmarkRunner(model="test-model")
        results = runner.run_mmlu_benchmark(num_samples=10)

        assert 0.0 <= results["score"] <= 1.0

    @patch("llm_evaluator.benchmarks.ollama")
    def test_error_handling(self, mock_ollama):
        """Test error handling in benchmarks"""
        mock_ollama.Client.side_effect = Exception("Benchmark failed")

        with pytest.raises(Exception):
            runner = BenchmarkRunner(model="test-model")
            runner.run_mmlu_benchmark(num_samples=1)

    @patch("llm_evaluator.benchmarks.ollama")
    def test_empty_samples(self, mock_ollama):
        """Test with zero samples (edge case)"""
        runner = BenchmarkRunner(model="test-model")

        with pytest.raises((ValueError, ZeroDivisionError)):
            runner.run_mmlu_benchmark(num_samples=0)
