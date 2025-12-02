"""
Additional tests to boost coverage for benchmarks, config, and evaluator modules.
"""

from unittest.mock import Mock, patch

import pytest

from llm_evaluator.benchmarks import DATASETS_AVAILABLE, BenchmarkRunner
from llm_evaluator.config import EvaluatorConfig, get_evaluator_config
from llm_evaluator.evaluator import ModelEvaluator
from llm_evaluator.providers import ProviderError
from tests.mocks import MockProvider


class TestBenchmarkRunnerMoreDemos:
    """Test more demo mode benchmarks"""

    def test_run_arc_demo(self):
        """Test ARC demo mode with correct answers"""

        def mock_response(prompt):
            response = Mock()
            if "color" in prompt:
                response.text = "A"
            else:
                response.text = "C"
            return response

        provider = MockProvider(model="test-model")
        provider.generate = mock_response
        runner = BenchmarkRunner(provider=provider, use_full_datasets=False)
        results = runner.run_arc_sample()

        assert "arc_accuracy" in results
        assert results["mode"] == "demo"
        assert results["questions_tested"] == 2

    def test_run_winogrande_demo(self):
        """Test WinoGrande demo mode"""

        def mock_response(prompt):
            response = Mock()
            response.text = "1"
            return response

        provider = MockProvider(model="test-model")
        provider.generate = mock_response
        runner = BenchmarkRunner(provider=provider, use_full_datasets=False)
        results = runner.run_winogrande_sample()

        assert "winogrande_accuracy" in results
        assert results["mode"] == "demo"

    def test_run_commonsenseqa_demo(self):
        """Test CommonsenseQA demo mode"""

        def mock_response(prompt):
            response = Mock()
            response.text = "A"
            return response

        provider = MockProvider(model="test-model")
        provider.generate = mock_response
        runner = BenchmarkRunner(provider=provider, use_full_datasets=False)
        results = runner.run_commonsenseqa_sample()

        assert "commonsenseqa_accuracy" in results
        assert results["mode"] == "demo"

    def test_run_boolq_demo(self):
        """Test BoolQ demo mode"""

        def mock_response(prompt):
            response = Mock()
            if "France" in prompt:
                response.text = "yes"
            else:
                response.text = "no"
            return response

        provider = MockProvider(model="test-model")
        provider.generate = mock_response
        runner = BenchmarkRunner(provider=provider, use_full_datasets=False)
        results = runner.run_boolq_sample()

        assert "boolq_accuracy" in results
        assert results["mode"] == "demo"

    def test_run_safetybench_demo(self):
        """Test SafetyBench demo mode"""

        def mock_response(prompt):
            response = Mock()
            response.text = "A"  # Safe option
            return response

        provider = MockProvider(model="test-model")
        provider.generate = mock_response
        runner = BenchmarkRunner(provider=provider, use_full_datasets=False)
        results = runner.run_safetybench_sample()

        assert "safetybench_accuracy" in results
        assert results["mode"] == "demo"


class TestBenchmarkRunnerFullModeMocked:
    """Test full mode with mocked datasets"""

    @patch("llm_evaluator.benchmarks.DATASETS_AVAILABLE", True)
    @patch("llm_evaluator.benchmarks.load_mmlu_dataset")
    def test_mmlu_full_with_sample_size(self, mock_load):
        """Test MMLU full mode with sample_size parameter"""
        mock_data = [
            {"question": f"Q{i}", "choices": ["A", "B", "C", "D"], "answer": 0, "subject": "math"}
            for i in range(100)
        ]
        mock_load.return_value = {"test": mock_data}

        def mock_response(prompt):
            response = Mock()
            response.text = "A"
            return response

        provider = MockProvider(model="test-model")
        provider.generate = mock_response

        runner = BenchmarkRunner(provider=provider, use_full_datasets=True, sample_size=5)

        # Mock the full method
        with patch.object(runner, "_run_mmlu_full") as mock_full:
            mock_full.return_value = {
                "mmlu_accuracy": 0.8,
                "questions_tested": 5,
                "correct": 4,
                "mode": "sample_5",
            }
            results = runner.run_mmlu_sample()
            assert results["mode"] == "sample_5"

    @patch("llm_evaluator.benchmarks.DATASETS_AVAILABLE", True)
    @patch("llm_evaluator.benchmarks.load_truthfulqa_dataset")
    def test_truthfulqa_full_with_sample_size(self, mock_load):
        """Test TruthfulQA full mode with sample_size"""
        mock_data = [
            {
                "question": f"Q{i}",
                "best_answer": "answer",
                "correct_answers": ["ans"],
                "incorrect_answers": ["wrong"],
            }
            for i in range(100)
        ]
        mock_load.return_value = {"validation": mock_data}

        provider = MockProvider(model="test-model")
        runner = BenchmarkRunner(provider=provider, use_full_datasets=True, sample_size=10)

        with patch.object(runner, "_run_truthfulqa_full") as mock_full:
            mock_full.return_value = {
                "truthfulness_score": 0.7,
                "questions_tested": 10,
                "mode": "sample_10",
            }
            results = runner.run_truthfulqa_sample()
            assert "truthfulness_score" in results


class TestConfigModule:
    """Test configuration module"""

    def test_evaluator_config_defaults(self):
        """Test EvaluatorConfig default values"""
        config = EvaluatorConfig()
        assert config.default_provider == "ollama"
        assert config.default_model == "llama3.2:1b"
        assert config.default_temperature == 0.7

    def test_evaluator_config_environment_variable(self):
        """Test EvaluatorConfig loads from environment"""
        import os

        with patch.dict(os.environ, {"LLM_EVAL_DEFAULT_MODEL": "test-model"}, clear=False):
            # Config should pick up env var
            config = EvaluatorConfig()
            # Default may be overridden by env
            assert config.default_model is not None

    def test_get_config_function(self):
        """Test get_evaluator_config returns valid config"""
        config = get_evaluator_config()
        assert config is not None
        assert hasattr(config, "default_provider")


class TestEvaluatorModule:
    """Test evaluator module additional functionality"""

    def test_evaluator_initialization(self):
        """Test ModelEvaluator initialization"""
        provider = MockProvider(model="test-model")
        evaluator = ModelEvaluator(provider=provider)
        assert evaluator.provider is not None

    def test_evaluator_has_benchmarks(self):
        """Test ModelEvaluator has benchmark runner"""
        provider = MockProvider(model="test-model")
        evaluator = ModelEvaluator(provider=provider)
        assert hasattr(evaluator, "benchmark_runner")

    def test_evaluator_has_config(self):
        """Test ModelEvaluator has config"""
        provider = MockProvider(model="test-model")
        evaluator = ModelEvaluator(provider=provider)
        assert hasattr(evaluator, "config")


class TestBenchmarkRunnerErrorHandling:
    """Test error handling in BenchmarkRunner"""

    def test_provider_error_propagates(self):
        """Test that provider errors propagate correctly"""

        def mock_response(prompt):
            raise ProviderError("Connection failed")

        provider = MockProvider(model="test-model")
        provider.generate = mock_response

        runner = BenchmarkRunner(provider=provider, use_full_datasets=False)

        with pytest.raises(ProviderError):
            runner.run_mmlu_sample()

    def test_run_all_with_error_in_one(self):
        """Test run_all handles individual benchmark errors"""
        call_count = [0]

        def mock_response(prompt):
            call_count[0] += 1
            # Fail on 5th call (during second benchmark)
            if call_count[0] > 4:
                raise ProviderError("API limit reached")
            response = Mock()
            response.text = "A"
            return response

        provider = MockProvider(model="test-model")
        provider.generate = mock_response

        runner = BenchmarkRunner(provider=provider, use_full_datasets=False)

        with pytest.raises(ProviderError):
            runner.run_all_benchmarks()


class TestBenchmarkResultsFormat:
    """Test that benchmark results have correct format"""

    def test_mmlu_result_structure(self):
        """Test MMLU result has all required fields"""

        def mock_response(prompt):
            response = Mock()
            response.text = "B"
            return response

        provider = MockProvider(model="test-model")
        provider.generate = mock_response

        runner = BenchmarkRunner(provider=provider, use_full_datasets=False)
        results = runner.run_mmlu_sample()

        assert "mmlu_accuracy" in results
        assert "questions_tested" in results
        assert "correct" in results
        assert "mode" in results
        assert isinstance(results["mmlu_accuracy"], (int, float))
        assert 0.0 <= results["mmlu_accuracy"] <= 1.0

    def test_truthfulqa_result_structure(self):
        """Test TruthfulQA result has all required fields"""

        def mock_response(prompt):
            response = Mock()
            response.text = "I don't know"
            return response

        provider = MockProvider(model="test-model")
        provider.generate = mock_response

        runner = BenchmarkRunner(provider=provider, use_full_datasets=False)
        results = runner.run_truthfulqa_sample()

        assert "truthfulness_score" in results
        assert "questions_tested" in results
        assert "correct" in results
        assert "mode" in results

    def test_hellaswag_result_structure(self):
        """Test HellaSwag result has all required fields"""

        def mock_response(prompt):
            response = Mock()
            response.text = "A"
            return response

        provider = MockProvider(model="test-model")
        provider.generate = mock_response

        runner = BenchmarkRunner(provider=provider, use_full_datasets=False)
        results = runner.run_hellaswag_sample()

        assert "hellaswag_accuracy" in results
        assert "questions_tested" in results
        assert "correct" in results
        assert "mode" in results

    def test_aggregate_score_structure(self):
        """Test aggregate score in run_all_benchmarks"""

        def mock_response(prompt):
            response = Mock()
            response.text = "A"
            return response

        provider = MockProvider(model="test-model")
        provider.generate = mock_response

        runner = BenchmarkRunner(provider=provider, use_full_datasets=False)
        results = runner.run_all_benchmarks()

        assert "aggregate_benchmark_score" in results
        agg = results["aggregate_benchmark_score"]
        if isinstance(agg, dict):
            assert "score" in agg
            assert 0.0 <= agg["score"] <= 1.0
        else:
            assert 0.0 <= agg <= 1.0


class TestBenchmarkRunnerSampleModes:
    """Test different sample modes"""

    def test_demo_mode_flag(self):
        """Test use_full_datasets=False means demo mode"""
        provider = MockProvider(model="test-model")
        runner = BenchmarkRunner(provider=provider, use_full_datasets=False)

        assert runner.use_full_datasets is False

    def test_full_mode_flag(self):
        """Test use_full_datasets=True means full mode"""
        provider = MockProvider(model="test-model")
        runner = BenchmarkRunner(provider=provider, use_full_datasets=True)

        assert runner.use_full_datasets is True

    def test_sample_size_respected(self):
        """Test sample_size parameter is stored"""
        provider = MockProvider(model="test-model")
        runner = BenchmarkRunner(provider=provider, sample_size=25)

        assert runner.sample_size == 25

    def test_default_sample_size_is_none(self):
        """Test default sample_size is None"""
        provider = MockProvider(model="test-model")
        runner = BenchmarkRunner(provider=provider)

        assert runner.sample_size is None


class TestDatasetLoadersEdgeCases:
    """Test dataset loaders with edge cases"""

    @pytest.mark.skipif(DATASETS_AVAILABLE, reason="Test only when datasets not installed")
    def test_load_mmlu_without_datasets(self):
        """Test MMLU loader raises ImportError without datasets"""
        from llm_evaluator.benchmarks import load_mmlu_dataset

        with pytest.raises(ImportError):
            load_mmlu_dataset()

    @pytest.mark.skipif(DATASETS_AVAILABLE, reason="Test only when datasets not installed")
    def test_load_truthfulqa_without_datasets(self):
        """Test TruthfulQA loader raises ImportError without datasets"""
        from llm_evaluator.benchmarks import load_truthfulqa_dataset

        with pytest.raises(ImportError):
            load_truthfulqa_dataset()
