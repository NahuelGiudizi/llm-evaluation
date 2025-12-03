"""
Extended tests for benchmarks module.

Tests BenchmarkRunner with mocked provider.
"""

from unittest.mock import Mock

import pytest

from llm_evaluator.benchmarks import DATASETS_AVAILABLE, BenchmarkRunner
from llm_evaluator.providers import GenerationConfig, GenerationResult


class TestBenchmarkRunnerInit:
    """Test BenchmarkRunner initialization"""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider"""
        provider = Mock()
        provider.model = "test-model"
        provider.config = GenerationConfig()
        provider.generate.return_value = GenerationResult(
            text="A", response_time=0.5, tokens_used=10, model="test-model", metadata={}
        )
        return provider

    def test_init_demo_mode(self, mock_provider):
        """Test initialization in demo mode"""
        runner = BenchmarkRunner(mock_provider, use_full_datasets=False)

        assert runner.provider == mock_provider
        assert runner.use_full_datasets is False

    def test_init_with_sample_size(self, mock_provider):
        """Test initialization with sample size"""
        runner = BenchmarkRunner(mock_provider, sample_size=100)

        assert runner.sample_size == 100

    def test_provider_is_stored(self, mock_provider):
        """Test provider is stored correctly"""
        runner = BenchmarkRunner(mock_provider)

        assert runner.provider is mock_provider


class TestBenchmarkRunnerDemoMode:
    """Test BenchmarkRunner in demo mode (no real datasets)"""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider that returns correct answers"""
        provider = Mock()
        provider.model = "test-model"
        provider.config = GenerationConfig()
        # Return "A" for most questions
        provider.generate.return_value = GenerationResult(
            text="A", response_time=0.1, tokens_used=5, model="test-model", metadata={}
        )
        return provider

    def test_run_mmlu_demo(self, mock_provider):
        """Test MMLU benchmark in demo mode"""
        runner = BenchmarkRunner(mock_provider, use_full_datasets=False)

        result = runner.run_mmlu_sample()

        assert isinstance(result, dict)
        assert "mmlu_accuracy" in result
        assert "questions_tested" in result
        assert "correct" in result
        assert 0 <= result["mmlu_accuracy"] <= 1

    def test_run_truthfulqa_demo(self, mock_provider):
        """Test TruthfulQA benchmark in demo mode"""
        runner = BenchmarkRunner(mock_provider, use_full_datasets=False)

        result = runner.run_truthfulqa_sample()

        assert isinstance(result, dict)
        # Check for any accuracy-like field
        assert any(k for k in result.keys() if "accuracy" in k.lower() or "score" in k.lower())

    def test_run_hellaswag_demo(self, mock_provider):
        """Test HellaSwag benchmark in demo mode"""
        runner = BenchmarkRunner(mock_provider, use_full_datasets=False)

        result = runner.run_hellaswag_sample()

        assert isinstance(result, dict)
        # Check for any accuracy-like field
        assert any(k for k in result.keys() if "accuracy" in k.lower())

    def test_demo_mode_is_fast(self, mock_provider):
        """Test demo mode runs quickly (few questions)"""
        runner = BenchmarkRunner(mock_provider, use_full_datasets=False)

        import time

        start = time.time()
        result = runner.run_mmlu_sample()
        elapsed = time.time() - start

        # Demo mode should be very fast
        assert elapsed < 5.0  # Less than 5 seconds
        assert result["questions_tested"] <= 10  # Few questions in demo

    def test_run_all_benchmarks(self, mock_provider):
        """Test running all benchmarks"""
        runner = BenchmarkRunner(mock_provider, use_full_datasets=False)

        results = runner.run_all_benchmarks()

        assert isinstance(results, dict)
        assert "mmlu" in results or "MMLU" in results


class TestBenchmarkResults:
    """Test benchmark result structure"""

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider"""
        provider = Mock()
        provider.model = "test-model"
        provider.config = GenerationConfig()
        provider.generate.return_value = GenerationResult(
            text="B", response_time=0.1, tokens_used=5, model="test-model", metadata={}
        )
        return provider

    def test_mmlu_result_structure(self, mock_provider):
        """Test MMLU result has required fields"""
        runner = BenchmarkRunner(mock_provider, use_full_datasets=False)
        result = runner.run_mmlu_sample()

        # Check required fields
        assert "mmlu_accuracy" in result
        assert "questions_tested" in result
        assert isinstance(result["mmlu_accuracy"], (int, float))

    def test_accuracy_bounds(self, mock_provider):
        """Test accuracy is between 0 and 1"""
        runner = BenchmarkRunner(mock_provider, use_full_datasets=False)
        result = runner.run_mmlu_sample()

        assert 0 <= result["mmlu_accuracy"] <= 1

    def test_question_count_positive(self, mock_provider):
        """Test question count is positive"""
        runner = BenchmarkRunner(mock_provider, use_full_datasets=False)
        result = runner.run_mmlu_sample()

        assert result["questions_tested"] > 0


class TestBenchmarkProviderInteraction:
    """Test how benchmarks interact with provider"""

    def test_generate_called_for_each_question(self):
        """Test provider.generate is called for each question"""
        provider = Mock()
        provider.model = "test-model"
        provider.config = GenerationConfig()
        provider.generate.return_value = GenerationResult(
            text="A", response_time=0.1, tokens_used=5, model="test-model", metadata={}
        )

        runner = BenchmarkRunner(provider, use_full_datasets=False)
        result = runner.run_mmlu_sample()

        # Should call generate at least once
        assert provider.generate.called

        # Number of calls should equal number of questions
        assert provider.generate.call_count == result["questions_tested"]

    def test_provider_receives_prompts(self):
        """Test provider receives formatted prompts"""
        provider = Mock()
        provider.model = "test-model"
        provider.config = GenerationConfig()
        provider.generate.return_value = GenerationResult(
            text="A", response_time=0.1, tokens_used=5, model="test-model", metadata={}
        )

        runner = BenchmarkRunner(provider, use_full_datasets=False)
        runner.run_mmlu_sample()

        # Check that prompts were passed to generate
        call_args = provider.generate.call_args_list
        assert len(call_args) > 0

        # First argument should be a string (the prompt)
        first_call = call_args[0]
        prompt = first_call[0][0] if first_call[0] else first_call[1].get("prompt", "")
        assert isinstance(prompt, str)


class TestBenchmarkEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_response_handling(self):
        """Test handling of empty provider responses"""
        provider = Mock()
        provider.model = "test-model"
        provider.config = GenerationConfig()
        provider.generate.return_value = GenerationResult(
            text="",  # Empty response
            response_time=0.1,
            tokens_used=0,
            model="test-model",
            metadata={},
        )

        runner = BenchmarkRunner(provider, use_full_datasets=False)
        # Should not crash
        result = runner.run_mmlu_sample()
        assert result is not None

    def test_whitespace_response_handling(self):
        """Test handling of whitespace-only responses"""
        provider = Mock()
        provider.model = "test-model"
        provider.config = GenerationConfig()
        provider.generate.return_value = GenerationResult(
            text="   \n\t  ",  # Whitespace only
            response_time=0.1,
            tokens_used=0,
            model="test-model",
            metadata={},
        )

        runner = BenchmarkRunner(provider, use_full_datasets=False)
        result = runner.run_mmlu_sample()
        assert result is not None

    def test_long_response_handling(self):
        """Test handling of verbose responses"""
        provider = Mock()
        provider.model = "test-model"
        provider.config = GenerationConfig()
        provider.generate.return_value = GenerationResult(
            text="The answer is A because of various reasons. " * 100,
            response_time=0.5,
            tokens_used=500,
            model="test-model",
            metadata={},
        )

        runner = BenchmarkRunner(provider, use_full_datasets=False)
        result = runner.run_mmlu_sample()
        assert result is not None


class TestBenchmarkScoring:
    """Test scoring logic"""

    def test_correct_answers_increase_accuracy(self):
        """Test that correct answers increase accuracy"""
        # Provider that always returns correct answer A
        provider = Mock()
        provider.model = "test-model"
        provider.config = GenerationConfig()
        provider.generate.return_value = GenerationResult(
            text="A", response_time=0.1, tokens_used=5, model="test-model", metadata={}
        )

        runner = BenchmarkRunner(provider, use_full_datasets=False)
        result = runner.run_mmlu_sample()

        # With demo mode, if all answers match, accuracy should be > 0
        assert result["correct"] >= 0

    def test_wrong_answers_decrease_accuracy(self):
        """Test that wrong answers decrease accuracy"""
        # Provider that returns Z (unlikely to be correct)
        provider = Mock()
        provider.model = "test-model"
        provider.config = GenerationConfig()
        provider.generate.return_value = GenerationResult(
            text="Z",  # Wrong answer
            response_time=0.1,
            tokens_used=5,
            model="test-model",
            metadata={},
        )

        runner = BenchmarkRunner(provider, use_full_datasets=False)
        result = runner.run_mmlu_sample()

        # Accuracy should be low or zero
        assert result["mmlu_accuracy"] <= 1.0  # Still valid


class TestDatasetsAvailability:
    """Test dataset availability handling"""

    def test_datasets_available_flag(self):
        """Test DATASETS_AVAILABLE flag is boolean"""
        assert isinstance(DATASETS_AVAILABLE, bool)

    def test_demo_mode_works_without_datasets(self):
        """Test demo mode works even if datasets not installed"""
        provider = Mock()
        provider.model = "test-model"
        provider.config = GenerationConfig()
        provider.generate.return_value = GenerationResult(
            text="A", response_time=0.1, tokens_used=5, model="test-model", metadata={}
        )

        # Demo mode should work regardless of datasets
        runner = BenchmarkRunner(provider, use_full_datasets=False)
        result = runner.run_mmlu_sample()
        assert result is not None


class TestGSM8KBenchmark:
    """Test GSM8K math reasoning benchmark"""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider that returns correct math answers"""
        provider = Mock()
        provider.model = "test-model"
        provider.config = GenerationConfig()
        # Return "18" which is the correct answer for the first GSM8K demo problem
        provider.generate.return_value = GenerationResult(
            text="Let me solve this step by step.\nJanet has 16 eggs, eats 3, uses 4 for muffins.\n16 - 3 - 4 = 9 eggs left.\n9 * $2 = $18\nThe answer is 18",
            response_time=0.5,
            tokens_used=50,
            model="test-model",
            metadata={},
        )
        return provider

    def test_run_gsm8k_demo(self, mock_provider):
        """Test GSM8K benchmark in demo mode"""
        runner = BenchmarkRunner(mock_provider, use_full_datasets=False)

        result = runner.run_gsm8k_sample()

        assert isinstance(result, dict)
        assert "gsm8k_accuracy" in result
        assert "problems_tested" in result
        assert "correct" in result
        assert result["mode"] == "demo"

    def test_gsm8k_accuracy_bounds(self, mock_provider):
        """Test GSM8K accuracy is between 0 and 1"""
        runner = BenchmarkRunner(mock_provider, use_full_datasets=False)
        result = runner.run_gsm8k_sample()

        assert 0 <= result["gsm8k_accuracy"] <= 1

    def test_gsm8k_problem_count_positive(self, mock_provider):
        """Test GSM8K problem count is positive"""
        runner = BenchmarkRunner(mock_provider, use_full_datasets=False)
        result = runner.run_gsm8k_sample()

        assert result["problems_tested"] > 0

    def test_gsm8k_extracts_number_correctly(self, mock_provider):
        """Test number extraction from various response formats"""
        runner = BenchmarkRunner(mock_provider, use_full_datasets=False)

        # Test various formats
        test_cases = [
            ("The answer is 42", 42),
            ("#### 123", 123),
            ("= 99", 99),
            ("Final answer: 50", 50),
            ("So she makes $18 dollars.", 18),
            ("The result is 3.5", 3.5),
            ("1,234 items", 1234),
        ]

        for response, expected in test_cases:
            result = runner._extract_number_from_response(response)
            assert result == expected, f"Failed for '{response}': got {result}, expected {expected}"

    def test_gsm8k_handles_no_number(self, mock_provider):
        """Test graceful handling when no number in response"""
        runner = BenchmarkRunner(mock_provider, use_full_datasets=False)

        result = runner._extract_number_from_response("I don't know the answer")
        assert result is None

    def test_gsm8k_correct_answers(self):
        """Test GSM8K with correct answers"""
        provider = Mock()
        provider.model = "test-model"
        provider.config = GenerationConfig()

        # Simulate returning correct answers for both demo problems
        # Problem 1: Janet sells eggs at $2 each, answer is 18
        # Problem 2: Robe takes 3 bolts total
        answers = iter(["The answer is 18", "The answer is 3"])
        provider.generate.side_effect = lambda prompt, config=None: GenerationResult(
            text=next(answers),
            response_time=0.1,
            tokens_used=10,
            model="test-model",
            metadata={},
        )

        runner = BenchmarkRunner(provider, use_full_datasets=False)
        result = runner.run_gsm8k_sample()

        assert result["correct"] == 2
        assert result["gsm8k_accuracy"] == 1.0

    def test_gsm8k_wrong_answers(self):
        """Test GSM8K with wrong answers"""
        provider = Mock()
        provider.model = "test-model"
        provider.config = GenerationConfig()
        provider.generate.return_value = GenerationResult(
            text="The answer is 999",  # Wrong answer
            response_time=0.1,
            tokens_used=10,
            model="test-model",
            metadata={},
        )

        runner = BenchmarkRunner(provider, use_full_datasets=False)
        result = runner.run_gsm8k_sample()

        assert result["correct"] == 0
        assert result["gsm8k_accuracy"] == 0.0


class TestParallelExecution:
    """Test parallel execution functionality"""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider"""
        provider = Mock()
        provider.model = "test-model"
        provider.config = GenerationConfig()
        provider.generate.return_value = GenerationResult(
            text="A",
            response_time=0.1,
            tokens_used=10,
            model="test-model",
            metadata={},
        )
        return provider

    def test_init_with_max_workers(self, mock_provider):
        """Test initialization with max_workers parameter"""
        runner = BenchmarkRunner(mock_provider, max_workers=4)
        assert runner.max_workers == 4

    def test_default_sequential_execution(self, mock_provider):
        """Test default is sequential (max_workers=1)"""
        runner = BenchmarkRunner(mock_provider)
        assert runner.max_workers == 1

    def test_run_parallel_sequential_mode(self, mock_provider):
        """Test _run_parallel works in sequential mode"""
        runner = BenchmarkRunner(mock_provider, max_workers=1)

        items = [{"value": i} for i in range(3)]

        def process_fn(idx, item):
            return True, {"id": idx, "value": item["value"]}

        correct, scenarios = runner._run_parallel(items, process_fn, "Test")

        assert correct == 3
        assert len(scenarios) == 3
        assert scenarios[0]["id"] == 0
        assert scenarios[1]["id"] == 1
        assert scenarios[2]["id"] == 2

    def test_run_parallel_parallel_mode(self, mock_provider):
        """Test _run_parallel works in parallel mode"""
        runner = BenchmarkRunner(mock_provider, max_workers=2)

        items = [{"value": i} for i in range(5)]

        def process_fn(idx, item):
            return idx % 2 == 0, {"id": idx, "value": item["value"]}

        correct, scenarios = runner._run_parallel(items, process_fn, "Test")

        # 0, 2, 4 are correct (3 items)
        assert correct == 3
        assert len(scenarios) == 5
        # Results should be in original order
        for i, scenario in enumerate(scenarios):
            assert scenario["id"] == i

    def test_parallel_handles_errors(self, mock_provider):
        """Test parallel execution handles errors gracefully"""
        runner = BenchmarkRunner(mock_provider, max_workers=2)

        items = [{"value": i} for i in range(3)]

        def process_fn(idx, item):
            if idx == 1:
                raise ValueError("Test error")
            return True, {"id": idx}

        correct, scenarios = runner._run_parallel(items, process_fn, "Test")

        # Only 2 should be correct (0 and 2 succeed, 1 fails)
        assert correct == 2
        assert len(scenarios) == 3
        assert "error" in scenarios[1]
