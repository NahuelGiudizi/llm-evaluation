"""
Tests for academic evaluation features.

Tests statistical metrics, baselines, error analysis, and export functions.
"""

import pytest

from llm_evaluator.academic_baselines import (
    ACADEMIC_BASELINES,
    compare_to_baselines,
    generate_comparison_table,
    get_baselines,
    list_available_baselines,
)
from llm_evaluator.error_analysis import ErrorAnalyzer, cohens_kappa, expected_calibration_error
from llm_evaluator.export import (
    export_to_latex,
    generate_bibtex,
    generate_methods_section,
    generate_references_bibtex,
    generate_reproducibility_manifest,
)
from llm_evaluator.statistical_metrics import (
    bootstrap_confidence_interval,
    calculate_all_statistics,
    calculate_standard_error,
    calculate_wilson_ci,
    cohens_h,
    mcnemar_test,
)


class TestWilsonCI:
    """Tests for Wilson confidence interval calculation."""

    def test_standard_case(self) -> None:
        """Test Wilson CI with typical accuracy."""
        lower, upper = calculate_wilson_ci(850, 1000)
        assert 0.82 < lower < 0.84
        assert 0.86 < upper < 0.88

    def test_perfect_score(self) -> None:
        """Test CI for 100% accuracy."""
        lower, upper = calculate_wilson_ci(100, 100)
        assert lower > 0.95
        assert upper == 1.0

    def test_zero_score(self) -> None:
        """Test CI for 0% accuracy."""
        lower, upper = calculate_wilson_ci(0, 100)
        assert lower < 0.001  # Essentially zero
        assert upper < 0.05

    def test_small_sample(self) -> None:
        """Test CI with small sample size."""
        lower, upper = calculate_wilson_ci(8, 10)
        # CI should be wider for small samples
        assert (upper - lower) > 0.2

    def test_invalid_inputs(self) -> None:
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            calculate_wilson_ci(-1, 100)  # Negative correct

        with pytest.raises(ValueError):
            calculate_wilson_ci(10, 0)  # Zero total

        with pytest.raises(ValueError):
            calculate_wilson_ci(100, 50)  # correct > total


class TestStandardError:
    """Tests for standard error calculation."""

    def test_standard_case(self) -> None:
        """Test SE with typical values."""
        se = calculate_standard_error(850, 1000)
        assert 0.01 < se < 0.02

    def test_perfect_score(self) -> None:
        """Test SE for p=1.0."""
        se = calculate_standard_error(100, 100)
        assert se == 0.0  # SE is 0 when p=1

    def test_fifty_percent(self) -> None:
        """Test SE for p=0.5 (maximum variance)."""
        se = calculate_standard_error(50, 100)
        # SE should be at maximum for p=0.5
        assert se == pytest.approx(0.05, rel=0.01)


class TestBootstrapCI:
    """Tests for bootstrap confidence interval."""

    def test_standard_case(self) -> None:
        """Test bootstrap CI with typical predictions."""
        predictions = [True] * 85 + [False] * 15
        lower, upper = bootstrap_confidence_interval(predictions)
        assert 0.75 < lower < 0.85
        assert 0.85 < upper < 0.95

    def test_reproducibility(self) -> None:
        """Test that same seed gives same results."""
        predictions = [True] * 70 + [False] * 30
        ci1 = bootstrap_confidence_interval(predictions, random_seed=42)
        ci2 = bootstrap_confidence_interval(predictions, random_seed=42)
        assert ci1 == ci2

    def test_empty_predictions(self) -> None:
        """Test error handling for empty list."""
        with pytest.raises(ValueError):
            bootstrap_confidence_interval([])


class TestMcNemarTest:
    """Tests for McNemar's test."""

    def test_significant_difference(self) -> None:
        """Test McNemar when models differ significantly."""
        # Model A: 90% correct, Model B: 60% correct
        # Discordant pairs: A right B wrong = 30, A wrong B right = 0
        model_a = [True] * 90 + [False] * 10
        model_b = [True] * 60 + [False] * 40
        truth = [True] * 100

        result = mcnemar_test(model_a, model_b, truth)
        assert result["p_value"] < 0.05
        assert result["significant"]  # True or np.True_ both work
        assert "significantly better" in result["conclusion"].lower()

    def test_no_difference(self) -> None:
        """Test McNemar when models have same performance."""
        # Same predictions
        predictions = [True] * 80 + [False] * 20
        truth = [True] * 100

        result = mcnemar_test(predictions, predictions, truth)
        assert result["p_value"] > 0.05
        assert result["significant"] is False

    def test_length_mismatch(self) -> None:
        """Test error handling for mismatched lengths."""
        with pytest.raises(ValueError):
            mcnemar_test([True] * 10, [True] * 5, [True] * 10)


class TestCohensH:
    """Tests for Cohen's h effect size."""

    def test_large_effect(self) -> None:
        """Test with large effect size."""
        result = cohens_h(0.95, 0.50)
        assert result["magnitude"] == "large"
        assert result["h"] > 0.8

    def test_medium_effect(self) -> None:
        """Test with medium effect size."""
        result = cohens_h(0.80, 0.50)  # Larger difference for medium
        assert result["magnitude"] in ["medium", "large"]
        assert abs(result["h"]) >= 0.5

    def test_small_effect(self) -> None:
        """Test with small effect size."""
        result = cohens_h(0.70, 0.60)
        assert result["magnitude"] in ["small", "negligible"]

    def test_invalid_proportions(self) -> None:
        """Test error handling for invalid proportions."""
        with pytest.raises(ValueError):
            cohens_h(1.5, 0.5)  # p1 > 1

        with pytest.raises(ValueError):
            cohens_h(0.5, -0.1)  # p2 < 0


class TestAcademicBaselines:
    """Tests for academic baselines module."""

    def test_baselines_structure(self) -> None:
        """Test that baselines have correct structure."""
        assert "mmlu" in ACADEMIC_BASELINES
        assert "truthfulqa" in ACADEMIC_BASELINES
        assert "hellaswag" in ACADEMIC_BASELINES

        for benchmark, baselines in ACADEMIC_BASELINES.items():
            assert "random_chance" in baselines
            for model, data in baselines.items():
                assert "score" in data
                assert isinstance(data["score"], (int, float))
                assert 0 <= data["score"] <= 1

    def test_get_baselines(self) -> None:
        """Test getting baselines for a benchmark."""
        mmlu = get_baselines("mmlu")
        assert "gpt-4" in mmlu
        assert mmlu["gpt-4"]["score"] > 0.8

    def test_get_baselines_invalid(self) -> None:
        """Test error for invalid benchmark."""
        with pytest.raises(ValueError):
            get_baselines("invalid_benchmark")

    def test_compare_to_baselines(self) -> None:
        """Test model comparison to baselines."""
        result = compare_to_baselines("my-model", 0.75, "mmlu")

        assert result["model_score"] == 0.75
        assert "gpt-4" in result["comparisons"]
        assert result["comparisons"]["gpt-4"]["delta"] < 0  # Below GPT-4
        assert 1 <= result["rank"] <= 20
        assert result["tier"] in ["sota", "strong", "mid-range", "weak", "below-random"]

    def test_comparison_table_markdown(self) -> None:
        """Test markdown table generation."""
        results = {"model-a": 0.75, "model-b": 0.68}
        table = generate_comparison_table(results, "mmlu", "markdown")
        assert "| Model |" in table
        assert "model-a" in table
        assert "75.0%" in table

    def test_list_available_baselines(self) -> None:
        """Test listing all baselines."""
        available = list_available_baselines()
        assert "mmlu" in available
        assert "gpt-4" in available["mmlu"]


class TestErrorAnalyzer:
    """Tests for error analysis module."""

    def test_analyze_errors(self) -> None:
        """Test basic error analysis."""
        analyzer = ErrorAnalyzer()

        predictions = ["A", "B", "C", "A", "B"]
        ground_truth = ["A", "C", "C", "B", "B"]
        questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]

        result = analyzer.analyze_errors(predictions, ground_truth, questions)

        assert result.total_errors == 2
        assert result.total_correct == 3
        assert result.error_rate == 0.4

    def test_perfect_predictions(self) -> None:
        """Test with no errors."""
        analyzer = ErrorAnalyzer()

        predictions = ["A", "B", "C"]
        ground_truth = ["A", "B", "C"]
        questions = ["Q1", "Q2", "Q3"]

        result = analyzer.analyze_errors(predictions, ground_truth, questions)

        assert result.total_errors == 0
        assert result.error_rate == 0.0

    def test_length_mismatch(self) -> None:
        """Test error handling for mismatched lengths."""
        analyzer = ErrorAnalyzer()

        with pytest.raises(ValueError):
            analyzer.analyze_errors(["A", "B"], ["A"], ["Q1", "Q2"])


class TestCalibrationError:
    """Tests for Expected Calibration Error."""

    def test_well_calibrated(self) -> None:
        """Test ECE for well-calibrated predictions."""
        # Confidence matches accuracy perfectly
        confidences = [0.9] * 90 + [0.1] * 10
        correct = [True] * 90 + [False] * 10

        result = expected_calibration_error(confidences, correct)
        assert result.ece < 0.1  # Should be well-calibrated

    def test_overconfident(self) -> None:
        """Test ECE for overconfident model."""
        # High confidence but poor accuracy
        confidences = [0.95] * 100
        correct = [True] * 50 + [False] * 50

        result = expected_calibration_error(confidences, correct)
        assert result.ece > 0.3  # Should show poor calibration
        assert "overconfident" in result.interpretation.lower()

    def test_empty_predictions(self) -> None:
        """Test with empty input."""
        result = expected_calibration_error([], [])
        assert result.ece == 0.0
        assert result.n_samples == 0


class TestCohensKappa:
    """Tests for Cohen's Kappa inter-annotator agreement."""

    def test_perfect_agreement(self) -> None:
        """Test kappa for perfect agreement."""
        annotations = ["A", "B", "C", "A", "B"]
        result = cohens_kappa(annotations, annotations)
        assert result["kappa"] == 1.0
        assert "perfect" in result["interpretation"].lower()

    def test_moderate_agreement(self) -> None:
        """Test kappa for moderate agreement."""
        a = ["A", "B", "C", "A", "B", "A"]
        b = ["A", "B", "C", "B", "B", "A"]  # 1 disagreement
        result = cohens_kappa(a, b)
        assert 0.5 < result["kappa"] < 1.0

    def test_no_agreement(self) -> None:
        """Test kappa for chance agreement."""
        a = ["A", "A", "A", "A"]
        b = ["B", "B", "B", "B"]
        result = cohens_kappa(a, b)
        assert result["kappa"] <= 0  # Can be negative or zero


class TestExport:
    """Tests for export module."""

    def test_export_to_latex(self) -> None:
        """Test LaTeX table generation."""
        results = {
            "llama-3": {
                "mmlu": 0.68,
                "mmlu_ci": (0.65, 0.71),
                "truthfulqa": 0.45,
                "hellaswag": 0.72,
            }
        }
        latex = export_to_latex(results)

        assert r"\begin{table}" in latex
        assert r"\toprule" in latex
        assert "llama-3" in latex
        assert "68.0" in latex

    def test_generate_bibtex(self) -> None:
        """Test BibTeX citation generation."""
        metadata = {
            "version": "2.0.0",
            "date": "2024-12-01",
            "author": "Test Author",
            "github_url": "https://github.com/test/repo",
        }
        bibtex = generate_bibtex(metadata)

        assert "@software{" in bibtex
        assert "2.0.0" in bibtex
        assert "Test Author" in bibtex

    def test_reproducibility_manifest(self) -> None:
        """Test manifest generation."""
        config = {"temperature": 0.0, "random_seed": 42}
        results = {"mmlu_accuracy": 0.75}

        manifest = generate_reproducibility_manifest(config, results)

        assert "evaluation_hash" in manifest
        assert manifest["evaluation_hash"].startswith("sha256:")
        assert "timestamp" in manifest
        assert manifest["config"]["random_seed"] == 42

    def test_methods_section(self) -> None:
        """Test methods section generation."""
        config = {"n_samples": 1000, "confidence_level": 0.95}
        methods = generate_methods_section(config)

        assert "1000" in methods
        assert "95" in methods  # 95% without the exact formatting
        assert "Wilson" in methods

    def test_references_bibtex(self) -> None:
        """Test standard references generation."""
        refs = generate_references_bibtex()

        assert "hendrycks2021measuring" in refs
        assert "lin2022truthfulqa" in refs
        assert "zellers2019hellaswag" in refs


class TestCalculateAllStatistics:
    """Tests for comprehensive statistics function."""

    def test_all_statistics(self) -> None:
        """Test that all statistics are computed."""
        stats = calculate_all_statistics(850, 1000)

        assert "accuracy" in stats
        assert "wilson_ci" in stats
        assert "standard_error" in stats
        assert "ci_width" in stats
        assert stats["accuracy"] == 0.85

    def test_with_predictions_list(self) -> None:
        """Test with predictions list for bootstrap."""
        predictions = [True] * 85 + [False] * 15
        stats = calculate_all_statistics(85, 100, predictions)

        assert "bootstrap_ci" in stats
        assert len(stats["bootstrap_ci"]) == 2


class TestPowerAnalysis:
    """Tests for power analysis functions."""

    def test_power_analysis_basic(self) -> None:
        """Test basic power analysis calculation."""
        from llm_evaluator.statistical_metrics import power_analysis_sample_size

        result = power_analysis_sample_size(
            expected_difference=0.05,
            baseline_accuracy=0.75,
            power=0.80,
            alpha=0.05,
        )

        assert "n_per_group" in result
        assert "total_n" in result
        assert "effect_size_h" in result
        assert "interpretation" in result
        assert "recommendations" in result
        assert result["n_per_group"] > 0
        assert result["total_n"] == result["n_per_group"] * 2

    def test_power_analysis_larger_difference_needs_smaller_sample(self) -> None:
        """Larger effect size needs smaller sample."""
        from llm_evaluator.statistical_metrics import power_analysis_sample_size

        small_diff = power_analysis_sample_size(expected_difference=0.05)
        large_diff = power_analysis_sample_size(expected_difference=0.15)

        assert large_diff["n_per_group"] < small_diff["n_per_group"]

    def test_power_analysis_higher_power_needs_larger_sample(self) -> None:
        """Higher power needs larger sample size."""
        from llm_evaluator.statistical_metrics import power_analysis_sample_size

        power_80 = power_analysis_sample_size(power=0.80)
        power_95 = power_analysis_sample_size(power=0.95)

        assert power_95["n_per_group"] > power_80["n_per_group"]

    def test_power_analysis_effect_size(self) -> None:
        """Test Cohen's h effect size is calculated."""
        from llm_evaluator.statistical_metrics import power_analysis_sample_size

        result = power_analysis_sample_size(
            expected_difference=0.10,
            baseline_accuracy=0.70,
        )

        assert result["effect_size_h"] > 0
        assert 0 < result["effect_size_h"] < 1

    def test_power_analysis_recommendations(self) -> None:
        """Test benchmark recommendations are included."""
        from llm_evaluator.statistical_metrics import power_analysis_sample_size

        result = power_analysis_sample_size()

        recs = result["recommendations"]
        assert "mmlu" in recs
        assert "truthfulqa" in recs
        assert "hellaswag" in recs
        assert "gsm8k" in recs

    def test_power_analysis_invalid_inputs(self) -> None:
        """Test error handling for invalid inputs."""
        from llm_evaluator.statistical_metrics import power_analysis_sample_size

        with pytest.raises(ValueError):
            power_analysis_sample_size(baseline_accuracy=1.5)  # > 1

        with pytest.raises(ValueError):
            power_analysis_sample_size(expected_difference=-0.1)  # negative

        with pytest.raises(ValueError):
            power_analysis_sample_size(baseline_accuracy=0.95, expected_difference=0.10)  # sum > 1

    def test_minimum_sample_size_table(self) -> None:
        """Test reference table generation."""
        from llm_evaluator.statistical_metrics import minimum_sample_size_table

        table = minimum_sample_size_table()

        assert "power_80" in table
        assert "power_90" in table
        assert "power_95" in table

        assert "diff_2pct" in table["power_80"]
        assert "diff_5pct" in table["power_80"]
        assert "diff_10pct" in table["power_80"]
        assert "diff_15pct" in table["power_80"]

        # Smaller differences need larger samples
        assert table["power_80"]["diff_2pct"] > table["power_80"]["diff_10pct"]

        # Higher power needs larger samples
        assert table["power_95"]["diff_5pct"] > table["power_80"]["diff_5pct"]


class TestReproducibilitySeeds:
    """Tests for reproducibility with seeds."""

    def test_benchmark_runner_accepts_seed(self) -> None:
        """Test BenchmarkRunner accepts seed parameter."""
        from unittest.mock import Mock

        from llm_evaluator.benchmarks import BenchmarkRunner
        from llm_evaluator.providers import GenerationConfig, GenerationResult

        provider = Mock()
        provider.model = "test"
        provider.config = GenerationConfig()
        provider.generate.return_value = GenerationResult(
            text="A",
            response_time=0.1,
            tokens_used=10,
            model="test",
            metadata={},
        )

        runner = BenchmarkRunner(provider, use_full_datasets=False, seed=42)
        assert runner.seed == 42

    def test_benchmark_runner_accepts_temperature(self) -> None:
        """Test BenchmarkRunner accepts temperature parameter."""
        from unittest.mock import Mock

        from llm_evaluator.benchmarks import BenchmarkRunner
        from llm_evaluator.providers import GenerationConfig, GenerationResult

        provider = Mock()
        provider.model = "test"
        provider.config = GenerationConfig()
        provider.generate.return_value = GenerationResult(
            text="A",
            response_time=0.1,
            tokens_used=10,
            model="test",
            metadata={},
        )

        runner = BenchmarkRunner(provider, use_full_datasets=False, temperature=0.0)
        assert runner.temperature == 0.0

    def test_generation_config_created_with_temperature(self) -> None:
        """Test _get_generation_config creates config with temperature."""
        from unittest.mock import Mock

        from llm_evaluator.benchmarks import BenchmarkRunner
        from llm_evaluator.providers import GenerationConfig, GenerationResult

        provider = Mock()
        provider.model = "test"
        provider.config = GenerationConfig()
        provider.generate.return_value = GenerationResult(
            text="A",
            response_time=0.1,
            tokens_used=10,
            model="test",
            metadata={},
        )

        runner = BenchmarkRunner(provider, use_full_datasets=False, temperature=0.5)
        config = runner._get_generation_config()

        assert config is not None
        assert config.temperature == 0.5

    def test_no_config_when_no_temperature(self) -> None:
        """Test _get_generation_config returns None when no temperature set."""
        from unittest.mock import Mock

        from llm_evaluator.benchmarks import BenchmarkRunner
        from llm_evaluator.providers import GenerationConfig, GenerationResult

        provider = Mock()
        provider.model = "test"
        provider.config = GenerationConfig()
        provider.generate.return_value = GenerationResult(
            text="A",
            response_time=0.1,
            tokens_used=10,
            model="test",
            metadata={},
        )

        runner = BenchmarkRunner(provider, use_full_datasets=False)
        config = runner._get_generation_config()

        assert config is None
