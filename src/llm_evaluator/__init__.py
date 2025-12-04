"""
LLM Evaluation Suite

A production-ready framework for comprehensive evaluation of Large Language Models.

Key Features:
- Performance Metrics: Response time, throughput, token efficiency
- Quality Metrics: Accuracy, coherence, hallucination detection
- Standard Benchmarks: MMLU (14,042), TruthfulQA (817), HellaSwag (10,042)
- Interactive Visualizations: Dashboards, radar charts, heatmaps
- Clean Architecture: Provider abstraction, dependency injection

Quick Start:
    >>> from llm_evaluator import ModelEvaluator
    >>> from llm_evaluator.providers.ollama_provider import OllamaProvider
    >>>
    >>> provider = OllamaProvider(model="llama3.2:1b")
    >>> evaluator = ModelEvaluator(provider=provider)
    >>> results = evaluator.evaluate_all()
    >>> print(f"Score: {results.overall_score:.1%}")
    Score: 73.5%

Advanced Usage:
    >>> # Compare multiple models
    >>> from llm_evaluator import quick_comparison
    >>>
    >>> results = {
    ...     "llama3.2:1b": {"mmlu": 0.65, "accuracy": 0.75},
    ...     "mistral:7b": {"mmlu": 0.78, "accuracy": 0.82}
    ... }
    >>> quick_comparison(results, output_dir="outputs")

    >>> # Full benchmark evaluation
    >>> from llm_evaluator.benchmarks import BenchmarkRunner
    >>>
    >>> runner = BenchmarkRunner(provider, use_full_datasets=True, sample_size=100)
    >>> benchmark_results = runner.run_all_benchmarks()
    >>> print(f"MMLU: {benchmark_results['mmlu_accuracy']:.1%}")
    MMLU: 67.3%

Examples: examples/demo.py
Documentation: docs/
Repository: https://github.com/NahuelGiudizi/llm-evaluation
"""

__version__ = "2.3.3"
__author__ = "Nahuel Giudizi"
__license__ = "MIT"

from .academic_baselines import ACADEMIC_BASELINES, compare_to_baselines, get_baselines
from .benchmarks import BenchmarkRunner
from .error_analysis import ErrorAnalyzer, expected_calibration_error
from .evaluator import EvaluationResults, ModelEvaluator
from .export import export_to_latex, generate_bibtex, generate_reproducibility_manifest
from .metrics import PerformanceMetrics, QualityMetrics
from .providers import GenerationConfig, LLMProvider, ProviderError

# Academic features (v2.0)
from .statistical_metrics import (
    bootstrap_confidence_interval,
    calculate_all_statistics,
    calculate_standard_error,
    calculate_wilson_ci,
    cohens_h,
    mcnemar_test,
)
from .visualizations import EvaluationVisualizer, quick_comparison

__all__ = [
    # Core
    "ModelEvaluator",
    "EvaluationResults",
    "PerformanceMetrics",
    "QualityMetrics",
    "BenchmarkRunner",
    "EvaluationVisualizer",
    "quick_comparison",
    "LLMProvider",
    "GenerationConfig",
    "ProviderError",
    # Academic statistics
    "calculate_wilson_ci",
    "calculate_standard_error",
    "bootstrap_confidence_interval",
    "mcnemar_test",
    "cohens_h",
    "calculate_all_statistics",
    # Academic baselines
    "ACADEMIC_BASELINES",
    "compare_to_baselines",
    "get_baselines",
    # Error analysis
    "ErrorAnalyzer",
    "expected_calibration_error",
    # Export
    "export_to_latex",
    "generate_bibtex",
    "generate_reproducibility_manifest",
]
