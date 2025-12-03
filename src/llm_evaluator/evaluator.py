"""
Main ModelEvaluator class for comprehensive LLM testing

Refactored with Clean Architecture principles:
- Dependency Injection for LLM providers
- Separation of concerns
- Comprehensive error handling
- Academic-grade statistical analysis (v2.0)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .benchmarks import BenchmarkRunner
from .metrics import PerformanceMetrics, QualityMetrics
from .providers import GenerationConfig, LLMProvider, ProviderError
from .providers.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


@dataclass
class DetailedMetrics:
    """Strongly typed detailed metrics (no Any types)"""

    performance: Dict[str, float] = field(default_factory=lambda: {})
    quality: Dict[str, float] = field(default_factory=lambda: {})
    benchmarks: Dict[str, float] = field(default_factory=lambda: {})
    errors: List[str] = field(default_factory=lambda: [])


@dataclass
class EvaluationResults:
    """Container for evaluation results"""

    model_name: str
    accuracy: float
    avg_response_time: float
    token_efficiency: float
    hallucination_rate: float
    coherence_score: float
    overall_score: float
    detailed_metrics: DetailedMetrics
    system_info: Optional[Dict[str, Any]] = None  # Hardware/software info


@dataclass
class AcademicEvaluationResults:
    """
    Academic-grade evaluation results with statistical rigor.

    Includes confidence intervals, baseline comparisons, and
    export-ready data for academic papers.
    """

    model_name: str
    # MMLU results
    mmlu_accuracy: float
    mmlu_ci: Tuple[float, float]
    mmlu_se: float
    mmlu_n: int | float
    # TruthfulQA results
    truthfulqa_accuracy: float
    truthfulqa_ci: Tuple[float, float]
    truthfulqa_se: float
    truthfulqa_n: int | float
    # HellaSwag results
    hellaswag_accuracy: float
    hellaswag_ci: Tuple[float, float]
    hellaswag_se: float
    hellaswag_n: int | float
    # Aggregate
    average_accuracy: float
    # Baseline comparison
    baseline_comparison: Dict[str, Any]
    # Reproducibility
    reproducibility_manifest: Dict[str, Any]
    # Metadata
    elapsed_time: float
    timestamp: str
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "mmlu": {
                "accuracy": self.mmlu_accuracy,
                "confidence_interval_95": self.mmlu_ci,
                "standard_error": self.mmlu_se,
                "n_samples": self.mmlu_n,
            },
            "truthfulqa": {
                "accuracy": self.truthfulqa_accuracy,
                "confidence_interval_95": self.truthfulqa_ci,
                "standard_error": self.truthfulqa_se,
                "n_samples": self.truthfulqa_n,
            },
            "hellaswag": {
                "accuracy": self.hellaswag_accuracy,
                "confidence_interval_95": self.hellaswag_ci,
                "standard_error": self.hellaswag_se,
                "n_samples": self.hellaswag_n,
            },
            "average_accuracy": self.average_accuracy,
            "baseline_comparison": self.baseline_comparison,
            "reproducibility_manifest": self.reproducibility_manifest,
            "elapsed_time": self.elapsed_time,
            "timestamp": self.timestamp,
        }


class ModelEvaluator:
    """
    Comprehensive LLM evaluation framework with Dependency Injection

    Refactored to use LLMProvider interface, enabling:
    - Easy provider swapping (Ollama, OpenAI, etc.)
    - Proper testing with mocks
    - Clean separation of concerns
    - Robust error handling

    Args:
        provider: LLM provider implementation (default: OllamaProvider)
        config: Generation configuration

    Example:
        >>> from llm_evaluator.providers.ollama_provider import OllamaProvider
        >>> provider = OllamaProvider(model="llama3.2:1b")
        >>> evaluator = ModelEvaluator(provider=provider)
        >>> results = evaluator.evaluate_all()
        >>> print(f"Overall Score: {results.overall_score:.2f}")
    """

    def __init__(
        self, provider: Optional[LLMProvider] = None, config: Optional[GenerationConfig] = None
    ):
        """
        Initialize evaluator with dependency injection

        Args:
            provider: LLM provider instance (defaults to Ollama)
            config: Generation configuration
        """
        # Load configuration from environment/config file
        from llm_evaluator.config import get_evaluator_config

        self.app_config = get_evaluator_config()

        # Default to Ollama if no provider specified (backwards compatibility)
        self.provider = provider or OllamaProvider(
            model=self.app_config.default_model, config=config
        )

        # Use provided config or create from app_config defaults
        self.config = config or GenerationConfig(
            temperature=self.app_config.default_temperature,
            max_tokens=self.app_config.default_max_tokens,
            timeout_seconds=self.app_config.default_timeout,
            retry_attempts=self.app_config.default_retry_attempts,
        )

        self.performance_metrics = PerformanceMetrics()
        self.quality_metrics = QualityMetrics()
        self.benchmark_runner = BenchmarkRunner(self.provider)

    def chat(self, prompt: str) -> tuple[str, float]:
        """
        Send prompt to LLM and return response with timing

        Now uses provider abstraction with error handling

        Args:
            prompt: User prompt to send to the model

        Returns:
            (response_text, response_time_in_seconds)

        Raises:
            ProviderError: If generation fails after retries
        """
        try:
            result = self.provider.generate(prompt, None, self.config)
            return result.text, result.response_time
        except ProviderError as e:
            logger.error(f"Provider error in chat: {e}")
            raise

    def evaluate_performance(self, num_samples: int = 10) -> Dict[str, float]:
        """
        Evaluate performance metrics: response time, token efficiency

        Now uses batch generation for better performance

        Args:
            num_samples: Number of test prompts to run

        Returns:
            Dictionary with performance metrics

        Raises:
            ProviderError: If generation fails
        """
        test_prompts = [
            "What is Python?",
            "Explain machine learning in one sentence.",
            "What is 2+2?",
            "Name three programming languages.",
            "What is the capital of France?",
            "Define artificial intelligence.",
            "What is a neural network?",
            "Explain what an API is.",
            "What does CPU stand for?",
            "What is cloud computing?",
        ][:num_samples]

        try:
            # Use batch generation for better efficiency
            results = self.provider.generate_batch(test_prompts, None, self.config)

            response_times = [r.response_time for r in results]
            token_counts = [r.tokens_used or len(r.text) / 4 for r in results]  # Fallback estimate

            return {
                "avg_response_time": sum(response_times) / len(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "avg_tokens_per_response": sum(token_counts) / len(token_counts),
                "tokens_per_second": sum(token_counts) / sum(response_times),
            }
        except ProviderError as e:
            logger.error(f"Performance evaluation failed: {e}")
            raise

    def evaluate_quality(self, test_set: Optional[List[Dict[str, str]]] = None) -> Dict[str, float]:
        """
        Evaluate quality metrics: accuracy, coherence, hallucination

        Uses provider interface with error handling

        Args:
            test_set: List of {"prompt": str, "expected": str} dicts

        Returns:
            Dictionary with quality metrics

        Raises:
            ProviderError: If generation fails
        """
        if test_set is None:
            # Default test set
            test_set = [
                {"prompt": "What is 5+3?", "expected": "8"},
                {"prompt": "What is the capital of Japan?", "expected": "Tokyo"},
                {"prompt": "How many continents are there?", "expected": "7"},
                {"prompt": "What year did World War 2 end?", "expected": "1945"},
                {"prompt": "What is H2O?", "expected": "water"},
            ]

        correct = 0
        coherent = 0

        try:
            for test in test_set:
                response, _ = self.chat(test["prompt"])

                # Check accuracy (simple substring match)
                if test["expected"].lower() in response.lower():
                    correct += 1

                # Check coherence (basic heuristics)
                if (
                    len(response) > 10
                    and not response.startswith("Error")
                    and response.count(".") <= 10
                ):  # Not too fragmented
                    coherent += 1

            accuracy = correct / len(test_set) if test_set else 0
            coherence = coherent / len(test_set) if test_set else 0

            # Hallucination detection (simplified)
            hallucination_prompts = [
                "Who won the 2025 World Cup?",  # Future event
                "What is the capital of Atlantis?",  # Fictional place
            ]

            hallucinations = 0
            for prompt in hallucination_prompts:
                response, _ = self.chat(prompt)
                # Good model should express uncertainty
                uncertainty_markers = [
                    "don't know",
                    "not sure",
                    "cannot",
                    "no information",
                    "unclear",
                    "uncertain",
                ]
                if not any(marker in response.lower() for marker in uncertainty_markers):
                    hallucinations += 1

            hallucination_rate = hallucinations / len(hallucination_prompts)

            return {
                "accuracy": accuracy,
                "coherence_score": coherence,
                "hallucination_rate": hallucination_rate,
            }
        except ProviderError as e:
            logger.error(f"Quality evaluation failed: {e}")
            raise

    def evaluate_all(self) -> EvaluationResults:
        """
        Run comprehensive evaluation across all metrics with system info

        With proper error handling and logging

        Returns:
            EvaluationResults object with all metrics and system information

        Raises:
            ProviderError: If evaluation fails
        """
        # Collect system information
        from .system_info import collect_system_info

        system_info = collect_system_info()

        model_name = self.provider.model
        logger.info(f"Starting comprehensive evaluation for {model_name}")

        print(f"\n🔍 Evaluating {model_name}...")
        print("=" * 60)

        # Print system info
        print(system_info.to_markdown())
        print("=" * 60)

        errors: List[str] = []

        # Performance metrics
        print("\n📊 Performance Metrics...")
        try:
            perf_metrics = self.evaluate_performance()
        except ProviderError as e:
            logger.error(f"Performance evaluation failed: {e}")
            errors.append(f"Performance: {str(e)}")
            perf_metrics = {"avg_response_time": 0.0, "tokens_per_second": 0.0}

        # Quality metrics
        print("✅ Quality Metrics...")
        try:
            quality_metrics = self.evaluate_quality()
        except ProviderError as e:
            logger.error(f"Quality evaluation failed: {e}")
            errors.append(f"Quality: {str(e)}")
            quality_metrics = {
                "accuracy": 0.0,
                "coherence_score": 0.0,
                "hallucination_rate": 1.0,
            }

        # Calculate overall score
        # Normalize and combine metrics
        speed_score = min(
            1.0, 2.0 / max(perf_metrics["avg_response_time"], 0.1)
        )  # Faster is better
        accuracy_score = quality_metrics["accuracy"]
        coherence_score = quality_metrics["coherence_score"]
        anti_hallucination_score = 1.0 - quality_metrics["hallucination_rate"]

        overall_score = (
            speed_score * 0.2
            + accuracy_score * 0.3
            + coherence_score * 0.2
            + anti_hallucination_score * 0.3
        )

        results = EvaluationResults(
            model_name=model_name,
            accuracy=quality_metrics["accuracy"],
            avg_response_time=perf_metrics["avg_response_time"],
            token_efficiency=perf_metrics.get("tokens_per_second", 0),
            hallucination_rate=quality_metrics["hallucination_rate"],
            coherence_score=quality_metrics["coherence_score"],
            overall_score=overall_score,
            detailed_metrics=DetailedMetrics(
                performance=perf_metrics, quality=quality_metrics, benchmarks={}, errors=errors
            ),
            system_info=system_info.to_dict(),
        )

        # Print summary
        print("\n" + "=" * 60)
        print(f"📋 EVALUATION SUMMARY: {model_name}")
        print("=" * 60)
        print(f"  Accuracy:          {results.accuracy:.1%}")
        print(f"  Avg Response Time: {results.avg_response_time:.2f}s")
        print(f"  Token Efficiency:  {results.token_efficiency:.1f} tokens/s")
        print(f"  Hallucination Rate: {results.hallucination_rate:.1%}")
        print(f"  Coherence Score:   {results.coherence_score:.1%}")
        print(f"  Overall Score:     {results.overall_score:.2f}/1.00")

        if errors:
            print(f"\n⚠️  Errors encountered: {len(errors)}")
            for error in errors:
                print(f"    - {error}")

        print("=" * 60 + "\n")

        logger.info(f"Evaluation completed: {model_name} scored {overall_score:.2f}")

        return results

    def generate_report(self, results: EvaluationResults, output: str = "report.md") -> None:
        """
        Generate markdown report from evaluation results with system info

        Args:
            results: Evaluation results to report
            output: Output file path
        """
        logger.info(f"Generating report: {output}")

        # Build error section outside f-string to avoid Python 3.11 syntax issues
        error_section = ""
        if results.detailed_metrics.errors:
            newline = "\n"
            errors_list = newline.join(f"- {e}" for e in results.detailed_metrics.errors)
            error_section = f"## Errors\n\n{errors_list}"

        # Build system info section
        system_section = ""
        if results.system_info:
            from .system_info import SystemInfo

            sys_info = SystemInfo(**results.system_info)
            system_section = sys_info.to_markdown()

        report = f"""# Evaluation Report: {results.model_name}

## Summary

| Metric | Value |
|--------|-------|
| Accuracy | {results.accuracy:.1%} |
| Avg Response Time | {results.avg_response_time:.2f}s |
| Token Efficiency | {results.token_efficiency:.1f} tokens/s |
| Hallucination Rate | {results.hallucination_rate:.1%} |
| Coherence Score | {results.coherence_score:.1%} |
| **Overall Score** | **{results.overall_score:.2f}/1.00** |

{system_section}

## Performance Details

```
{results.detailed_metrics.performance}
```

## Quality Details

```
{results.detailed_metrics.quality}
```

{error_section}

---
Generated by LLM Evaluator v0.2.0
"""

        with open(output, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Report saved to {output}")
        print(f"✅ Report saved to: {output}")

    def evaluate_all_academic(
        self,
        sample_size: int = 100,
        confidence_level: float = 0.95,
        seed: Optional[int] = None,
        temperature: float = 0.0,
    ) -> AcademicEvaluationResults:
        """
        Run comprehensive academic evaluation with statistical rigor.

        Returns results with:
        - 95% Confidence intervals for all metrics (Wilson method)
        - Standard errors
        - Comparison to published baselines (GPT-4, Claude, etc.)
        - Reproducibility manifest with SHA256 hash
        - Export-ready format for LaTeX papers

        Args:
            sample_size: Number of questions per benchmark (default: 100)
            confidence_level: Confidence level for intervals (default: 0.95)
            seed: Random seed for reproducible sample selection (default: None)
            temperature: LLM temperature for deterministic outputs (default: 0.0)

        Returns:
            AcademicEvaluationResults with full statistical analysis

        Example:
            >>> results = evaluator.evaluate_all_academic(sample_size=500, seed=42)
            >>> print(f"MMLU: {results.mmlu_accuracy:.1%} "
            ...       f"(95% CI: [{results.mmlu_ci[0]:.1%}, {results.mmlu_ci[1]:.1%}])")
        """
        import time
        from datetime import datetime, timezone

        from .academic_baselines import compare_to_baselines
        from .export import generate_reproducibility_manifest
        from .statistical_metrics import calculate_standard_error, calculate_wilson_ci

        model_name = self.provider.model
        logger.info(f"Starting academic evaluation for {model_name}")

        print(f"\n{'='*60}")
        print(f"🎓 ACADEMIC EVALUATION: {model_name}")
        print(f"{'='*60}")
        print(f"Sample size: {sample_size} questions per benchmark")
        print(f"Confidence level: {confidence_level:.0%}")
        print()

        start_time = time.time()

        # Create benchmark runner with sample size and reproducibility settings
        runner = BenchmarkRunner(
            provider=self.provider,
            use_full_datasets=True,
            sample_size=sample_size,
            seed=seed,
            temperature=temperature,
        )

        # Run MMLU
        print("📚 Running MMLU...")
        mmlu_results = runner.run_mmlu_sample()
        mmlu_acc = float(mmlu_results["mmlu_accuracy"])
        mmlu_correct = int(mmlu_results["correct"])
        mmlu_n = int(mmlu_results["questions_tested"])
        mmlu_ci = calculate_wilson_ci(mmlu_correct, mmlu_n, confidence_level)
        mmlu_se = calculate_standard_error(mmlu_correct, mmlu_n)

        # Run TruthfulQA
        print("🎯 Running TruthfulQA...")
        tqa_results = runner.run_truthfulqa_sample()
        tqa_acc = float(tqa_results["truthfulness_score"])
        tqa_correct = int(tqa_results["correct"])
        tqa_n = int(tqa_results["questions_tested"])
        tqa_ci = calculate_wilson_ci(tqa_correct, tqa_n, confidence_level)
        tqa_se = calculate_standard_error(tqa_correct, tqa_n)

        # Run HellaSwag
        print("🧠 Running HellaSwag...")
        hs_results = runner.run_hellaswag_sample()
        hs_acc = float(hs_results["hellaswag_accuracy"])
        hs_correct = int(hs_results["correct"])
        hs_n = int(
            hs_results.get("scenarios_tested", hs_results.get("questions_tested", sample_size))
        )
        hs_ci = calculate_wilson_ci(hs_correct, hs_n, confidence_level)
        hs_se = calculate_standard_error(hs_correct, hs_n)

        elapsed = time.time() - start_time
        avg_accuracy = (mmlu_acc + tqa_acc + hs_acc) / 3

        # Compare to baselines
        baseline_comparison = compare_to_baselines(model_name, mmlu_acc, "mmlu")

        # Generate reproducibility manifest
        config = {
            "sample_size": sample_size,
            "confidence_level": confidence_level,
            "temperature": temperature,
            "random_seed": seed,
            "version": "2.3.0",
        }
        results_for_manifest = {
            "mmlu_accuracy": mmlu_acc,
            "truthfulqa_accuracy": tqa_acc,
            "hellaswag_accuracy": hs_acc,
        }
        manifest = generate_reproducibility_manifest(config, results_for_manifest)

        timestamp = datetime.now(timezone.utc).isoformat()

        # Print results
        print(f"\n{'='*60}")
        print(f"📊 ACADEMIC RESULTS: {model_name}")
        print(f"{'='*60}")
        print(
            f"MMLU:       {mmlu_acc:.1%} (95% CI: [{mmlu_ci[0]:.1%}, {mmlu_ci[1]:.1%}]) "
            f"SE: {mmlu_se:.3f}"
        )
        print(
            f"TruthfulQA: {tqa_acc:.1%} (95% CI: [{tqa_ci[0]:.1%}, {tqa_ci[1]:.1%}]) "
            f"SE: {tqa_se:.3f}"
        )
        print(
            f"HellaSwag:  {hs_acc:.1%} (95% CI: [{hs_ci[0]:.1%}, {hs_ci[1]:.1%}]) "
            f"SE: {hs_se:.3f}"
        )
        print(f"\nAverage:    {avg_accuracy:.1%}")
        print("\n📈 Baseline Comparison (MMLU):")
        print(f"   vs GPT-4: {baseline_comparison['comparisons']['gpt-4']['delta']:+.1%}")
        print(f"   Rank: {baseline_comparison['rank']}/{baseline_comparison['total_baselines']}")
        print(f"   Tier: {baseline_comparison['tier']}")
        print(f"\n⏱️  Elapsed time: {elapsed:.1f}s")
        print(f"🔐 Reproducibility hash: {manifest['evaluation_hash'][:24]}...")
        print(f"{'='*60}\n")

        logger.info(f"Academic evaluation completed for {model_name}")

        return AcademicEvaluationResults(
            model_name=model_name,
            mmlu_accuracy=mmlu_acc,
            mmlu_ci=mmlu_ci,
            mmlu_se=mmlu_se,
            mmlu_n=mmlu_n,
            truthfulqa_accuracy=tqa_acc,
            truthfulqa_ci=tqa_ci,
            truthfulqa_se=tqa_se,
            truthfulqa_n=tqa_n,
            hellaswag_accuracy=hs_acc,
            hellaswag_ci=hs_ci,
            hellaswag_se=hs_se,
            hellaswag_n=hs_n,
            average_accuracy=avg_accuracy,
            baseline_comparison=baseline_comparison,
            reproducibility_manifest=manifest,
            elapsed_time=elapsed,
            timestamp=timestamp,
            config=config,
        )


if __name__ == "__main__":
    # Quick demo with new provider architecture
    from .providers.ollama_provider import OllamaProvider

    provider = OllamaProvider(model="llama3.2:1b")
    evaluator = ModelEvaluator(provider=provider)
    results = evaluator.evaluate_all()
    evaluator.generate_report(results)
