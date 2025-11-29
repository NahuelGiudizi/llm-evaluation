"""
Main ModelEvaluator class for comprehensive LLM testing
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import ollama

from .metrics import PerformanceMetrics, QualityMetrics
from .benchmarks import BenchmarkRunner


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
    detailed_metrics: Dict[str, Any]


class ModelEvaluator:
    """
    Comprehensive LLM evaluation framework

    Args:
        model: Model name from Ollama library (default: llama3.2:1b)

    Example:
        >>> evaluator = ModelEvaluator(model="llama3.2:1b")
        >>> results = evaluator.evaluate_all()
        >>> print(f"Overall Score: {results.overall_score:.2f}")
    """

    def __init__(self, model: str = "llama3.2:1b"):
        self.model = model
        self.performance_metrics = PerformanceMetrics()
        self.quality_metrics = QualityMetrics()
        self.benchmark_runner = BenchmarkRunner(model)

    def chat(self, prompt: str) -> tuple[str, float]:
        """
        Send prompt to LLM and return response with timing

        Returns:
            (response, response_time_in_seconds)
        """
        start_time = time.time()
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        response_time = time.time() - start_time

        return response["message"]["content"], response_time

    def evaluate_performance(self, num_samples: int = 10) -> Dict[str, float]:
        """
        Evaluate performance metrics: response time, token efficiency

        Args:
            num_samples: Number of test prompts to run

        Returns:
            Dictionary with performance metrics
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

        response_times = []
        token_counts = []

        for prompt in test_prompts:
            response, resp_time = self.chat(prompt)
            response_times.append(resp_time)
            # Rough token estimate (4 chars per token)
            token_counts.append(len(response) / 4)

        return {
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "avg_tokens_per_response": sum(token_counts) / len(token_counts),
            "tokens_per_second": sum(token_counts) / sum(response_times),
        }

    def evaluate_quality(self, test_set: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        Evaluate quality metrics: accuracy, coherence, hallucination

        Args:
            test_set: List of {"prompt": str, "expected": str} dicts

        Returns:
            Dictionary with quality metrics
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

    def evaluate_all(self) -> EvaluationResults:
        """
        Run comprehensive evaluation across all metrics

        Returns:
            EvaluationResults object with all metrics
        """
        print(f"\n🔍 Evaluating {self.model}...")
        print("=" * 60)

        # Performance metrics
        print("\n📊 Performance Metrics...")
        perf_metrics = self.evaluate_performance()

        # Quality metrics
        print("✅ Quality Metrics...")
        quality_metrics = self.evaluate_quality()

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
            model_name=self.model,
            accuracy=quality_metrics["accuracy"],
            avg_response_time=perf_metrics["avg_response_time"],
            token_efficiency=perf_metrics["tokens_per_second"],
            hallucination_rate=quality_metrics["hallucination_rate"],
            coherence_score=quality_metrics["coherence_score"],
            overall_score=overall_score,
            detailed_metrics={"performance": perf_metrics, "quality": quality_metrics},
        )

        # Print summary
        print("\n" + "=" * 60)
        print(f"📋 EVALUATION SUMMARY: {self.model}")
        print("=" * 60)
        print(f"  Accuracy:          {results.accuracy:.1%}")
        print(f"  Avg Response Time: {results.avg_response_time:.2f}s")
        print(f"  Token Efficiency:  {results.token_efficiency:.1f} tokens/s")
        print(f"  Hallucination Rate: {results.hallucination_rate:.1%}")
        print(f"  Coherence Score:   {results.coherence_score:.1%}")
        print(f"  Overall Score:     {results.overall_score:.2f}/1.00")
        print("=" * 60 + "\n")

        return results

    def generate_report(self, results: EvaluationResults, output: str = "report.md"):
        """Generate markdown report from evaluation results"""
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

## Performance Details

```
{results.detailed_metrics['performance']}
```

## Quality Details

```
{results.detailed_metrics['quality']}
```

---
Generated by LLM Evaluator v0.1.0
"""

        with open(output, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"✅ Report saved to: {output}")


if __name__ == "__main__":
    # Quick demo
    evaluator = ModelEvaluator(model="llama3.2:1b")
    results = evaluator.evaluate_all()
    evaluator.generate_report(results)
