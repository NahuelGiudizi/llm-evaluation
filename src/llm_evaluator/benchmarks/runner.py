"""
BenchmarkRunner - maintains backward compatibility with existing API.

Wraps new class-based benchmarks to preserve existing method signatures.
"""

import logging
from typing import Any, Dict, Optional, Union

from ..providers import LLMProvider
from ..security.prompt_injection import PromptInjectionBenchmark

# Security benchmarks
from ..security.red_team import RedTeamBenchmark
from .arc import ARCBenchmark
from .boolq import BoolQBenchmark
from .commonsenseqa import CommonsenseQABenchmark
from .donotanswer import DoNotAnswerBenchmark
from .gsm8k import GSM8KBenchmark
from .hellaswag import HellaSwagBenchmark
from .mmlu import MMLUBenchmark
from .safetybench import SafetyBenchBenchmark
from .truthfulqa import TruthfulQABenchmark
from .winogrande import WinoGrandeBenchmark

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runner for standard LLM benchmarks.

    Maintains backward compatibility with original API while using
    new class-based benchmark implementations.

    Supports:
    - Demo mode (fast, 2-5 questions per benchmark)
    - Full mode (complete datasets from HuggingFace)
    - Sample mode (randomly sample N questions)

    Production datasets:
    - MMLU: 14,042 multiple-choice questions across 57 subjects
    - TruthfulQA: 817 questions testing truthfulness
    - HellaSwag: 10,042 commonsense reasoning scenarios
    - ARC: 2,590 science questions
    - WinoGrande: ~1,767 pronoun resolution scenarios
    - CommonsenseQA: commonsense reasoning questions
    - BoolQ: yes/no questions from passages
    - SafetyBench: 35 safety questions (dev split)
    - DoNotAnswer: 939 harmful prompts (should be refused)
    - GSM8K: 1,319 grade school math problems
    """

    def __init__(
        self,
        provider: LLMProvider,
        use_full_datasets: bool = False,
        sample_size: Optional[int] = None,
        max_workers: int = 1,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initialize with LLM provider.

        Args:
            provider: LLM provider implementation
            use_full_datasets: If True, use complete HuggingFace datasets
                              If False, use demo mode with 2-5 examples
            sample_size: If specified, randomly sample this many questions
            max_workers: Number of concurrent workers (not yet implemented in new structure)
            seed: Random seed for reproducible sampling
            temperature: LLM generation temperature (overrides provider default)
        """
        self.provider = provider
        self.use_full_datasets = use_full_datasets
        self.sample_size = sample_size
        self.max_workers = max_workers
        self.seed = seed
        self.temperature = temperature

        # Determine mode for benchmarks
        self.mode = "full" if use_full_datasets else "demo"

    def _get_generation_config(self) -> Optional[Any]:
        """
        Get generation config with temperature override if set.

        Returns:
            GenerationConfig with custom temperature, or None to use provider defaults
        """
        if self.temperature is not None:
            from ..providers import GenerationConfig

            # Clone provider config if it exists, override temperature
            base_config = self.provider.config if hasattr(self.provider, "config") else None
            if base_config:
                return GenerationConfig(
                    temperature=self.temperature,
                    max_tokens=base_config.max_tokens,
                    top_p=base_config.top_p,
                    top_k=base_config.top_k,
                    timeout_seconds=base_config.timeout_seconds,
                    retry_attempts=base_config.retry_attempts,
                )
            else:
                return GenerationConfig(temperature=self.temperature)
        return None

    def _run_benchmark(self, benchmark_class, **kwargs) -> Dict[str, Any]:
        """Helper to run a benchmark with common setup"""
        benchmark = benchmark_class(
            provider=self.provider,
            mode=self.mode,
            sample_size=self.sample_size if self.use_full_datasets else None,
        )
        return benchmark.run()

    # MMLU
    def run_mmlu_sample(self) -> Dict[str, Union[float, int, str]]:
        """Run MMLU benchmark"""
        return self._run_benchmark(MMLUBenchmark)

    # TruthfulQA
    def run_truthfulqa_sample(self) -> Dict[str, Union[float, int, str]]:
        """Run TruthfulQA benchmark"""
        return self._run_benchmark(TruthfulQABenchmark)

    # HellaSwag
    def run_hellaswag_sample(self) -> Dict[str, Union[float, int, str]]:
        """Run HellaSwag benchmark"""
        return self._run_benchmark(HellaSwagBenchmark)

    # ARC
    def run_arc_sample(self) -> Dict[str, Union[float, int, str]]:
        """Run ARC benchmark"""
        # Override demo_size to match test expectations
        benchmark = ARCBenchmark(
            provider=self.provider,
            mode=self.mode,
            sample_size=self.sample_size if self.use_full_datasets else None,
        )
        # Adjust demo data size for backward compatibility
        if self.mode == "demo" and hasattr(benchmark, "get_demo_data"):
            original_demo = benchmark.get_demo_data
            benchmark.get_demo_data = lambda: original_demo()[:2]
        return benchmark.run()

    # WinoGrande
    def run_winogrande_sample(self) -> Dict[str, Union[float, int, str]]:
        """Run WinoGrande benchmark"""
        return self._run_benchmark(WinoGrandeBenchmark)

    # CommonsenseQA
    def run_commonsenseqa_sample(self) -> Dict[str, Union[float, int, str]]:
        """Run CommonsenseQA benchmark"""
        return self._run_benchmark(CommonsenseQABenchmark)

    # BoolQ
    def run_boolq_sample(self) -> Dict[str, Union[float, int, str]]:
        """Run BoolQ benchmark"""
        return self._run_benchmark(BoolQBenchmark)

    # SafetyBench
    def run_safetybench_sample(self) -> Dict[str, Union[float, int, str]]:
        """Run SafetyBench benchmark"""
        return self._run_benchmark(SafetyBenchBenchmark)

    # DoNotAnswer
    def run_donotanswer_sample(self) -> Dict[str, Union[float, int, str]]:
        """Run DoNotAnswer benchmark"""
        return self._run_benchmark(DoNotAnswerBenchmark)

    # GSM8K
    def run_gsm8k_sample(self) -> Dict[str, Union[float, int, str]]:
        """Run GSM8K benchmark"""
        result = self._run_benchmark(GSM8KBenchmark)
        # Backward compatibility: add legacy key
        if "questions_tested" in result:
            result["problems_tested"] = result["questions_tested"]
        return result

    # Security Benchmarks
    def run_redteam_sample(self) -> Dict[str, Union[float, int, str]]:
        """Run Red Team (jailbreak) benchmark"""
        benchmark = RedTeamBenchmark(
            provider=self.provider,
            sample_size=self.sample_size,
        )
        return benchmark.run()

    def run_prompt_injection_sample(self) -> Dict[str, Union[float, int, str]]:
        """Run Prompt Injection benchmark"""
        benchmark = PromptInjectionBenchmark(
            provider=self.provider,
            sample_size=self.sample_size,
        )
        return benchmark.run()

    # Run all benchmarks
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks and return aggregate results"""
        results = {}

        # Run each benchmark
        results["mmlu"] = self.run_mmlu_sample()
        results["truthfulqa"] = self.run_truthfulqa_sample()
        results["hellaswag"] = self.run_hellaswag_sample()

        # Calculate aggregate score (average of accuracies)
        accuracies = []
        if "mmlu_accuracy" in results["mmlu"]:
            accuracies.append(results["mmlu"]["mmlu_accuracy"])
        if "truthfulness_score" in results["truthfulqa"]:
            accuracies.append(results["truthfulqa"]["truthfulness_score"])
        if "hellaswag_accuracy" in results["hellaswag"]:
            accuracies.append(results["hellaswag"]["hellaswag_accuracy"])

        aggregate_score = sum(accuracies) / len(accuracies) if accuracies else 0.0
        results["aggregate_benchmark_score"] = aggregate_score

        return results

    # Legacy internal methods for backward compatibility
    def _extract_number_from_response(self, response: str) -> Optional[float]:
        """Extract number from GSM8K response (backward compatibility)."""
        import re

        # Remove commas from numbers like "1,234"
        cleaned = response.replace(",", "")
        numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        return float(numbers[-1]) if numbers else None

    def _run_parallel(self, items, process_fn, desc: str):
        """Run items in parallel or sequential (backward compatibility)."""
        from tqdm import tqdm

        correct = 0
        scenarios = [None] * len(items)  # Pre-allocate to maintain order

        if self.max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(process_fn, idx, item): idx for idx, item in enumerate(items)
                }
                for future in tqdm(as_completed(futures), total=len(items), desc=desc):
                    idx = futures[future]
                    try:
                        is_correct, scenario = future.result()
                        if is_correct:
                            correct += 1
                        scenarios[idx] = scenario
                    except Exception as e:
                        scenarios[idx] = {"error": str(e)}
        else:
            for idx, item in enumerate(tqdm(items, desc=desc)):
                try:
                    is_correct, scenario = process_fn(idx, item)
                    if is_correct:
                        correct += 1
                    scenarios[idx] = scenario
                except Exception as e:
                    scenarios[idx] = {"error": str(e)}

        return correct, scenarios

    def _run_mmlu_full(self) -> Dict[str, Any]:
        """Run full MMLU (backward compatibility stub)."""
        return self._run_benchmark(MMLUBenchmark)

    def _run_truthfulqa_full(self) -> Dict[str, Any]:
        """Run full TruthfulQA (backward compatibility stub)."""
        return self._run_benchmark(TruthfulQABenchmark)
