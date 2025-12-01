"""
CLI tool for LLM Evaluation Suite

Provides command-line interface for running evaluations, comparisons, and visualizations.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from llm_evaluator import ModelEvaluator
from llm_evaluator.benchmarks import BenchmarkRunner
from llm_evaluator.providers import LLMProvider
from llm_evaluator.providers.ollama_provider import OllamaProvider

# Import optional providers
try:
    from llm_evaluator.providers.openai_provider import OpenAIProvider

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from llm_evaluator.providers.anthropic_provider import AnthropicProvider

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from llm_evaluator.providers.huggingface_provider import HuggingFaceProvider

    HAS_HUGGINGFACE = True
except ImportError:
    HAS_HUGGINGFACE = False

from llm_evaluator.providers.cached_provider import CachedProvider


@click.group()
@click.version_option(version="0.2.0", prog_name="llm-eval")
def cli():
    """
    🚀 LLM Evaluation Suite - Command Line Interface

    Evaluate, compare, and visualize LLM performance across multiple models and benchmarks.

    Examples:
        llm-eval run --model llama3.2:1b
        llm-eval compare --models llama3.2,mistral:7b
        llm-eval benchmark --model gpt-3.5-turbo --provider openai
    """
    pass


def create_provider(model: str, provider_type: str, cache: bool = False) -> LLMProvider:
    """Create provider instance based on type"""
    base_provider: LLMProvider

    if provider_type == "ollama":
        base_provider = OllamaProvider(model=model)
    elif provider_type == "openai":
        if not HAS_OPENAI:
            click.echo("❌ OpenAI provider not installed. Run: pip install openai", err=True)
            sys.exit(1)
        base_provider = OpenAIProvider(model=model)
    elif provider_type == "anthropic":
        if not HAS_ANTHROPIC:
            click.echo("❌ Anthropic provider not installed. Run: pip install anthropic", err=True)
            sys.exit(1)
        base_provider = AnthropicProvider(model=model)
    elif provider_type == "huggingface":
        if not HAS_HUGGINGFACE:
            click.echo(
                "❌ HuggingFace provider not installed. Run: pip install huggingface-hub", err=True
            )
            sys.exit(1)
        base_provider = HuggingFaceProvider(model=model)
    else:
        click.echo(f"❌ Unknown provider: {provider_type}", err=True)
        sys.exit(1)

    # Wrap with cache if requested
    if cache:
        return CachedProvider(base_provider)
    return base_provider


@cli.command()
@click.option("--model", "-m", default="llama3.2:1b", help="Model name")
@click.option(
    "--provider",
    "-p",
    default="ollama",
    type=click.Choice(["ollama", "openai", "anthropic", "huggingface"]),
    help="Provider type",
)
@click.option("--cache/--no-cache", default=True, help="Enable caching")
@click.option("--output", "-o", default="evaluation_report.md", help="Output file")
def run(model: str, provider: str, cache: bool, output: str):
    """
    Run full evaluation on a single model

    Example:
        llm-eval run --model llama3.2:1b --provider ollama
    """
    click.echo(f"🚀 Running evaluation on {model} ({provider})")

    # Create provider
    llm_provider = create_provider(model, provider, cache)

    # Check availability
    if not llm_provider.is_available():
        click.echo(f"❌ Provider not available. Is {provider} running?", err=True)
        sys.exit(1)

    # Run evaluation
    evaluator = ModelEvaluator(provider=llm_provider)

    with click.progressbar(length=3, label="Evaluating") as bar:
        bar.update(1)
        results = evaluator.evaluate_all()
        bar.update(2)

    # Generate report
    evaluator.generate_report(results, output)

    # Print summary
    click.echo("\n✅ Evaluation complete!")
    click.echo(f"📊 Overall Score: {results.overall_score:.1%}")
    click.echo(f"📄 Report saved to: {output}")

    # Show cache stats if caching enabled
    if cache and isinstance(llm_provider, CachedProvider):
        stats = llm_provider.get_cache_stats()
        click.echo("\n💾 Cache Stats:")
        click.echo(f"   Hit rate: {stats['hit_rate_percent']:.1f}%")
        click.echo(f"   Hits: {stats['hits']} | Misses: {stats['misses']}")


@cli.command()
@click.option("--models", "-m", required=True, help="Comma-separated model names")
@click.option(
    "--provider",
    "-p",
    default="ollama",
    type=click.Choice(["ollama", "openai", "anthropic", "huggingface"]),
    help="Provider type (same for all models)",
)
@click.option("--cache/--no-cache", default=True, help="Enable caching")
@click.option("--output", "-o", default="comparison.json", help="Output JSON file")
def compare(models: str, provider: str, cache: bool, output: str):
    """
    Compare multiple models side-by-side

    Example:
        llm-eval compare --models llama3.2:1b,mistral:7b --provider ollama
    """
    model_list = [m.strip() for m in models.split(",")]

    click.echo(f"🔄 Comparing {len(model_list)} models: {', '.join(model_list)}")

    results = {}

    for model in model_list:
        click.echo(f"\n📊 Evaluating {model}...")

        llm_provider = create_provider(model, provider, cache)

        if not llm_provider.is_available():
            click.echo(f"⚠️  {model} not available, skipping", err=True)
            continue

        evaluator = ModelEvaluator(provider=llm_provider)
        eval_results = evaluator.evaluate_all()

        results[model] = {
            "overall_score": eval_results.overall_score,
            "accuracy": eval_results.accuracy,
            "avg_response_time": eval_results.avg_response_time,
            "coherence_score": eval_results.coherence_score,
        }

        click.echo(f"   Overall: {eval_results.overall_score:.1%}")

    # Save results
    Path(output).write_text(json.dumps(results, indent=2))

    # Print comparison table
    click.echo("\n📊 Comparison Results:")
    click.echo(f"{'Model':<30} {'Score':<10} {'Accuracy':<10} {'Speed (s)':<12}")
    click.echo("=" * 70)

    for model_name, data in sorted(
        results.items(), key=lambda x: x[1]["overall_score"], reverse=True
    ):
        click.echo(
            f"{model_name:<30} "
            f"{data['overall_score']:<10.1%} "
            f"{data['accuracy']:<10.1%} "
            f"{data['avg_response_time']:<12.2f}"
        )

    click.echo(f"\n✅ Results saved to: {output}")


@cli.command()
@click.option("--model", "-m", default="llama3.2:1b", help="Model name")
@click.option(
    "--provider",
    "-p",
    default="ollama",
    type=click.Choice(["ollama", "openai", "anthropic", "huggingface"]),
    help="Provider type",
)
@click.option(
    "--benchmarks",
    "-b",
    default="mmlu,truthfulqa,hellaswag",
    help="Comma-separated benchmark names",
)
@click.option("--sample-size", "-s", type=int, help="Sample size (None = demo mode)")
@click.option("--full", is_flag=True, help="Run full benchmarks (24,901 questions)")
@click.option("--cache/--no-cache", default=True, help="Enable caching")
@click.option("--output", "-o", default="benchmark_results.json", help="Output file")
def benchmark(
    model: str,
    provider: str,
    benchmarks: str,
    sample_size: Optional[int],
    full: bool,
    cache: bool,
    output: str,
):
    """
    Run specific benchmarks on a model

    Examples:
        llm-eval benchmark --model llama3.2:1b --benchmarks mmlu
        llm-eval benchmark --model gpt-3.5-turbo --provider openai --sample-size 100
        llm-eval benchmark --model llama3.2:1b --full  # Warning: takes hours!
    """
    click.echo(f"📊 Running benchmarks on {model} ({provider})")

    if full and not click.confirm("⚠️  Full benchmarks take 2-8 hours. Continue?"):
        click.echo("Aborted.")
        sys.exit(0)

    # Create provider
    llm_provider = create_provider(model, provider, cache)

    if not llm_provider.is_available():
        click.echo(f"❌ Provider not available. Is {provider} running?", err=True)
        sys.exit(1)

    # Setup benchmark runner
    use_full = full or (sample_size is not None)
    runner = BenchmarkRunner(
        provider=llm_provider, use_full_datasets=use_full, sample_size=None if full else sample_size
    )

    # Parse benchmarks
    benchmark_list = [b.strip() for b in benchmarks.split(",")]

    results = {}

    for bench in benchmark_list:
        click.echo(f"\n🎯 Running {bench.upper()}...")

        if bench == "mmlu":
            results["mmlu"] = runner.run_mmlu_sample()
        elif bench == "truthfulqa":
            results["truthfulqa"] = runner.run_truthfulqa_sample()
        elif bench == "hellaswag":
            results["hellaswag"] = runner.run_hellaswag_sample()
        else:
            click.echo(f"⚠️  Unknown benchmark: {bench}", err=True)
            continue

        # Print result
        accuracy_key = f"{bench}_accuracy" if bench != "truthfulqa" else "truthfulness_score"
        if accuracy_key in results[bench]:
            click.echo(f"   Accuracy: {results[bench][accuracy_key]:.1%}")

    # Save results
    Path(output).write_text(json.dumps(results, indent=2))
    click.echo(f"\n✅ Results saved to: {output}")

    # Show cache stats
    if cache and isinstance(llm_provider, CachedProvider):
        stats = llm_provider.get_cache_stats()
        click.echo(f"\n💾 Cache Stats: {stats['hit_rate_percent']:.1f}% hit rate")


@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--output", "-o", default="visualization.html", help="Output HTML file")
def visualize(results_file: str, output: str):
    """
    Generate interactive visualizations from results

    Example:
        llm-eval visualize comparison.json --output dashboard.html
    """
    click.echo(f"📈 Generating visualizations from {results_file}")

    # TODO: Implement visualization generation
    # This would use the visualizations.py module to create charts
    # data = json.loads(Path(results_file).read_text())

    click.echo("⚠️  Visualization feature coming soon!")
    click.echo("For now, use: from llm_evaluator.visualizations import EvaluationVisualizer")


@cli.command()
def providers():
    """List available providers and their status"""
    click.echo("🔌 Available Providers:\n")

    providers_status = [
        ("ollama", True, "Local LLMs (llama3.2, mistral, etc.)"),
        ("openai", HAS_OPENAI, "GPT-3.5, GPT-4 (pip install openai)"),
        ("anthropic", HAS_ANTHROPIC, "Claude 3/3.5 (pip install anthropic)"),
        ("huggingface", HAS_HUGGINGFACE, "Inference API (pip install huggingface-hub)"),
    ]

    for name, available, description in providers_status:
        status = "✅" if available else "❌"
        click.echo(f"{status} {name:<15} - {description}")


@cli.command("quick")
@click.argument("model", default="llama3.2:1b")
@click.option(
    "--provider",
    "-p",
    default="ollama",
    type=click.Choice(["ollama", "openai", "anthropic", "huggingface"]),
    help="Provider type",
)
@click.option(
    "--samples",
    "-n",
    default=10,
    help="Number of samples per benchmark (default: 10 for fast CI, use 100+ for production)",
)
def quick_benchmark(model: str, provider: str, samples: int):
    """
    Quick benchmark with configurable sample size

    Examples:
        llm-eval quick llama3.2:1b                    # 10 samples (fast, ~1 min)
        llm-eval quick llama3.2:1b -n 50              # 50 samples (~5 min)
        llm-eval quick gpt-3.5-turbo -p openai -n 100 # 100 samples (~10 min)
    """
    click.echo(f"\n🚀 Quick Benchmark: {model} ({provider})")
    click.echo(f"📊 Samples per benchmark: {samples}")
    click.echo("=" * 50)

    # Create provider
    llm_provider = create_provider(model, provider, cache=True)

    if not llm_provider.is_available():
        click.echo(f"❌ Provider not available. Is {provider} running?", err=True)
        sys.exit(1)

    click.echo(f"✅ {model} is ready!\n")

    # Run with configurable sample size
    runner = BenchmarkRunner(provider=llm_provider, use_full_datasets=True, sample_size=samples)

    results = {}

    # Run each benchmark
    click.echo(f"📚 Running MMLU ({samples} questions)...")
    results["mmlu"] = runner.run_mmlu_sample()

    click.echo(f"📖 Running TruthfulQA ({samples} questions)...")
    results["truthfulqa"] = runner.run_truthfulqa_sample()

    click.echo(f"🧠 Running HellaSwag ({samples} questions)...")
    results["hellaswag"] = runner.run_hellaswag_sample()

    # Print results
    click.echo("\n" + "=" * 50)
    click.echo(f"📊 Results for {model}:")
    click.echo("=" * 50)

    mmlu_acc = results["mmlu"].get("mmlu_accuracy", 0)
    tqa_score = results["truthfulqa"].get("truthfulness_score", 0)
    hs_acc = results["hellaswag"].get("hellaswag_accuracy", 0)

    click.echo(f"MMLU:       {mmlu_acc:.1%}")
    click.echo(f"TruthfulQA: {tqa_score:.1%}")
    click.echo(f"HellaSwag:  {hs_acc:.1%}")
    click.echo("=" * 50)

    # Save results
    output_file = f"benchmark_{model.replace(':', '_')}.json"
    Path(output_file).write_text(json.dumps(results, indent=2))
    click.echo(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    cli()
