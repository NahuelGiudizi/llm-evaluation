"""
CLI tool for LLM Evaluation Suite

Provides command-line interface for running evaluations, comparisons, and visualizations.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import click

from llm_evaluator import ModelEvaluator
from llm_evaluator.benchmarks import BenchmarkRunner
from llm_evaluator.evaluator import AcademicEvaluationResults
from llm_evaluator.export import export_to_latex, generate_bibtex
from llm_evaluator.providers import LLMProvider
from llm_evaluator.providers.ollama_provider import OllamaProvider
from llm_evaluator.system_info import collect_system_info

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

try:
    from llm_evaluator.providers.deepseek_provider import DeepSeekProvider

    HAS_DEEPSEEK = True
except ImportError:
    HAS_DEEPSEEK = False

try:
    from llm_evaluator.providers.groq_provider import GroqProvider

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    from llm_evaluator.providers.together_provider import TogetherProvider

    HAS_TOGETHER = True
except ImportError:
    HAS_TOGETHER = False

try:
    from llm_evaluator.providers.fireworks_provider import FireworksProvider

    HAS_FIREWORKS = True
except ImportError:
    HAS_FIREWORKS = False

from llm_evaluator.providers.cached_provider import CachedProvider

# Version
__version__ = "2.1.0"


def detect_provider_from_env() -> Tuple[Optional[str], Optional[str]]:
    """
    Auto-detect provider and model from environment variables.

    Returns:
        Tuple of (provider_name, suggested_model) or (None, None)
    """
    if os.environ.get("OPENAI_API_KEY"):
        return ("openai", "gpt-4o-mini")
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ("anthropic", "claude-3-5-sonnet-20241022")
    if os.environ.get("DEEPSEEK_API_KEY"):
        return ("deepseek", "deepseek-chat")
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY"):
        return ("huggingface", "meta-llama/Llama-2-7b-chat-hf")

    # Check if Ollama is running - try multiple methods
    ollama_model = _detect_ollama()
    if ollama_model:
        return ("ollama", ollama_model)

    return (None, None)


def _detect_ollama() -> Optional[str]:
    """
    Detect if Ollama is running and get an available model.

    Returns:
        Model name if Ollama is available, None otherwise
    """
    import socket
    import urllib.error
    import urllib.request

    # First, quick socket check
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 11434))
        sock.close()
        if result != 0:
            return None
    except Exception:
        return None

    # Ollama port is open, try to get models list
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as response:
            import json

            data = json.loads(response.read().decode("utf-8"))
            models = data.get("models", [])
            if models:
                # Prefer smaller/faster models for quick evaluation
                preferred_order = [
                    "qwen2.5:0.5b",
                    "tinyllama",
                    "phi3:mini",
                    "llama3.2:1b",
                    "gemma:2b",
                    "mistral:7b",
                ]
                for preferred in preferred_order:
                    for model in models:
                        model_name = str(model.get("name", ""))
                        if preferred in model_name.lower():
                            return model_name
                # Return first available model
                first_model = models[0]
                return str(first_model.get("name", "llama3.2:1b"))
    except Exception:
        pass

    # Fallback: try version endpoint to confirm Ollama is running
    try:
        req = urllib.request.Request("http://localhost:11434/api/version")
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                return "llama3.2:1b"  # Default model
    except Exception:
        pass

    return None


def _get_ollama_models() -> list[str]:
    """Get list of available Ollama models."""
    import json
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
            models = data.get("models", [])
            return [str(m.get("name", "")) for m in models if m.get("name")]
    except Exception:
        return []


def echo_success(msg: str) -> None:
    """Print success message in green"""
    click.echo(click.style(f"✅ {msg}", fg="green"))


def echo_error(msg: str) -> None:
    """Print error message in red"""
    click.echo(click.style(f"❌ {msg}", fg="red"), err=True)


def echo_warning(msg: str) -> None:
    """Print warning message in yellow"""
    click.echo(click.style(f"⚠️  {msg}", fg="yellow"))


def echo_info(msg: str) -> None:
    """Print info message in blue"""
    click.echo(click.style(f"ℹ️  {msg}", fg="blue"))


@click.group()
@click.version_option(version=__version__, prog_name="llm-eval")
def cli() -> None:
    """
    🚀 LLM Evaluation Suite - Command Line Interface

    Evaluate, compare, and visualize LLM performance across multiple models and benchmarks.

    Examples:
        llm-eval run --model llama3.2:1b
        llm-eval compare --models llama3.2,mistral:7b
        llm-eval benchmark --model gpt-3.5-turbo --provider openai
    """
    pass


def create_provider(
    model: str,
    provider_type: str,
    cache: bool = False,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMProvider:
    """Create provider instance based on type.

    Args:
        model: Model name to use
        provider_type: Provider type (ollama, openai, anthropic, etc.)
        cache: Whether to wrap provider with caching layer
        base_url: Custom base URL for OpenAI-compatible APIs (vLLM, LM Studio, Together.ai, etc.)
        api_key: Optional API key override (uses env var if not provided)
    """
    base_provider: LLMProvider

    if provider_type == "ollama":
        base_provider = OllamaProvider(model=model, base_url=base_url)
    elif provider_type == "openai":
        if not HAS_OPENAI:
            echo_error("OpenAI provider not installed. Run: pip install openai")
            sys.exit(1)
        base_provider = OpenAIProvider(model=model, base_url=base_url, api_key=api_key)
    elif provider_type == "anthropic":
        if not HAS_ANTHROPIC:
            echo_error("Anthropic provider not installed. Run: pip install anthropic")
            sys.exit(1)
        base_provider = AnthropicProvider(model=model)
    elif provider_type == "huggingface":
        if not HAS_HUGGINGFACE:
            echo_error("HuggingFace provider not installed. Run: pip install huggingface-hub")
            sys.exit(1)
        base_provider = HuggingFaceProvider(model=model)
    elif provider_type == "deepseek":
        if not HAS_DEEPSEEK:
            echo_error("DeepSeek provider not installed. Run: pip install openai")
            sys.exit(1)
        base_provider = DeepSeekProvider(model=model)
    elif provider_type == "groq":
        if not HAS_GROQ:
            echo_error("Groq provider requires openai package. Run: pip install openai")
            sys.exit(1)
        base_provider = GroqProvider(model=model, api_key=api_key)
    elif provider_type == "together":
        if not HAS_TOGETHER:
            echo_error("Together provider requires openai package. Run: pip install openai")
            sys.exit(1)
        base_provider = TogetherProvider(model=model, api_key=api_key)
    elif provider_type == "fireworks":
        if not HAS_FIREWORKS:
            echo_error("Fireworks provider requires openai package. Run: pip install openai")
            sys.exit(1)
        base_provider = FireworksProvider(model=model, api_key=api_key)
    elif provider_type == "auto":
        # Auto-detect from environment
        detected_provider, detected_model = detect_provider_from_env()
        if detected_provider and detected_model:
            echo_info(f"Auto-detected provider: {detected_provider} (model: {detected_model})")
            return create_provider(
                detected_model if model == "auto" else model, detected_provider, cache
            )
        else:
            echo_error("No provider detected. Set an API key or start Ollama.")
            echo_info(
                "Supported env vars: OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, HF_TOKEN"
            )
            sys.exit(1)
    else:
        echo_error(f"Unknown provider: {provider_type}")
        sys.exit(1)

    # Wrap with cache if requested
    if cache:
        return CachedProvider(base_provider)
    return base_provider


@cli.command()
@click.option("--model", "-m", default=None, help="Model name (auto-detected if not set)")
@click.option("--sample-size", "-s", type=int, default=20, help="Sample size (default: 20)")
@click.option("--output", "-o", default=None, help="Output file (optional)")
def quick(model: Optional[str], sample_size: int, output: Optional[str]) -> None:
    """
    🚀 Quick evaluation with zero configuration!

    Auto-detects your provider from environment variables:
    - OPENAI_API_KEY → Uses OpenAI (gpt-4o-mini)
    - ANTHROPIC_API_KEY → Uses Anthropic (claude-3-5-sonnet)
    - DEEPSEEK_API_KEY → Uses DeepSeek (deepseek-chat)
    - HF_TOKEN → Uses HuggingFace
    - Ollama running → Uses Ollama (llama3.2:1b)

    Examples:
        llm-eval quick                      # Auto-detect everything
        llm-eval quick --model gpt-4o       # Use specific model
        llm-eval quick -s 50                # Larger sample size
    """
    click.echo("\n" + "=" * 50)
    click.echo(click.style("🚀 LLM QUICK EVALUATION", fg="cyan", bold=True))
    click.echo("=" * 50)

    # Auto-detect provider
    detected_provider, detected_model = detect_provider_from_env()

    if not detected_provider:
        echo_error("No provider detected!")
        click.echo("\n📋 To use quick evaluation, set one of these environment variables:")
        click.echo("   • OPENAI_API_KEY    → For GPT models")
        click.echo("   • ANTHROPIC_API_KEY → For Claude models")
        click.echo("   • DEEPSEEK_API_KEY  → For DeepSeek models")
        click.echo("   • HF_TOKEN          → For HuggingFace models")
        click.echo("   • Or start Ollama   → ollama serve")
        sys.exit(1)

    # Use provided model or detected one
    use_model: str = model if model else (detected_model if detected_model else "")
    if not use_model:
        echo_error("No model detected")
        sys.exit(1)

    echo_success(f"Provider: {detected_provider}")
    echo_success(f"Model: {use_model}")
    echo_success(f"Sample size: {sample_size}")

    # Show available models for Ollama if using default
    if detected_provider == "ollama" and not model:
        available = _get_ollama_models()
        if available and len(available) > 1:
            other_models = [m for m in available if m != use_model][:5]
            if other_models:
                click.echo(click.style("   💡 Other models: ", fg="blue") + ", ".join(other_models))
                click.echo(click.style("   💡 Use --model <name> to choose", fg="blue"))

    click.echo("\n⏳ Starting evaluation...")

    # Collect system info
    sys_info = collect_system_info()

    # Create provider with caching
    if not detected_provider:
        echo_error("No provider detected")
        sys.exit(1)
    llm_provider = create_provider(use_model, detected_provider, cache=True)

    if not llm_provider.is_available():
        echo_error(f"Provider {detected_provider} is not responding")
        sys.exit(1)

    # Run benchmarks
    runner = BenchmarkRunner(provider=llm_provider, use_full_datasets=True, sample_size=sample_size)

    click.echo("\n📊 Running benchmarks...")

    results = {}
    start_time = time.time()

    with click.progressbar(["mmlu", "truthfulqa", "hellaswag"], label="Progress") as benchmarks:
        for bench in benchmarks:
            if bench == "mmlu":
                results["mmlu"] = runner.run_mmlu_sample()
            elif bench == "truthfulqa":
                results["truthfulqa"] = runner.run_truthfulqa_sample()
            elif bench == "hellaswag":
                results["hellaswag"] = runner.run_hellaswag_sample()

    total_time = time.time() - start_time

    # Display results
    click.echo("\n" + "=" * 50)
    click.echo(click.style("📊 RESULTS", fg="green", bold=True))
    click.echo("=" * 50)

    click.echo(f"\n  🎯 MMLU:       {results.get('mmlu', {}).get('mmlu_accuracy', 0):.1%}")
    click.echo(f"  🎯 TruthfulQA: {results.get('truthfulqa', {}).get('truthfulness_score', 0):.1%}")
    click.echo(f"  🎯 HellaSwag:  {results.get('hellaswag', {}).get('hellaswag_accuracy', 0):.1%}")

    # Calculate overall
    scores: list[float] = [
        float(results.get("mmlu", {}).get("mmlu_accuracy", 0) or 0),
        float(results.get("truthfulqa", {}).get("truthfulness_score", 0) or 0),
        float(results.get("hellaswag", {}).get("hellaswag_accuracy", 0) or 0),
    ]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    click.echo(f"\n  📈 Overall:    {avg_score:.1%}")

    # Show timing and system info
    click.echo("\n" + "-" * 50)
    click.echo(click.style("🖥️  TEST ENVIRONMENT", fg="cyan"))
    click.echo("-" * 50)
    click.echo(f"  Model:    {use_model}")
    click.echo(f"  Provider: {detected_provider}")
    click.echo(f"  Samples:  {sample_size} per benchmark ({sample_size * 3} total)")
    click.echo(f"  Time:     {total_time:.1f}s ({total_time / (sample_size * 3):.2f}s/question)")
    click.echo("")
    click.echo(f"  CPU:      {sys_info.cpu_model}")
    click.echo(f"  RAM:      {sys_info.ram_total_gb:.0f} GB")
    if sys_info.gpu_info:
        gpu_str = sys_info.gpu_info
        if sys_info.gpu_vram_gb:
            gpu_str += f" ({sys_info.gpu_vram_gb:.0f} GB)"
        click.echo(f"  GPU:      {gpu_str}")
    click.echo(f"  OS:       {sys_info.os_name} {sys_info.os_version}")
    if sys_info.ollama_version:
        click.echo(f"  Ollama:   v{sys_info.ollama_version}")

    # Save if output specified
    if output:
        output_data = {
            "model": use_model,
            "provider": detected_provider,
            "sample_size": sample_size,
            "total_time_seconds": total_time,
            "results": results,
            "system_info": sys_info.to_dict(),
        }
        Path(output).write_text(json.dumps(output_data, indent=2, default=str))
        echo_success(f"Results saved to: {output}")

    # Cache stats
    if isinstance(llm_provider, CachedProvider):
        stats = llm_provider.get_cache_stats()
        click.echo(f"\n  💾 Cache: {stats['hit_rate_percent']:.0f}% hit rate")

    click.echo("\n" + "=" * 50)
    click.echo("✨ Evaluation complete!")
    click.echo("=" * 50 + "\n")


@cli.command()
@click.option("--model", "-m", default="llama3.2:1b", help="Model name")
@click.option(
    "--provider",
    "-p",
    default="ollama",
    type=click.Choice(
        [
            "ollama",
            "openai",
            "anthropic",
            "huggingface",
            "deepseek",
            "groq",
            "together",
            "fireworks",
            "auto",
        ]
    ),
    help="Provider type",
)
@click.option(
    "--base-url",
    "-u",
    default=None,
    help="Custom base URL for OpenAI-compatible APIs (vLLM, LM Studio, Azure)",
)
@click.option("--api-key", "-k", default=None, help="API key (uses env var if not provided)")
@click.option("--cache/--no-cache", default=True, help="Enable caching")
@click.option("--output", "-o", default="evaluation_report.md", help="Output file")
def run(
    model: str,
    provider: str,
    base_url: Optional[str],
    api_key: Optional[str],
    cache: bool,
    output: str,
) -> None:
    """
    Run full evaluation on a single model

    Examples:
        llm-eval run --model llama3.2:1b --provider ollama
        llm-eval run --model gpt-4o --provider openai
        llm-eval run --model my-model --provider openai --base-url http://localhost:8000/v1
    """
    click.echo(f"🚀 Running evaluation on {model} ({provider})")
    if base_url:
        click.echo(f"   Custom endpoint: {base_url}")

    # Create provider
    llm_provider = create_provider(model, provider, cache, base_url=base_url, api_key=api_key)

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
    type=click.Choice(
        [
            "ollama",
            "openai",
            "anthropic",
            "huggingface",
            "deepseek",
            "groq",
            "together",
            "fireworks",
        ]
    ),
    help="Provider type (same for all models)",
)
@click.option(
    "--base-url",
    "-u",
    default=None,
    help="Custom base URL for OpenAI-compatible APIs (vLLM, LM Studio, Azure)",
)
@click.option("--api-key", "-k", default=None, help="API key (uses env var if not provided)")
@click.option("--cache/--no-cache", default=True, help="Enable caching")
@click.option("--output", "-o", default="comparison.json", help="Output JSON file")
def compare(
    models: str,
    provider: str,
    base_url: Optional[str],
    api_key: Optional[str],
    cache: bool,
    output: str,
) -> None:
    """
    Compare multiple models side-by-side

    Examples:
        llm-eval compare --models llama3.2:1b,mistral:7b --provider ollama
        llm-eval compare --models gpt-4o,gpt-4o-mini --provider openai
        llm-eval compare --models model1,model2 --base-url http://localhost:8000/v1 --provider openai
    """
    model_list = [m.strip() for m in models.split(",")]

    click.echo(f"🔄 Comparing {len(model_list)} models: {', '.join(model_list)}")
    if base_url:
        click.echo(f"   Custom endpoint: {base_url}")

    results = {}

    for model in model_list:
        click.echo(f"\n📊 Evaluating {model}...")

        llm_provider = create_provider(model, provider, cache, base_url=base_url, api_key=api_key)

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
    type=click.Choice(
        [
            "ollama",
            "openai",
            "anthropic",
            "huggingface",
            "deepseek",
            "groq",
            "together",
            "fireworks",
        ]
    ),
    help="Provider type",
)
@click.option(
    "--base-url",
    "-u",
    default=None,
    help="Custom base URL for OpenAI-compatible APIs (vLLM, LM Studio, Azure)",
)
@click.option("--api-key", "-k", default=None, help="API key (uses env var if not provided)")
@click.option(
    "--benchmarks",
    "-b",
    default="mmlu,truthfulqa,hellaswag",
    help="Comma-separated benchmarks: mmlu,truthfulqa,hellaswag,arc,winogrande,commonsenseqa,boolq,safetybench,donotanswer,gsm8k",
)
@click.option("--sample-size", "-s", type=int, help="Sample size (None = demo mode)")
@click.option("--full", is_flag=True, help="Run full benchmarks (~132,000 questions)")
@click.option("--cache/--no-cache", default=True, help="Enable caching")
@click.option("--output", "-o", default="benchmark_results.json", help="Output file")
@click.option(
    "--workers",
    "-w",
    type=int,
    default=1,
    help="Number of parallel workers (default: 1 = sequential). Higher values give 5-10x speedup.",
)
def benchmark(
    model: str,
    provider: str,
    base_url: Optional[str],
    api_key: Optional[str],
    benchmarks: str,
    sample_size: Optional[int],
    full: bool,
    cache: bool,
    output: str,
    workers: int,
) -> None:
    """
    Run specific benchmarks on a model

    Examples:
        llm-eval benchmark --model llama3.2:1b --benchmarks mmlu
        llm-eval benchmark --model gpt-3.5-turbo --provider openai --sample-size 100
        llm-eval benchmark --model llama3.2:1b --full  # Warning: takes hours!
        llm-eval benchmark --model my-model --base-url http://localhost:8000/v1 --provider openai
        llm-eval benchmark --model llama3.2:1b --workers 4  # 4x parallel speedup
    """
    click.echo(f"📊 Running benchmarks on {model} ({provider})")
    if base_url:
        click.echo(f"   Custom endpoint: {base_url}")
    if workers > 1:
        click.echo(f"   ⚡ Parallel mode: {workers} workers")

    if full and not click.confirm("⚠️  Full benchmarks take 2-8 hours. Continue?"):
        click.echo("Aborted.")
        sys.exit(0)

    # Create provider
    llm_provider = create_provider(model, provider, cache, base_url=base_url, api_key=api_key)

    if not llm_provider.is_available():
        click.echo(f"❌ Provider not available. Is {provider} running?", err=True)
        sys.exit(1)

    # Setup benchmark runner
    use_full = full or (sample_size is not None)
    runner = BenchmarkRunner(
        provider=llm_provider,
        use_full_datasets=use_full,
        sample_size=None if full else sample_size,
        max_workers=workers,
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
        elif bench == "arc":
            results["arc"] = runner.run_arc_sample()
        elif bench == "winogrande":
            results["winogrande"] = runner.run_winogrande_sample()
        elif bench == "commonsenseqa":
            results["commonsenseqa"] = runner.run_commonsenseqa_sample()
        elif bench == "boolq":
            results["boolq"] = runner.run_boolq_sample()
        elif bench == "safetybench":
            results["safetybench"] = runner.run_safetybench_sample()
        elif bench == "donotanswer":
            results["donotanswer"] = runner.run_donotanswer_sample()
        elif bench == "gsm8k":
            results["gsm8k"] = runner.run_gsm8k_sample()
        else:
            click.echo(f"⚠️  Unknown benchmark: {bench}", err=True)
            continue

        # Print result
        accuracy_keys = {
            "mmlu": "mmlu_accuracy",
            "truthfulqa": "truthfulness_score",
            "hellaswag": "hellaswag_accuracy",
            "arc": "arc_accuracy",
            "winogrande": "winogrande_accuracy",
            "commonsenseqa": "commonsenseqa_accuracy",
            "boolq": "boolq_accuracy",
            "safetybench": "safetybench_accuracy",
            "donotanswer": "donotanswer_refusal_rate",
            "gsm8k": "gsm8k_accuracy",
        }
        accuracy_key = accuracy_keys.get(bench, f"{bench}_accuracy")
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
def visualize(results_file: str, output: str) -> None:
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
def providers() -> None:
    """List available providers and their status"""
    click.echo("\n🔌 Available Providers:\n")

    # Auto-detection status
    detected_provider, detected_model = detect_provider_from_env()
    if detected_provider:
        echo_success(f"Auto-detected: {detected_provider} ({detected_model})")
        click.echo("")

    providers_status = [
        ("ollama", True, "Local LLMs (llama3.2, mistral, etc.)"),
        ("openai", HAS_OPENAI, "GPT-3.5, GPT-4, GPT-4o (pip install openai)"),
        ("anthropic", HAS_ANTHROPIC, "Claude 3/3.5 (pip install anthropic)"),
        ("deepseek", HAS_DEEPSEEK, "DeepSeek-V3, DeepSeek-R1 (pip install openai)"),
        ("groq", HAS_GROQ, "Ultra-fast Llama, Mixtral (pip install openai)"),
        ("together", HAS_TOGETHER, "100+ open models (pip install openai)"),
        ("fireworks", HAS_FIREWORKS, "Optimized open models (pip install openai)"),
        ("huggingface", HAS_HUGGINGFACE, "Inference API (pip install huggingface-hub)"),
    ]

    for name, available, description in providers_status:
        status = "✅" if available else "❌"
        click.echo(f"  {status} {name:<15} - {description}")

    click.echo("\n📋 Environment Variables:")
    env_vars = [
        (
            "OPENAI_API_KEY",
            (
                os.environ.get("OPENAI_API_KEY", "")[:8] + "..."
                if os.environ.get("OPENAI_API_KEY")
                else "Not set"
            ),
        ),
        (
            "ANTHROPIC_API_KEY",
            (
                os.environ.get("ANTHROPIC_API_KEY", "")[:8] + "..."
                if os.environ.get("ANTHROPIC_API_KEY")
                else "Not set"
            ),
        ),
        (
            "DEEPSEEK_API_KEY",
            (
                os.environ.get("DEEPSEEK_API_KEY", "")[:8] + "..."
                if os.environ.get("DEEPSEEK_API_KEY")
                else "Not set"
            ),
        ),
        (
            "GROQ_API_KEY",
            (
                os.environ.get("GROQ_API_KEY", "")[:8] + "..."
                if os.environ.get("GROQ_API_KEY")
                else "Not set"
            ),
        ),
        (
            "TOGETHER_API_KEY",
            (
                os.environ.get("TOGETHER_API_KEY", "")[:8] + "..."
                if os.environ.get("TOGETHER_API_KEY")
                else "Not set"
            ),
        ),
        (
            "FIREWORKS_API_KEY",
            (
                os.environ.get("FIREWORKS_API_KEY", "")[:8] + "..."
                if os.environ.get("FIREWORKS_API_KEY")
                else "Not set"
            ),
        ),
        (
            "HF_TOKEN",
            os.environ.get("HF_TOKEN", "")[:8] + "..." if os.environ.get("HF_TOKEN") else "Not set",
        ),
    ]

    for var, value in env_vars:
        status = "✅" if "Not set" not in value else "❌"
        click.echo(f"  {status} {var:<20} {value}")

    click.echo("")


@cli.command()
@click.option("--model", "-m", default="llama3.2:1b", help="Model name")
@click.option(
    "--provider",
    "-p",
    default="ollama",
    type=click.Choice(
        [
            "ollama",
            "openai",
            "anthropic",
            "huggingface",
            "deepseek",
            "groq",
            "together",
            "fireworks",
            "auto",
        ]
    ),
    help="Provider type",
)
@click.option("--sample-size", "-s", type=int, default=100, help="Sample size for benchmarks")
@click.option("--cache/--no-cache", default=True, help="Enable caching")
@click.option("--output-latex", type=click.Path(), help="Output LaTeX table file")
@click.option("--output-bibtex", type=click.Path(), help="Output BibTeX citations file")
@click.option("--output-json", "-o", default="academic_results.json", help="Output JSON file")
@click.option(
    "--compare-baselines/--no-compare-baselines",
    default=True,
    help="Compare against published baselines",
)
def academic(
    model: str,
    provider: str,
    sample_size: int,
    cache: bool,
    output_latex: Optional[str],
    output_bibtex: Optional[str],
    output_json: str,
    compare_baselines: bool,
) -> None:
    """
    Run academic-quality evaluation with statistical rigor

    Produces results suitable for academic papers with:
    - 95% confidence intervals (Wilson method)
    - Comparison against published baselines
    - Error analysis and calibration metrics
    - LaTeX tables and BibTeX citations

    Examples:
        llm-eval academic --model llama3.2:1b --output-latex results.tex
        llm-eval academic --model gpt-4 --provider openai --output-bibtex citations.bib
    """
    click.echo(f"🎓 Running academic evaluation on {model} ({provider})")
    click.echo(f"   Sample size: {sample_size}")
    click.echo(f"   Compare baselines: {compare_baselines}")

    # Create provider
    llm_provider = create_provider(model, provider, cache)

    if not llm_provider.is_available():
        click.echo(f"❌ Provider not available. Is {provider} running?", err=True)
        sys.exit(1)

    # Run academic evaluation
    evaluator = ModelEvaluator(provider=llm_provider)

    click.echo("\n📊 Running benchmarks with statistical analysis...")

    try:
        results: AcademicEvaluationResults = evaluator.evaluate_all_academic(
            sample_size=sample_size,
        )
    except Exception as e:
        click.echo(f"❌ Error during evaluation: {e}", err=True)
        sys.exit(1)

    # Print results summary
    click.echo("\n" + "=" * 60)
    click.echo("📊 ACADEMIC EVALUATION RESULTS")
    click.echo("=" * 60)

    click.echo(f"\n📈 MMLU Accuracy: {results.mmlu_accuracy:.1%}")
    if results.mmlu_ci:
        click.echo(f"   95% CI: [{results.mmlu_ci[0]:.1%}, {results.mmlu_ci[1]:.1%}]")

    click.echo(f"\n📈 TruthfulQA Score: {results.truthfulqa_accuracy:.1%}")
    if results.truthfulqa_ci:
        click.echo(f"   95% CI: [{results.truthfulqa_ci[0]:.1%}, {results.truthfulqa_ci[1]:.1%}]")

    click.echo(f"\n📈 HellaSwag Accuracy: {results.hellaswag_accuracy:.1%}")
    if results.hellaswag_ci:
        click.echo(f"   95% CI: [{results.hellaswag_ci[0]:.1%}, {results.hellaswag_ci[1]:.1%}]")

    # Baseline comparison
    if compare_baselines and results.baseline_comparison:
        click.echo("\n" + "-" * 40)
        click.echo("📊 BASELINE COMPARISON")
        click.echo("-" * 40)
        for baseline_name, comparison in results.baseline_comparison.items():
            diff = comparison.get("difference", 0)
            sign = "+" if diff > 0 else ""
            click.echo(f"   vs {baseline_name}: {sign}{diff:.1%}")

    # Save JSON results
    results_dict = {
        "model": model,
        "provider": provider,
        "sample_size": sample_size,
        "mmlu_accuracy": results.mmlu_accuracy,
        "mmlu_ci": results.mmlu_ci,
        "truthfulqa_accuracy": results.truthfulqa_accuracy,
        "truthfulqa_ci": results.truthfulqa_ci,
        "hellaswag_accuracy": results.hellaswag_accuracy,
        "hellaswag_ci": results.hellaswag_ci,
        "baseline_comparison": results.baseline_comparison,
        "reproducibility_manifest": results.reproducibility_manifest,
    }
    Path(output_json).write_text(json.dumps(results_dict, indent=2, default=str))
    click.echo(f"\n✅ Results saved to: {output_json}")

    # Export LaTeX if requested
    if output_latex:
        # Prepare results in format expected by export_to_latex
        latex_results = {
            model: {
                "mmlu": results.mmlu_accuracy,
                "mmlu_ci": results.mmlu_ci,
                "truthfulqa": results.truthfulqa_accuracy,
                "truthfulqa_ci": results.truthfulqa_ci,
                "hellaswag": results.hellaswag_accuracy,
                "hellaswag_ci": results.hellaswag_ci,
            }
        }
        latex_content = export_to_latex(latex_results)
        Path(output_latex).write_text(latex_content)
        click.echo(f"📄 LaTeX table saved to: {output_latex}")

    # Export BibTeX if requested
    if output_bibtex:
        eval_metadata = {
            "version": "2.0.0",
            "date": results.timestamp,
            "author": "LLM Evaluation Suite",
            "models_evaluated": [model],
            "n_samples": sample_size,
        }
        bibtex_content = generate_bibtex(eval_metadata)
        Path(output_bibtex).write_text(bibtex_content)
        click.echo(f"📚 BibTeX citations saved to: {output_bibtex}")

    # Show cache stats
    if cache and isinstance(llm_provider, CachedProvider):
        stats = llm_provider.get_cache_stats()
        click.echo(f"\n💾 Cache Stats: {stats['hit_rate_percent']:.1f}% hit rate")


@cli.command()
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
)
@click.option(
    "--port",
    default=8888,
    type=int,
    help="Port to run on (default: 8888)",
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="Don't open browser automatically",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable hot reload for development",
)
def dashboard(host: str, port: int, no_browser: bool, reload: bool) -> None:
    """
    🌐 Launch the Web Dashboard

    Start the interactive web dashboard for running evaluations,
    viewing results, and comparing models.

    Examples:
        llm-eval dashboard
        llm-eval dashboard --port 9000
        llm-eval dashboard --no-browser
    """
    try:
        from llm_evaluator.dashboard import run_dashboard
    except ImportError as e:
        echo_error("Dashboard dependencies not installed!")
        echo_info("Install with: pip install llm-benchmark-toolkit[dashboard]")
        echo_info(f"Missing: {e}")
        sys.exit(1)

    click.echo(click.style("\n⚡ Starting LLM Benchmark Dashboard...\n", fg="cyan", bold=True))

    run_dashboard(
        host=host,
        port=port,
        open_browser=not no_browser,
        reload=reload,
    )


@cli.command()
@click.argument("models", nargs=-1, required=True)
@click.option(
    "--benchmark",
    "-b",
    default="mmlu,truthfulqa,hellaswag",
    help="Benchmark(s) to run: mmlu,truthfulqa,hellaswag,arc,winogrande,commonsenseqa,boolq,safetybench,donotanswer,gsm8k",
)
@click.option(
    "--samples",
    "-s",
    type=int,
    default=100,
    help="Samples per benchmark (default: 100)",
)
@click.option(
    "--provider",
    "-p",
    default="ollama",
    help="Provider(s), comma-separated if different per model",
)
@click.option("--cache/--no-cache", default=True, help="Enable caching")
@click.option(
    "--output-dir",
    "-o",
    default=None,
    help="Output directory (default: ~/.llm-benchmark/outputs)",
)
def vs(
    models: tuple[str, ...],
    benchmark: str,
    samples: int,
    provider: str,
    cache: bool,
    output_dir: Optional[str],
) -> None:
    """
    🥊 Run same benchmark on multiple models sequentially.

    Each model produces its own standard JSON result file.
    Great for comparing different models on the same task.

    Examples:
        llm-eval vs llama3.2:1b mistral:7b
        llm-eval vs llama3.2:1b mistral:7b -b mmlu -s 50
        llm-eval vs gpt-4o-mini claude-3.5-sonnet -p openai,anthropic
    """
    from datetime import datetime

    click.echo("\n" + "═" * 60)
    click.echo(click.style(f"🥊 Sequential Evaluation: {len(models)} models", fg="cyan", bold=True))
    click.echo("═" * 60)

    # Parse providers (can be comma-separated for different models)
    providers = [p.strip() for p in provider.split(",")]
    if len(providers) == 1:
        providers = providers * len(models)  # Same provider for all
    elif len(providers) != len(models):
        echo_error(f"Provider count ({len(providers)}) must match model count ({len(models)})")
        sys.exit(1)

    # Parse benchmarks
    benchmark_list = [b.strip() for b in benchmark.split(",")]

    # Setup output directory
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = Path.home() / ".llm-benchmark" / "outputs"
    out_path.mkdir(parents=True, exist_ok=True)

    # Store results for summary
    all_results: dict[str, dict[str, Any]] = {}
    output_files: list[str] = []

    click.echo(f"\n📋 Benchmarks: {', '.join(b.upper() for b in benchmark_list)}")
    click.echo(f"📊 Samples: {samples} per benchmark")
    click.echo(f"📁 Output: {out_path}\n")

    total_start = time.time()

    for i, (model, prov) in enumerate(zip(models, providers), 1):
        click.echo("-" * 60)
        click.echo(click.style(f"📊 Model {i}/{len(models)}: {model}", fg="white", bold=True))
        click.echo(f"   Provider: {prov}")
        click.echo("-" * 60)

        model_start = time.time()

        try:
            # Create provider
            llm_provider = create_provider(model, prov, cache)

            if not llm_provider.is_available():
                echo_error(f"Provider {prov} not available for {model}, skipping")
                all_results[model] = {"error": True}
                continue

            # Setup benchmark runner
            runner = BenchmarkRunner(
                provider=llm_provider, use_full_datasets=True, sample_size=samples
            )

            results: dict[str, Any] = {}

            for bench in benchmark_list:
                click.echo(f"\n   🎯 Running {bench.upper()}...")

                if bench == "mmlu":
                    results["mmlu"] = runner.run_mmlu_sample()
                elif bench == "truthfulqa":
                    results["truthfulqa"] = runner.run_truthfulqa_sample()
                elif bench == "hellaswag":
                    results["hellaswag"] = runner.run_hellaswag_sample()
                elif bench == "arc":
                    results["arc"] = runner.run_arc_sample()
                elif bench == "winogrande":
                    results["winogrande"] = runner.run_winogrande_sample()
                elif bench == "commonsenseqa":
                    results["commonsenseqa"] = runner.run_commonsenseqa_sample()
                elif bench == "boolq":
                    results["boolq"] = runner.run_boolq_sample()
                elif bench == "safetybench":
                    results["safetybench"] = runner.run_safetybench_sample()
                elif bench == "donotanswer":
                    results["donotanswer"] = runner.run_donotanswer_sample()
                elif bench == "gsm8k":
                    results["gsm8k"] = runner.run_gsm8k_sample()
                else:
                    click.echo(f"   ⚠️  Unknown benchmark: {bench}")
                    continue

                # Show result
                accuracy_keys = {
                    "mmlu": "mmlu_accuracy",
                    "truthfulqa": "truthfulness_score",
                    "hellaswag": "hellaswag_accuracy",
                    "arc": "arc_accuracy",
                    "winogrande": "winogrande_accuracy",
                    "commonsenseqa": "commonsenseqa_accuracy",
                    "boolq": "boolq_accuracy",
                    "safetybench": "safetybench_accuracy",
                    "donotanswer": "donotanswer_refusal_rate",
                    "gsm8k": "gsm8k_accuracy",
                }
                accuracy_key = accuracy_keys.get(bench, f"{bench}_accuracy")
                if accuracy_key in results[bench]:
                    score = results[bench][accuracy_key]
                    click.echo(f"   ✅ {bench.upper()}: {score:.1%}")

            # Calculate average
            scores = []
            for bench_name, bench_data in results.items():
                if isinstance(bench_data, dict):
                    score = (
                        bench_data.get("mmlu_accuracy")
                        or bench_data.get("truthfulness_score")
                        or bench_data.get("hellaswag_accuracy")
                        or bench_data.get("arc_accuracy")
                        or bench_data.get("winogrande_accuracy")
                        or bench_data.get("commonsenseqa_accuracy")
                        or bench_data.get("boolq_accuracy")
                        or bench_data.get("safetybench_accuracy")
                        or bench_data.get("donotanswer_refusal_rate")
                    )
                    if score is not None:
                        scores.append(score)

            avg_score = sum(scores) / len(scores) if scores else 0

            # Save to JSON (standard format)
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model.replace(':', '_').replace('/', '_')}"
            output_file = out_path / f"{run_id}.json"

            sys_info = collect_system_info()
            complete_data = {
                "run_id": run_id,
                "model": model,
                "provider": prov,
                "benchmarks": benchmark_list,
                "sample_size": samples,
                "status": "completed",
                "started_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
                "results": results,
                "system_info": sys_info.to_dict(),
            }

            with open(output_file, "w") as f:
                json.dump(complete_data, f, indent=2, default=str)

            output_files.append(output_file.name)

            # Store for summary
            model_scores: dict[str, float] = {}
            for bench_name, bench_data in results.items():
                if isinstance(bench_data, dict):
                    score = (
                        bench_data.get("mmlu_accuracy")
                        or bench_data.get("truthfulness_score")
                        or bench_data.get("hellaswag_accuracy")
                    )
                    if score is not None:
                        model_scores[bench_name] = score
            model_scores["average"] = avg_score
            all_results[model] = model_scores

            model_time = time.time() - model_start
            click.echo(
                f"\n   ✅ Completed: {avg_score:.1%} avg "
                f"({model_time:.0f}s) → {output_file.name}"
            )

        except Exception as e:
            echo_error(f"Failed to evaluate {model}: {e}")
            all_results[model] = {"error": True}

    # Summary
    total_time = time.time() - total_start

    click.echo("\n" + "═" * 60)
    click.echo(click.style("📊 SUMMARY", fg="green", bold=True))
    click.echo("═" * 60)

    # Find best model
    valid_results: dict[str, dict[str, Any]] = {
        k: v for k, v in all_results.items() if "error" not in v
    }

    if valid_results:
        # Build header
        benchmarks_in_results: set[str] = set()
        for result_scores in valid_results.values():
            benchmarks_in_results.update(str(k) for k in result_scores.keys() if k != "average")

        header = f"{'Model':<25}"
        for bench in sorted(benchmarks_in_results):
            header += f" {bench.upper():>12}"
        header += f" {'AVG':>10}"
        click.echo(header)
        click.echo("-" * len(header))

        best_model = max(valid_results.items(), key=lambda x: float(x[1].get("average", 0)))

        for model_name, result_scores in valid_results.items():
            is_best = model_name == best_model[0]
            row = f"{model_name:<25}"
            for bench in sorted(benchmarks_in_results):
                bench_score = result_scores.get(bench)
                if bench_score is not None:
                    row += f" {float(bench_score):>11.1%}"
                else:
                    row += f" {'-':>12}"
            avg = float(result_scores.get("average", 0))
            row += f" {avg:>9.1%}"
            if is_best and len(valid_results) > 1:
                row += " ← Best"
            click.echo(row)

        click.echo("-" * len(header))

        # Show difference if 2 models
        if len(valid_results) == 2:
            models_list = list(valid_results.keys())
            diff = valid_results[models_list[0]].get("average", 0) - valid_results[
                models_list[1]
            ].get("average", 0)
            sign = "+" if diff > 0 else ""
            click.echo(f"\nDifference: {sign}{diff:.1%}")

    click.echo(f"\n⏱️  Total time: {total_time/60:.1f} minutes")
    click.echo(f"📁 Results saved to: {out_path}")

    click.echo(click.style("\n💡 View detailed comparison: llm-eval dashboard", fg="blue"))
    click.echo("═" * 60 + "\n")


@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    "export_format",
    type=click.Choice(["json", "csv", "latex", "bibtex", "all"]),
    default="all",
    help="Export format (default: all)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=".",
    help="Output directory (default: current directory)",
)
def export(results_file: str, export_format: str, output_dir: str) -> None:
    """
    📤 Export evaluation results to various formats

    Convert evaluation results to JSON, CSV, LaTeX tables, or BibTeX citations
    for use in papers, reports, or data analysis pipelines.

    Examples:
        llm-eval export results.json --format all
        llm-eval export results.json --format latex -o ./paper/
        llm-eval export results.json --format csv
        llm-eval export results.json --format bibtex
    """
    import csv
    import io
    from pathlib import Path as PathLib

    from llm_evaluator.export import (
        export_to_latex,
        generate_bibtex,
        generate_references_bibtex,
        generate_reproducibility_manifest,
    )

    results_path = PathLib(results_file)
    out_path = PathLib(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load results
    try:
        with open(results_path) as f:
            data = json.load(f)
    except Exception as e:
        echo_error(f"Failed to load results file: {e}")
        sys.exit(1)

    run_id = data.get("run_id", results_path.stem)
    model_name = data.get("model", "Unknown Model")
    results_data = data.get("results", {})

    click.echo(f"\n📤 Exporting results for: {model_name}")
    click.echo(f"   Run ID: {run_id}")
    click.echo(f"   Output directory: {out_path}\n")

    exported_files = []

    # JSON Export (with manifest)
    if export_format in ("json", "all"):
        json_file = out_path / f"{run_id}_export.json"
        export_data = {
            "results": data,
            "manifest": generate_reproducibility_manifest(
                config=data.get("config", {}),
                results=results_data,
            ),
        }
        with open(json_file, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
        exported_files.append(("JSON", json_file))
        echo_success(f"JSON exported: {json_file}")

    # CSV Export
    if export_format in ("csv", "all"):
        csv_file = out_path / f"{run_id}_results.csv"
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "Benchmark",
                "Score (%)",
                "Correct",
                "Total",
                "CI Lower",
                "CI Upper",
                "Time (s)",
            ]
        )

        for bench_name, bench_data in results_data.items():
            if not isinstance(bench_data, dict):
                continue

            # Extract score
            score = None
            for key in [
                "score",
                "accuracy",
                f"{bench_name}_accuracy",
                "mmlu_accuracy",
                "truthfulness_score",
                "hellaswag_accuracy",
            ]:
                if key in bench_data:
                    score = bench_data[key]
                    break
            if score is None and "correct" in bench_data:
                total = bench_data.get("questions_tested") or bench_data.get("scenarios_tested", 0)
                if total > 0:
                    score = bench_data["correct"] / total

            ci = bench_data.get("confidence_interval") or bench_data.get("ci")
            ci_lower = f"{ci[0] * 100:.2f}" if ci else ""
            ci_upper = f"{ci[1] * 100:.2f}" if ci else ""

            writer.writerow(
                [
                    bench_name.upper(),
                    f"{score * 100:.2f}" if score else "",
                    bench_data.get("correct", ""),
                    bench_data.get("questions_tested") or bench_data.get("scenarios_tested", ""),
                    ci_lower,
                    ci_upper,
                    (
                        f"{bench_data.get('elapsed_time', ''):.2f}"
                        if bench_data.get("elapsed_time")
                        else ""
                    ),
                ]
            )

        with open(csv_file, "w", newline="") as f:
            f.write(output.getvalue())
        exported_files.append(("CSV", csv_file))
        echo_success(f"CSV exported: {csv_file}")

    # LaTeX Export
    if export_format in ("latex", "all"):
        latex_file = out_path / f"{run_id}_table.tex"

        # Prepare results for LaTeX export
        formatted_results: dict[str, dict[str, Any]] = {model_name: {}}
        benchmarks_used = []

        for bench_name, bench_data in results_data.items():
            if not isinstance(bench_data, dict):
                continue

            benchmarks_used.append(bench_name.lower())
            score = None
            for key in [
                "score",
                "accuracy",
                f"{bench_name}_accuracy",
                "mmlu_accuracy",
                "truthfulness_score",
                "hellaswag_accuracy",
            ]:
                if key in bench_data:
                    score = bench_data[key]
                    break
            if score is None and "correct" in bench_data:
                total = bench_data.get("questions_tested") or bench_data.get("scenarios_tested", 0)
                if total > 0:
                    score = bench_data["correct"] / total

            if score is not None:
                formatted_results[model_name][bench_name.lower()] = score
                ci = bench_data.get("confidence_interval") or bench_data.get("ci")
                if ci:
                    formatted_results[model_name][f"{bench_name.lower()}_ci"] = tuple(ci)

        latex_content = export_to_latex(
            results=formatted_results,
            include_ci=True,
            caption=f"Benchmark Results for {model_name}",
            label=f"tab:{run_id}",
            benchmarks=benchmarks_used if benchmarks_used else None,
        )

        with open(latex_file, "w") as f:
            f.write(latex_content)
        exported_files.append(("LaTeX", latex_file))
        echo_success(f"LaTeX table exported: {latex_file}")

    # BibTeX Export
    if export_format in ("bibtex", "all"):
        bib_file = out_path / f"{run_id}_references.bib"

        eval_metadata = {
            "version": "2.1.0",
            "date": data.get("started_at", "")[:10] if data.get("started_at") else "",
            "models_evaluated": [model_name],
            "n_samples": data.get("sample_size", "N/A"),
            "github_url": "https://github.com/NahuelGiudizi/llm-evaluation",
        }

        bibtex_content = f"""% Citation for this evaluation
{generate_bibtex(eval_metadata)}

% Standard benchmark references
{generate_references_bibtex()}
"""

        with open(bib_file, "w") as f:
            f.write(bibtex_content)
        exported_files.append(("BibTeX", bib_file))
        echo_success(f"BibTeX exported: {bib_file}")

    # Summary
    click.echo("\n" + "─" * 50)
    click.echo(click.style("📋 Export Summary", fg="green", bold=True))
    click.echo("─" * 50)

    for fmt, filepath in exported_files:
        click.echo(f"  ✅ {fmt:<10} → {filepath}")

    click.echo("─" * 50)
    click.echo(f"\n💡 {len(exported_files)} file(s) exported to {out_path}\n")


@cli.command()
def list_runs() -> None:
    """
    📋 List all saved evaluation runs

    Shows all evaluation results saved in the default output directory.
    """
    from pathlib import Path as PathLib

    outputs_dir = PathLib.home() / ".llm-benchmark" / "outputs"

    if not outputs_dir.exists():
        echo_warning("No runs found. Run an evaluation first!")
        echo_info(f"Outputs directory: {outputs_dir}")
        return

    json_files = list(outputs_dir.glob("*.json"))

    if not json_files:
        echo_warning("No runs found in outputs directory.")
        echo_info(f"Outputs directory: {outputs_dir}")
        return

    click.echo(f"\n📋 Saved Evaluation Runs ({len(json_files)} total)\n")
    click.echo(f"{'Run ID':<45} {'Model':<25} {'Status':<12} {'Date'}")
    click.echo("─" * 100)

    for json_file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(json_file) as f:
                data = json.load(f)

            run_id = data.get("run_id", json_file.stem)
            model = data.get("model", "Unknown")[:24]
            status = data.get("status", "unknown")
            started = data.get("started_at", "")[:19]

            status_color = (
                "green" if status == "completed" else "yellow" if status == "running" else "red"
            )
            status_display = click.style(f"{status:<12}", fg=status_color)

            click.echo(f"{run_id:<45} {model:<25} {status_display} {started}")
        except Exception:
            click.echo(f"{json_file.stem:<45} {'Error loading':<25} {'error':<12}")

    click.echo("─" * 100)
    click.echo(f"\n📁 Directory: {outputs_dir}")
    click.echo("💡 Export a run: llm-eval export <run_id>.json --format all\n")


if __name__ == "__main__":
    cli()
