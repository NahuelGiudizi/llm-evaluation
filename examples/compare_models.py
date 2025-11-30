"""
Multi-Model Comparison Tool
============================
Compare multiple LLM models side-by-side with comprehensive benchmarking
and automatic report generation.

Perfect for:
- Choosing the right model for your use case
- A/B testing model configurations
- Performance optimization analysis
- Blog posts and presentations
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_evaluator import ModelEvaluator, quick_comparison
from llm_evaluator.benchmarks import BenchmarkRunner, DATASETS_AVAILABLE
from llm_evaluator.providers.ollama_provider import OllamaProvider
from llm_evaluator.providers import GenerationConfig


def print_header(title: str) -> None:
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def evaluate_model(
    model: str,
    use_full_datasets: bool = False,
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on a single model
    
    Args:
        model: Ollama model name
        use_full_datasets: Use real HuggingFace datasets
        sample_size: Number of questions to sample
        
    Returns:
        Dictionary with all evaluation metrics
    """
    print(f"\n🔍 Evaluating {model}...")
    
    # Configure provider
    config = GenerationConfig(
        temperature=0.1,  # Low for reproducibility
        max_tokens=200,
        timeout_seconds=30
    )
    provider = OllamaProvider(model=model, config=config)
    
    results: Dict[str, Any] = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "mode": "full" if use_full_datasets else "demo"
    }
    
    try:
        # 1. Basic quality and performance metrics
        print(f"   📊 Running basic evaluation...")
        evaluator = ModelEvaluator(provider=provider)
        eval_results = evaluator.evaluate_all()
        
        results.update({
            "accuracy": eval_results.accuracy,
            "avg_response_time": eval_results.avg_response_time,
            "token_efficiency": eval_results.token_efficiency,
            "hallucination_rate": eval_results.hallucination_rate,
            "coherence_score": eval_results.coherence_score,
            "overall_score": eval_results.overall_score
        })
        
        # 2. Benchmarks
        print(f"   🎯 Running benchmarks...")
        runner = BenchmarkRunner(
            provider=provider,
            use_full_datasets=use_full_datasets,
            sample_size=sample_size if use_full_datasets else None
        )
        
        benchmark_results = runner.run_all_benchmarks()
        results.update(benchmark_results)
        
        print(f"   ✅ Complete!")
        return results
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results["error"] = str(e)
        return results


def compare_models(
    models: List[str],
    use_full_datasets: bool = False,
    sample_size: int = 100,
    output_dir: str = "outputs/comparisons"
) -> Dict[str, Any]:
    """
    Compare multiple models and generate reports
    
    Args:
        models: List of Ollama model names
        use_full_datasets: Use real datasets (slower but accurate)
        sample_size: Questions per benchmark
        output_dir: Where to save reports
        
    Returns:
        Dictionary with all results
    """
    print_header("🏆 MULTI-MODEL COMPARISON")
    
    print(f"\n📋 Configuration:")
    print(f"   Models: {', '.join(models)}")
    print(f"   Mode: {'Real datasets' if use_full_datasets else 'Demo mode'}")
    if use_full_datasets:
        print(f"   Sample size: {sample_size} questions per benchmark")
    print(f"   Output: {output_dir}/")
    
    # Evaluate each model
    all_results = {}
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model}")
        results = evaluate_model(model, use_full_datasets, sample_size)
        all_results[model] = results
    
    # Generate comparison reports
    print_header("📊 GENERATING REPORTS")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. JSON report
    json_file = output_path / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"   💾 JSON report: {json_file}")
    
    # 2. Markdown report
    md_file = output_path / "comparison_report.md"
    generate_markdown_report(all_results, md_file)
    print(f"   📝 Markdown report: {md_file}")
    
    # 3. Visual comparison (if possible)
    try:
        viz_results = {
            model: {
                "mmlu": data.get("mmlu_accuracy", 0),
                "truthful_qa": data.get("truthfulqa_score", 0),
                "hellaswag": data.get("hellaswag_accuracy", 0),
                "accuracy": data.get("accuracy", 0),
                "coherence": data.get("coherence_score", 0),
                "avg_response_time": data.get("avg_response_time", 0),
                "tokens_per_second": data.get("token_efficiency", 0)
            }
            for model, data in all_results.items()
        }
        quick_comparison(viz_results, output_dir=str(output_path))
        print(f"   📈 Visualizations: {output_path}/")
    except Exception as e:
        print(f"   ⚠️  Visualizations skipped: {e}")
    
    return all_results


def generate_markdown_report(results: Dict[str, Any], output_file: Path) -> None:
    """Generate a markdown comparison report"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# LLM Model Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary table
        f.write("## 📊 Summary\n\n")
        f.write("| Model | Overall Score | MMLU | TruthfulQA | HellaSwag | Avg Time | Hallucination |\n")
        f.write("|-------|---------------|------|------------|-----------|----------|---------------|\n")
        
        for model, data in results.items():
            if "error" in data:
                f.write(f"| {model} | ❌ Error | - | - | - | - | - |\n")
                continue
                
            overall = data.get('overall_score', 0)
            mmlu = data.get('mmlu_accuracy', 0)
            truthful = data.get('truthfulqa_score', 0)
            hellaswag = data.get('hellaswag_accuracy', 0)
            time = data.get('avg_response_time', 0)
            hallucination = data.get('hallucination_rate', 0)
            
            f.write(f"| {model} | {overall:.1%} | {mmlu:.1%} | {truthful:.1%} | "
                   f"{hellaswag:.1%} | {time:.2f}s | {hallucination:.1%} |\n")
        
        # Detailed results
        f.write("\n## 📈 Detailed Results\n\n")
        
        for model, data in results.items():
            f.write(f"### {model}\n\n")
            
            if "error" in data:
                f.write(f"**❌ Evaluation failed:** {data['error']}\n\n")
                continue
            
            f.write("**Quality Metrics:**\n")
            f.write(f"- Accuracy: {data.get('accuracy', 0):.1%}\n")
            f.write(f"- Coherence: {data.get('coherence_score', 0):.1%}\n")
            f.write(f"- Hallucination Rate: {data.get('hallucination_rate', 0):.1%}\n")
            f.write(f"- Overall Score: {data.get('overall_score', 0):.1%}\n\n")
            
            f.write("**Performance Metrics:**\n")
            f.write(f"- Avg Response Time: {data.get('avg_response_time', 0):.2f}s\n")
            f.write(f"- Token Efficiency: {data.get('token_efficiency', 0):.1f} tokens/s\n\n")
            
            f.write("**Benchmarks:**\n")
            f.write(f"- MMLU: {data.get('mmlu_accuracy', 0):.1%} "
                   f"({data.get('questions_tested', 0)} questions)\n")
            f.write(f"- TruthfulQA: {data.get('truthfulqa_score', 0):.1%}\n")
            f.write(f"- HellaSwag: {data.get('hellaswag_accuracy', 0):.1%}\n\n")
        
        # Recommendations
        f.write("## 💡 Recommendations\n\n")
        
        # Find best model for each metric
        best_overall = max(results.items(), 
                          key=lambda x: x[1].get('overall_score', 0) if 'error' not in x[1] else 0)
        best_speed = min(results.items(),
                        key=lambda x: x[1].get('avg_response_time', float('inf')) if 'error' not in x[1] else float('inf'))
        best_accuracy = max(results.items(),
                           key=lambda x: x[1].get('accuracy', 0) if 'error' not in x[1] else 0)
        
        f.write(f"- **Best Overall:** {best_overall[0]} ({best_overall[1].get('overall_score', 0):.1%})\n")
        f.write(f"- **Fastest:** {best_speed[0]} ({best_speed[1].get('avg_response_time', 0):.2f}s)\n")
        f.write(f"- **Most Accurate:** {best_accuracy[0]} ({best_accuracy[1].get('accuracy', 0):.1%})\n\n")
        
        f.write("---\n\n")
        f.write("*Generated by [LLM Evaluation Suite](https://github.com/NahuelGiudizi/llm-evaluation)*\n")


def main():
    """Main execution"""
    print_header("🏆 LLM MULTI-MODEL COMPARISON")
    
    print("\nThis tool compares multiple models side-by-side.")
    print("Choose models to compare (separate with commas):")
    print("\nUltra-fast models (< 1GB):")
    print("  • qwen2.5:0.5b (316MB, fastest)")
    print("  • tinyllama:1.1b (637MB, very fast)")
    print("\nFast models (1-3GB):")
    print("  • llama3.2:1b (1.3GB)")
    print("  • gemma2:2b (1.6GB)")
    print("  • phi3.5:3.8b (2.3GB)")
    print("\nPowerful models (4-8GB):")
    print("  • mistral:7b (4.1GB)")
    print("  • llama3.2:3b (2GB)")
    
    # Get user input
    model_input = input("\n📝 Models (e.g., qwen2.5:0.5b,tinyllama:1.1b): ").strip()
    if not model_input:
        print("❌ No models specified. Using default: qwen2.5:0.5b,tinyllama:1.1b")
        models = ["qwen2.5:0.5b", "tinyllama:1.1b"]
    else:
        models = [m.strip() for m in model_input.split(',')]
    
    # Ask about dataset mode
    if DATASETS_AVAILABLE:
        print("\n🎯 Evaluation mode:")
        print("  1. Demo mode (8 questions, ~30 seconds per model)")
        print("  2. Sampling mode (100 questions, ~3-5 minutes per model)")
        print("  3. Full mode (24,901 questions, ~2-8 hours per model)")
        
        mode = input("\nSelect mode (1/2/3) [1]: ").strip() or "1"
        
        use_full = mode in ["2", "3"]
        sample_size = None if mode == "3" else 100 if mode == "2" else None
    else:
        print("\n⚠️  'datasets' library not installed. Using demo mode only.")
        print("   Install with: pip install datasets")
        use_full = False
        sample_size = None
    
    # Run comparison
    try:
        results = compare_models(
            models=models,
            use_full_datasets=use_full,
            sample_size=sample_size
        )
        
        print_header("✅ COMPARISON COMPLETE")
        print("\n📊 Results saved to: outputs/comparisons/")
        print("   • JSON data")
        print("   • Markdown report")
        print("   • Interactive visualizations")
        print("\n💡 Open comparison_report.md to see detailed analysis!")
        
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("   Make sure Ollama is running and models are available")


if __name__ == "__main__":
    main()
