"""
Quick script to run full comparison between qwen2.5:0.5b and phi3.5:3.8b
"""

import sys
from llm_evaluator import ModelEvaluator
from llm_evaluator.providers.ollama_provider import OllamaProvider
from llm_evaluator.benchmarks import BenchmarkRunner
from llm_evaluator.visualizations import quick_comparison
from pathlib import Path

def main():
    models = ["qwen2.5:0.5b", "phi3.5:3.8b"]
    results = {}
    
    print("=" * 80)
    print("  🏆 LLM LIGHTWEIGHT COMPARISON: Qwen2.5 vs Phi-3.5")
    print("=" * 80)
    print()
    print("📋 Configuration:")
    print(f"   Models: {', '.join(models)}")
    print("   Mode: Full datasets (24,901 questions)")
    print("   Output: outputs/comparisons/")
    print()
    
    for i, model_name in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model_name}")
        print()
        
        # Create provider
        provider = OllamaProvider(model=model_name)
        
        # Basic evaluation
        print(f"🔍 Evaluating {model_name}...")
        print("   📊 Running basic evaluation...")
        evaluator = ModelEvaluator(provider=provider)
        basic_results = evaluator.evaluate_all()
        
        # Full benchmarks
        print(f"\n   🎯 Running full benchmarks (this will take ~1-2 hours)...")
        benchmark_runner = BenchmarkRunner(provider=provider, use_full_datasets=True)
        
        print("\n🧪 Running benchmarks on {}...".format(model_name))
        print()
        
        # Run all benchmarks with progress bars
        mmlu_results = benchmark_runner.run_mmlu_sample()
        truthfulqa_results = benchmark_runner.run_truthfulqa_sample()
        hellaswag_results = benchmark_runner.run_hellaswag_sample()
        
        # Combine results
        results[model_name] = {
            "basic": basic_results,
            "mmlu": mmlu_results,
            "truthfulqa": truthfulqa_results,
            "hellaswag": hellaswag_results,
        }
        
        print(f"\n✅ {model_name} completed!")
        print(f"   MMLU: {mmlu_results['mmlu_accuracy']:.1%}")
        print(f"   TruthfulQA: {truthfulqa_results['truthfulness_score']:.1%}")
        print(f"   HellaSwag: {hellaswag_results['hellaswag_accuracy']:.1%}")
        print()
    
    # Generate comparison visualizations
    print("\n" + "=" * 80)
    print("  📊 GENERATING COMPARISON VISUALIZATIONS")
    print("=" * 80)
    
    output_dir = Path("outputs/comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to format expected by quick_comparison
    comparison_results = {}
    detailed_results = {}
    
    for model, data in results.items():
        comparison_results[model] = data["basic"]
        # Store full EvaluationResults object for detailed info
        detailed_results[model] = data["basic"]
        # Add benchmark scores to detailed_metrics
        comparison_results[model].detailed_metrics.benchmarks = {
            "mmlu": data["mmlu"]["mmlu_accuracy"],
            "truthfulqa": data["truthfulqa"]["truthfulness_score"],
            "hellaswag": data["hellaswag"]["hellaswag_accuracy"],
        }
    
    quick_comparison(comparison_results, output_dir=str(output_dir), detailed_results=detailed_results)
    
    print("\n" + "=" * 80)
    print("  ✅ COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Visualizations saved to: {output_dir.absolute()}")
    print(f"🌐 Open in browser: {output_dir / 'comparison_dashboard.html'}")
    print()
    
    # Print summary comparison
    print("=" * 80)
    print("  📈 FINAL COMPARISON")
    print("=" * 80)
    print()
    
    for model in models:
        data = results[model]
        print(f"{model}:")
        print(f"  Overall Score:     {data['basic'].overall_score:.2f}/1.00")
        print(f"  MMLU:              {data['mmlu']['mmlu_accuracy']:.1%}")
        print(f"  TruthfulQA:        {data['truthfulqa']['truthfulness_score']:.1%}")
        print(f"  HellaSwag:         {data['hellaswag']['hellaswag_accuracy']:.1%}")
        print(f"  Token Efficiency:  {data['basic'].token_efficiency:.1f} tokens/s")
        print(f"  Avg Response Time: {data['basic'].avg_response_time:.2f}s")
        
        # Show timing if available
        if 'elapsed_time' in data['mmlu']:
            total_time = (data['mmlu']['elapsed_time'] + 
                         data['truthfulqa']['elapsed_time'] + 
                         data['hellaswag']['elapsed_time'])
            print(f"  Total Benchmark Time: {total_time/60:.1f} minutes")
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
