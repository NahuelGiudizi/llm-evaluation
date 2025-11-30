#!/usr/bin/env python3
"""
Regenerate dashboard with enhanced visualizations showing detailed stats.

This script creates a comprehensive dashboard from the latest comparison results,
including hardware info, timing data, and detailed tooltips.
"""

import json
from pathlib import Path
from llm_evaluator.visualizations import quick_comparison
from llm_evaluator.evaluator import EvaluationResults, DetailedMetrics


def load_latest_comparison():
    """Load the most recent comparison JSON file"""
    comparison_dir = Path("outputs/comparisons")
    
    # Find all comparison JSON files
    json_files = list(comparison_dir.glob("comparison_*.json"))
    if not json_files:
        raise FileNotFoundError("No comparison results found in outputs/comparisons/")
    
    # Get the most recent one
    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 Loading: {latest.name}")
    
    with open(latest, 'r') as f:
        return json.load(f)


def create_evaluation_results_from_dict(data: dict) -> EvaluationResults:
    """Convert JSON dict back to EvaluationResults object"""
    
    # Create detailed metrics
    detailed_metrics = DetailedMetrics(
        benchmarks={
            'mmlu': data.get('mmlu', {}).get('mmlu_accuracy', 0.0),
            'truthfulqa': data.get('truthfulqa', {}).get('truthfulness_score', 0.0),
            'hellaswag': data.get('hellaswag', {}).get('hellaswag_accuracy', 0.0),
        }
    )
    
    # Create EvaluationResults
    results = EvaluationResults(
        model_name=data.get('model', 'unknown'),
        accuracy=data.get('accuracy', 0.0),
        avg_response_time=data.get('avg_response_time', 0.0),
        token_efficiency=data.get('token_efficiency', 0.0),
        hallucination_rate=data.get('hallucination_rate', 0.0),
        coherence_score=data.get('coherence_score', 0.0),
        overall_score=data.get('overall_score', 0.0),
        detailed_metrics=detailed_metrics,
        system_info=data.get('system_info', None)
    )
    
    return results


def main():
    print("=" * 80)
    print("  🎨 REGENERATING DASHBOARD WITH ENHANCED INFO")
    print("=" * 80)
    print()
    
    # Load comparison data
    comparison_data = load_latest_comparison()
    
    print(f"✅ Found {len(comparison_data)} models to compare:")
    for model_name in comparison_data.keys():
        print(f"   • {model_name}")
    print()
    
    # Convert to format expected by quick_comparison
    comparison_results = {}
    detailed_results = {}
    
    for model_name, data in comparison_data.items():
        # Create full EvaluationResults object
        eval_results = create_evaluation_results_from_dict(data)
        
        # Store EvaluationResults for detailed info
        detailed_results[model_name] = eval_results
        
        # Create metrics dict for visualization
        comparison_results[model_name] = {
            'accuracy': eval_results.accuracy,
            'avg_response_time': eval_results.avg_response_time,
            'token_efficiency': eval_results.token_efficiency,
            'hallucination_rate': eval_results.hallucination_rate,
            'coherence': eval_results.coherence_score,
            'consistency': eval_results.coherence_score,  # Use same for now
            'mmlu': eval_results.detailed_metrics.benchmarks.get('mmlu', 0.0),
            'truthful_qa': eval_results.detailed_metrics.benchmarks.get('truthfulqa', 0.0),
            'hellaswag': eval_results.detailed_metrics.benchmarks.get('hellaswag', 0.0),
        }
    
    # Generate visualizations
    output_dir = Path("outputs/comparisons")
    
    print("📊 Generating enhanced visualizations...")
    quick_comparison(
        comparison_results, 
        output_dir=str(output_dir),
        detailed_results=detailed_results
    )
    
    print("\n" + "=" * 80)
    print("  ✅ DASHBOARD REGENERATED!")
    print("=" * 80)
    print(f"\n🌐 Open in browser: {output_dir / 'dashboard.html'}")
    print("\n💡 Hover over the charts to see detailed info:")
    print("   • Hardware specs (GPU, VRAM, RAM)")
    print("   • Token efficiency (tok/s)")
    print("   • Response times")
    print("   • Questions tested per benchmark")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
