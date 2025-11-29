"""
Quick demo of LLM Evaluator
"""

from llm_evaluator import ModelEvaluator

def main():
    print("="*60)
    print("LLM EVALUATOR - Quick Demo")
    print("="*60)
    
    # Initialize evaluator
    model = "llama3.2:1b"
    evaluator = ModelEvaluator(model=model)
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_all()
    
    # Generate report
    evaluator.generate_report(results, output="evaluation_report.md")
    
    print("\n✅ Demo complete!")
    print(f"📊 Overall Score: {results.overall_score:.2f}/1.00")
    print(f"📄 Report saved to: evaluation_report.md")


if __name__ == "__main__":
    main()
