# LLM Evaluation Suite

> Comprehensive evaluation framework for testing Large Language Model capabilities across multiple dimensions

## 🎯 Project Overview

Week 3-4 of AI Safety Engineer roadmap. Building on Week 1-2 security testing, this project focuses on ML evaluation metrics, performance benchmarking, and statistical analysis of LLM outputs.

## 🚀 Features

- **Performance Metrics**: Response time, token efficiency, memory usage
- **Quality Metrics**: Factual accuracy, coherence, hallucination detection
- **Standard Benchmarks**: MMLU, TruthfulQA, HellaSwag integration
- **Comparison Dashboard**: Side-by-side model analysis with visualizations
- **Statistical Analysis**: Significance testing, confidence intervals

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/NahuelGiudizi/llm-evaluation.git
cd llm-evaluation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## 🔧 Quick Start

```python
from llm_evaluator import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(model="llama3.2:1b")

# Run comprehensive evaluation
results = evaluator.evaluate_all()

# Print summary
print(f"Accuracy: {results.accuracy:.2%}")
print(f"Avg Response Time: {results.avg_response_time:.2f}s")
print(f"Hallucination Rate: {results.hallucination_rate:.2%}")

# Generate report
evaluator.generate_report(output="evaluation_report.html")
```

## 📊 Supported Benchmarks

- **MMLU** (Massive Multitask Language Understanding) - 57 subjects
- **TruthfulQA** - Truthfulness and informativeness
- **HellaSwag** - Commonsense reasoning
- **Custom Datasets** - Domain-specific evaluations

## 🎓 Learning Outcomes

- ✅ ML evaluation metrics (accuracy, precision, recall, F1)
- ✅ Statistical significance testing
- ✅ Model performance profiling
- ✅ Data visualization with matplotlib/plotly
- ✅ Integration with HuggingFace ecosystem

## 📁 Project Structure

```
llm-evaluation/
├── src/
│   └── llm_evaluator/
│       ├── __init__.py
│       ├── evaluator.py       # Main evaluation class
│       ├── metrics.py          # Performance & quality metrics
│       ├── benchmarks.py       # Standard benchmark integrations
│       └── visualizer.py       # Dashboard generation
├── tests/
│   └── test_evaluator.py
├── notebooks/
│   └── analysis.ipynb          # Interactive analysis
├── data/
│   └── results/                # Evaluation results storage
├── docs/
│   └── EXAMPLES.md
├── README.md
├── requirements.txt
└── setup.py
```

## 🛠️ Tech Stack

- **Python 3.11+**
- **Ollama** - Local LLM runtime
- **HuggingFace Datasets** - Standard benchmarks
- **lm-evaluation-harness** - EleutherAI evaluation framework
- **scikit-learn** - Statistical metrics
- **matplotlib/plotly** - Visualizations
- **Jupyter** - Interactive analysis

## 🔗 Integration with Week 1-2

Combine security testing with performance evaluation:

```python
from ai_safety_tester import SimpleAITester, SeverityScorer
from llm_evaluator import ModelEvaluator

# Initialize both
security_tester = SimpleAITester(model="llama3.2:1b")
performance_evaluator = ModelEvaluator(model="llama3.2:1b")

# Run combined analysis
security_score = security_tester.run_all_tests()
performance_metrics = performance_evaluator.evaluate_all()

# Holistic assessment
print(f"Security: {security_score.aggregate_score}/10")
print(f"Performance: {performance_metrics.overall_score:.2f}")
```

## 📈 Expected Results

- Benchmark 4+ models on standard datasets
- Statistical comparison with confidence intervals
- Identify strengths/weaknesses per model
- Reusable evaluation pipeline

## 📝 Notes

- **Cost**: $0 (100% local with Ollama)
- **Models Tested**: Llama 3.2, Mistral, Phi-3, Gemma
- **Evaluation Time**: ~10 minutes per model
- **Output**: HTML/JSON/Markdown reports

## 🔗 Resources

- [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
- [MMLU Benchmark](https://github.com/hendrycks/test)

---

**Author**: Nahuel  
**Date**: November 2025  
**Project**: AI Safety & Alignment Testing Roadmap - Week 3-4
