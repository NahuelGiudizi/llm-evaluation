# LLM Evaluation Suite

[![Tests](https://github.com/NahuelGiudizi/llm-evaluation/workflows/Tests/badge.svg)](https://github.com/NahuelGiudizi/llm-evaluation/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Enterprise-grade evaluation framework for Large Language Models with comprehensive metrics, benchmarking, and visualization capabilities.

## 🎯 Overview

A production-ready toolkit for evaluating LLM performance across multiple dimensions: quality, performance, and standard benchmarks. Features interactive dashboards, statistical analysis, and seamless integration with Ollama and HuggingFace ecosystems.

## 🚀 Features

- **Performance Metrics**: Response time, throughput, token efficiency, latency percentiles
- **Quality Metrics**: Accuracy, coherence, hallucination detection, BLEU scores
- **Standard Benchmarks**: MMLU, TruthfulQA, HellaSwag integration
- **Interactive Visualizations**: Comparison dashboards, radar charts, heatmaps, trend analysis
- **Statistical Analysis**: Significance testing, confidence intervals, distribution analysis
- **Export Capabilities**: HTML reports, PNG charts, JSON data exports
- **Multi-Model Support**: Compare multiple models side-by-side
- **100% Local**: No API costs, complete data privacy with Ollama

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

## 📊 Visualization Examples

Generate interactive dashboards with a single function:

```python
from llm_evaluator import quick_comparison

results = {
    "llama3.2:1b": {"mmlu": 0.65, "accuracy": 0.75, "coherence": 0.82},
    "mistral:7b": {"mmlu": 0.78, "accuracy": 0.82, "coherence": 0.88},
    "phi3:3.8b": {"mmlu": 0.71, "accuracy": 0.78, "coherence": 0.85}
}

# Generate all visualizations
quick_comparison(results, output_dir="outputs")
```

Creates:
- 📊 Bar charts for benchmark comparisons
- 🎯 Radar charts for multi-metric analysis
- 🔥 Heatmaps for performance overview
- 📈 Line charts for trend analysis
- 📦 Box plots for distribution analysis
- 🎨 Interactive HTML dashboards

## 🛠️ Tech Stack

- **Python 3.11+** - Modern Python with type hints
- **Ollama** - Local LLM runtime (no API costs)
- **HuggingFace Datasets** - Standard benchmark datasets
- **scikit-learn** - Statistical metrics and analysis
- **matplotlib** - Static visualizations
- **plotly** - Interactive charts and dashboards
- **seaborn** - Statistical data visualization
- **pytest** - Comprehensive test coverage (87%)

## 📈 Performance

Typical evaluation metrics:
- **Speed**: ~100-500ms per prompt (model dependent)
- **Memory**: <2GB RAM for 1B models, <8GB for 7B models
- **Benchmark Time**: ~5-10 minutes per model
- **Cost**: $0 (100% local execution)

## 🎓 Use Cases

- **Model Selection**: Compare multiple LLMs to choose the best fit for your use case
- **Performance Optimization**: Identify bottlenecks and optimize inference
- **Quality Assurance**: Validate model outputs before production deployment
- **Benchmark Tracking**: Monitor model performance over time
- **Research & Development**: Analyze model behavior across different metrics
- **Cost-Performance Analysis**: Balance quality vs. speed for your requirements

## 📁 Project Structure

```
llm-evaluation/
├── src/
│   └── llm_evaluator/
│       ├── __init__.py
│       ├── evaluator.py       # Main evaluation orchestrator
│       ├── metrics.py          # Performance & quality metrics
│       ├── benchmarks.py       # Standard benchmark integrations
│       └── visualizations.py   # Chart and dashboard generation
├── tests/
│   ├── conftest.py            # Pytest fixtures
│   ├── test_metrics.py        # Metrics tests (100% coverage)
│   ├── test_evaluator.py      # Evaluator tests
│   ├── test_benchmarks.py     # Benchmark tests
│   └── test_visualizations.py # Visualization tests
├── notebooks/
│   └── analysis.ipynb         # Interactive analysis examples
├── outputs/
│   └── visualizations/        # Generated charts
├── docs/
│   └── EXAMPLES.md            # Usage examples
├── .github/
│   └── workflows/
│       └── tests.yml          # CI/CD pipeline
├── README.md
├── pyproject.toml
├── pytest.ini
└── requirements.txt
```

## 🔗 Related Projects

- **[ai-safety-testing](https://github.com/NahuelGiudizi/ai-safety-testing)** - Security vulnerability testing for LLMs

## 📝 Development

Built with enterprise best practices:
- ✅ Test-Driven Development (TDD)
- ✅ 87% code coverage
- ✅ Type hints throughout
- ✅ Black code formatting
- ✅ Ruff linting
- ✅ GitHub Actions CI/CD
- ✅ Comprehensive documentation

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

**Nahuel Giudizi**
- GitHub: [@NahuelGiudizi](https://github.com/NahuelGiudizi)
- LinkedIn: [Nahuel Giudizi](https://www.linkedin.com/in/nahuel-giudizi/)

## 🙏 Acknowledgments

- [Ollama](https://ollama.com) for local LLM runtime
- [HuggingFace](https://huggingface.co) for datasets and models
- [EleutherAI](https://www.eleuther.ai/) for evaluation frameworks

---

**⭐ Star this repo if you find it useful!**
