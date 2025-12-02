# 🚀 LLM Benchmark Toolkit

<p align="center">
  <img src="https://img.shields.io/pypi/v/llm-benchmark-toolkit?style=for-the-badge&color=blue" alt="PyPI">
  <img src="https://img.shields.io/pypi/dm/llm-benchmark-toolkit?style=for-the-badge&color=green" alt="Downloads">
  <img src="https://img.shields.io/github/stars/NahuelGiudizi/llm-evaluation?style=for-the-badge" alt="Stars">
  <img src="https://img.shields.io/badge/coverage-79%25-brightgreen?style=for-the-badge" alt="Coverage">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=for-the-badge" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <b>🎯 Benchmark LLMs with 24,901 real questions from MMLU, TruthfulQA & HellaSwag</b>
</p>

<p align="center">
  <a href="#-get-started-60-seconds">Get Started</a> •
  <a href="#-compare-models">Compare Models</a> •
  <a href="#-python-api">Python API</a> •
  <a href="#-academic-use">Academic</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 🚀 Get Started (60 Seconds)

### Install

```bash
# Full installation (dashboard + all providers)
pip install llm-benchmark-toolkit
```

That's it! Everything included: Dashboard, OpenAI, Anthropic, Ollama, HuggingFace.

### 🌐 Web Dashboard (Recommended!)

The easiest way to evaluate models - a beautiful web interface:

```bash
# Launch the dashboard
llm-eval dashboard
```

Opens your browser to `http://localhost:8888` where you can:

- 🚀 **Run evaluations** with real-time progress tracking
- 📊 **Compare models** with interactive charts
- 🔍 **Inspect scenarios** - see every question & answer
- 📈 **View history** - track improvements over time
- 💾 **Export results** - JSON, charts, reports

### Quick CLI Evaluation

```bash
# Auto-detects provider from environment
export OPENAI_API_KEY="sk-..."
llm-eval quick
```

**Output:**

```
🚀 LLM QUICK EVALUATION
==================================================
✅ Provider: openai (gpt-4o-mini)
✅ Sample size: 20

📊 RESULTS
==================================================
  🎯 MMLU:       78.5%
  🎯 TruthfulQA: 71.2%
  🎯 HellaSwag:  82.4%
  
  📈 Overall:    77.4%
==================================================
✨ Evaluation complete!
```

**Auto-detection works with:**

- `OPENAI_API_KEY` → GPT-4o-mini
- `ANTHROPIC_API_KEY` → Claude 3.5 Sonnet
- `DEEPSEEK_API_KEY` → DeepSeek-V3
- Ollama running locally → Llama 3.2

---

## 🔄 Compare Models

```bash
llm-eval compare \
  --models gpt-4o-mini,claude-3-5-sonnet \
  --sample-size 100
```

**More examples:**

```bash
# Ollama (local models)
llm-eval quick --model llama3.2:1b

# OpenAI
llm-eval quick --model gpt-4o-mini

# Anthropic
llm-eval run --model claude-3-5-sonnet-20241022 --provider anthropic

# DeepSeek (super affordable!)
llm-eval quick --model deepseek-chat

# Run specific benchmarks
llm-eval benchmark --model gpt-4o --benchmarks mmlu,truthfulqa

# Full academic evaluation
llm-eval academic --model llama3.2:1b \
  --sample-size 500 \
  --output-latex results.tex
```

---

## 🐍 Python API

```python
from llm_evaluator import ModelEvaluator
from llm_evaluator.providers import OpenAIProvider

provider = OpenAIProvider(model="gpt-4o-mini")
evaluator = ModelEvaluator(provider=provider)

results = evaluator.evaluate_all()
print(f"Overall: {results.overall_score:.1%}")
```

**With caching (10x faster):**

```python
from llm_evaluator.providers import CachedProvider, OllamaProvider

provider = OllamaProvider(model="llama3.2:1b")
cached = CachedProvider(provider)  # Automatic caching!

evaluator = ModelEvaluator(provider=cached)
results = evaluator.evaluate_all()
```

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| 📊 **Real Benchmarks** | 14,042 MMLU + 817 TruthfulQA + 10,042 HellaSwag |
| 🔌 **5 Providers** | Ollama, OpenAI, Anthropic, DeepSeek, HuggingFace |
| ⚡ **Zero Config** | Auto-detects provider from environment |
| 💾 **10x Caching** | Intelligent caching for repeated evaluations |
| 📈 **Academic Rigor** | 95% CI, McNemar tests, baseline comparisons |
| 📄 **Paper Exports** | LaTeX tables, BibTeX citations |
| 🎨 **Beautiful CLI** | Progress bars, colored output |

---

## 🎓 Academic Use

For publication-quality evaluations:

```python
from llm_evaluator import ModelEvaluator
from llm_evaluator.providers import OllamaProvider
from llm_evaluator.export import export_to_latex, generate_bibtex

provider = OllamaProvider(model="llama3.2:1b")
evaluator = ModelEvaluator(provider=provider)

results = evaluator.evaluate_all_academic(
    sample_size=500,
    compare_baselines=True
)

# 95% confidence intervals
print(f"MMLU: {results.mmlu_accuracy:.1%}")
print(f"95% CI: [{results.mmlu_ci[0]:.1%}, {results.mmlu_ci[1]:.1%}]")

# Compare to GPT-4, Claude, Llama baselines
for baseline, comparison in results.baseline_comparison.items():
    print(f"vs {baseline}: {comparison['difference']:+.1%}")

# Export for papers
latex = export_to_latex(results, "My Model")
bibtex = generate_bibtex()
```

---

## 🎨 Visual Output Examples

### Benchmark Comparison

![Benchmark Comparison](docs/images/benchmark_comparison.png)

### Interactive Dashboard

![Dashboard](docs/images/dashboard.png)

*(Add screenshots to `docs/images/` folder)*

---

## 🔌 Check Available Providers

```bash
llm-eval providers
```

```
🔌 Available Providers:

✅ Auto-detected: openai (gpt-4o-mini)

  ✅ ollama          - Local LLMs (llama3.2, mistral, etc.)
  ✅ openai          - GPT-3.5, GPT-4, GPT-4o
  ❌ anthropic       - Claude 3/3.5 (pip install anthropic)
  ✅ deepseek        - DeepSeek-V3, DeepSeek-R1
  ❌ huggingface     - Inference API

📋 Environment Variables:
  ✅ OPENAI_API_KEY       sk-abc1...
  ❌ ANTHROPIC_API_KEY    Not set
```

---

## 🔬 Benchmarks Included

| Benchmark | Questions | Description |
|-----------|-----------|-------------|
| **MMLU** | 14,042 | Massive Multitask Language Understanding - 57 subjects |
| **TruthfulQA** | 817 | Truthfulness and avoiding misinformation |
| **HellaSwag** | 10,042 | Common-sense reasoning |

---

## 🤝 Contributing

This is open source. Make it better:

```bash
git clone https://github.com/NahuelGiudizi/llm-evaluation
cd llm-evaluation
pip install -e ".[dev]"
pytest tests/ -v
```

### Wanted

- [ ] More providers (Cohere, AI21, Groq, etc.)
- [ ] More benchmarks (GSM8K, HumanEval, GPQA)
- [ ] Async evaluation for faster throughput
- [ ] Web UI (Gradio/Streamlit dashboard)
- [ ] Batch evaluation mode
- [ ] Custom benchmark support

**Contributors welcome!** 🎉

---

## 📚 Documentation

| Doc | Description |
|-----|-------------|
| 📖 [Quick Start](docs/QUICKSTART.md) | Get running in 5 minutes |
| 🔌 [Providers Guide](docs/PROVIDERS.md) | Ollama, OpenAI, Anthropic, DeepSeek, HuggingFace |
| 🔬 [Benchmarks](docs/FULL_BENCHMARKS.md) | MMLU, TruthfulQA, HellaSwag details |
| 🎓 [Academic Usage](docs/ACADEMIC_USAGE.md) | Statistical methods, LaTeX export |
| 📘 [API Reference](docs/API.md) | Complete Python API documentation |

---

## 📊 Output Formats

```bash
# JSON (default)
llm-eval run --model llama3.2:1b --output results.json

# Markdown report
llm-eval run --model llama3.2:1b --output report.md

# LaTeX tables (for papers)
llm-eval academic --model llama3.2:1b --output-latex table.tex

# BibTeX citations
llm-eval academic --model llama3.2:1b --output-bibtex refs.bib
```

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## ⭐ Star History

If this project helped you, please star it! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=NahuelGiudizi/llm-evaluation&type=Date)](https://star-history.com/#NahuelGiudizi/llm-evaluation&Date)

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/NahuelGiudizi">Nahuel Giudizi</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/llm-benchmark-toolkit/">
    <img src="https://img.shields.io/badge/Install-pip%20install%20llm--benchmark--toolkit-blue?style=for-the-badge&logo=python" alt="Install">
  </a>
</p>
