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
  <b>🎯 Benchmark LLMs with 9 benchmarks & 100,000+ real questions</b><br>
  <sub>MMLU • TruthfulQA • HellaSwag • ARC • WinoGrande • CommonsenseQA • BoolQ • SafetyBench • Do-Not-Answer</sub>
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
python -m llm_evaluator.dashboard
```

Opens your browser to `http://localhost:8888` where you can:

- 🚀 **Run evaluations** with real-time progress tracking
- 📊 **Compare models** with interactive charts
- 🔍 **Inspect scenarios** - see every question & answer
- 📈 **View history** - track improvements over time
- 💾 **Export results** - JSON, charts, reports

### Quick CLI Evaluation

```bash
# Set your API key (Windows)
set OPENAI_API_KEY=sk-...

# Or on Linux/Mac
export OPENAI_API_KEY="sk-..."

# Run quick evaluation
python -m llm_evaluator.cli quick
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

# Run specific benchmarks (any combination!)
llm-eval benchmark --model gpt-4o --benchmarks mmlu,truthfulqa,arc,safetybench

# Run ALL benchmarks
llm-eval benchmark --model llama3.2:1b --benchmarks mmlu,truthfulqa,hellaswag,arc,winogrande,commonsenseqa,boolq,safetybench,donotanswer

# Full academic evaluation
llm-eval academic --model llama3.2:1b \
  --sample-size 500 \
  --output-latex results.tex
```

---

## 🖥️ CLI Commands Reference

| Command | Description |
|---------|-------------|
| `llm-eval quick` | 🚀 Zero-config evaluation (auto-detects provider) |
| `llm-eval run` | Full evaluation on a single model |
| `llm-eval benchmark` | Run specific benchmarks |
| `llm-eval compare` | Compare multiple models side-by-side |
| `llm-eval vs` | 🥊 Run same benchmark on multiple models sequentially |
| `llm-eval dashboard` | 🌐 Launch web dashboard |
| `llm-eval academic` | 🎓 Academic evaluation with statistics |
| `llm-eval export` | 📤 Export results (JSON, CSV, LaTeX, BibTeX) |
| `llm-eval providers` | Check available providers status |
| `llm-eval list-runs` | 📋 List saved evaluation runs |

### Key Options

```bash
# Common options for most commands
-m, --model TEXT       # Model name
-p, --provider TYPE    # ollama, openai, anthropic, huggingface, deepseek,
                       # groq, together, fireworks
-s, --sample-size INT  # Number of questions to test
-u, --base-url URL     # Custom API endpoint (vLLM, LM Studio, Azure)
--cache / --no-cache   # Enable/disable caching

# Benchmark selection
-b, --benchmarks TEXT  # Comma-separated: mmlu,truthfulqa,hellaswag,arc,
                       # winogrande,commonsenseqa,boolq,safetybench,donotanswer
```

### VS Command (Model Battle)

Compare models head-to-head:

```bash
# Compare two local models
llm-eval vs llama3.2:1b mistral:7b

# Compare with specific benchmarks
llm-eval vs llama3.2:1b mistral:7b -b mmlu,arc -s 50

# Compare models from different providers
llm-eval vs gpt-4o-mini claude-3.5-sonnet -p openai,anthropic

# Ultra-fast with Groq
llm-eval quick --model llama-3.1-8b-instant --provider groq
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
| 📊 **9 Benchmarks** | MMLU, TruthfulQA, HellaSwag, ARC, WinoGrande, CommonsenseQA, BoolQ, SafetyBench, Do-Not-Answer |
| 🔢 **100,000+ Questions** | Real academic datasets from HuggingFace |
| 🔌 **8 Providers** | Ollama, OpenAI, Anthropic, DeepSeek, Groq, Together.ai, Fireworks, HuggingFace |
| 🐳 **Docker Support** | `docker run llm-benchmark quick` |
| 🌐 **Web Dashboard** | Beautiful UI with real-time progress, charts, and history |
| ⚡ **Zero Config** | Auto-detects provider from environment variables |
| 💾 **Smart Caching** | 10x faster repeated evaluations |
| 📈 **Academic Rigor** | 95% CI, McNemar tests, baseline comparisons |
| 📄 **Paper Exports** | LaTeX tables, BibTeX citations, CSV, JSON |
| 🛡️ **Safety Testing** | SafetyBench + Do-Not-Answer for security evaluation |
| 🎨 **Beautiful CLI** | Progress bars, colored output, ETA tracking |

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

### 📚 Knowledge & Reasoning (7 benchmarks)

| Benchmark | Questions | Description |
|-----------|-----------|-------------|
| **MMLU** | 14,042 | Massive Multitask Language Understanding - 57 subjects |
| **TruthfulQA** | 817 | Truthfulness and avoiding misinformation |
| **HellaSwag** | 10,042 | Common-sense reasoning and sentence completion |
| **ARC-Challenge** | 2,590 | Grade-school science questions (hard subset) |
| **WinoGrande** | 44,000 | Pronoun resolution and commonsense reasoning |
| **CommonsenseQA** | 12,247 | Commonsense knowledge questions |
| **BoolQ** | 15,942 | Yes/no reading comprehension questions |

### 🛡️ Safety & Security (2 benchmarks)

| Benchmark | Questions | Description |
|-----------|-----------|-------------|
| **SafetyBench** | 11,000 | Safety evaluation across multiple risk categories |
| **Do-Not-Answer** | 939 | Harmful prompt detection and refusal testing |

**Total: 9 benchmarks, 100,000+ questions**

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

- [ ] Async evaluation for faster throughput
- [ ] More benchmarks (GSM8K, HumanEval, GPQA, MT-Bench)
- [ ] Batch evaluation mode
- [ ] Custom benchmark support
- [ ] Kubernetes deployment

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

## 🐳 Docker

Run benchmarks without installing anything:

```bash
# Build the image
docker build -t llm-benchmark .

# Quick evaluation with OpenAI
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY llm-benchmark quick

# Ultra-fast with Groq
docker run -e GROQ_API_KEY=$GROQ_API_KEY llm-benchmark quick \
  --model llama-3.1-8b-instant --provider groq

# Run specific benchmarks
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY llm-benchmark benchmark \
  --model gpt-4o-mini --benchmarks mmlu,truthfulqa -s 50

# Launch dashboard
docker run -p 8888:8888 -e OPENAI_API_KEY=$OPENAI_API_KEY \
  llm-benchmark dashboard --host 0.0.0.0

# With docker-compose
docker compose up dashboard
```

---

## 📊 Output Formats

```bash
# JSON (default)
llm-eval run --model llama3.2:1b --output results.json

# Export to multiple formats
llm-eval export results.json --format all

# Individual formats
llm-eval export results.json --format csv
llm-eval export results.json --format latex
llm-eval export results.json --format bibtex

# Academic evaluation with direct exports
llm-eval academic --model llama3.2:1b --output-latex table.tex --output-bibtex refs.bib
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
