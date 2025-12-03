# 🎓 Academic Usage Guide

> Publication-quality LLM evaluation with statistical rigor.

**Navigation:** [← README](../README.md) | [Quick Start](QUICKSTART.md) | [Providers](PROVIDERS.md) | [API Reference](API.md) | [Benchmarks](FULL_BENCHMARKS.md)

---

## Overview

LLM Benchmark Toolkit provides **publication-ready** evaluation capabilities designed for academic papers, including:

- ✅ **95% Confidence Intervals** (Wilson score method)
- ✅ **Statistical Significance Tests** (McNemar's test)
- ✅ **Effect Size Calculations** (Cohen's h)
- ✅ **Baseline Comparisons** (GPT-4, Claude, Llama)
- ✅ **LaTeX Export** (booktabs tables)
- ✅ **BibTeX Citations** (benchmark papers)
- ✅ **Reproducibility Manifests**

---

## Quick Start

### Command Line

```bash
# Basic academic evaluation (500 questions per benchmark)
llm-eval academic --model llama3.2:1b --sample-size 500

# Full export with LaTeX and BibTeX
llm-eval academic --model gpt-4o-mini \
    --sample-size 500 \
    --output-latex results.tex \
    --output-bibtex citations.bib \
    --output-json results.json
```

### Python API

```python
from llm_evaluator import ModelEvaluator
from llm_evaluator.providers import OpenAIProvider
from llm_evaluator.export import export_to_latex, generate_bibtex

# Setup
provider = OpenAIProvider(model="gpt-4o-mini")
evaluator = ModelEvaluator(provider=provider)

# Run academic evaluation
results = evaluator.evaluate_all_academic(
    sample_size=500,
    compare_baselines=True
)

# Results with confidence intervals
print(f"MMLU: {results.mmlu_accuracy:.1%}")
print(f"95% CI: [{results.mmlu_ci[0]:.1%}, {results.mmlu_ci[1]:.1%}]")
print(f"SE: ±{results.mmlu_se*100:.2f}%")

# Compare to baselines
for baseline, comparison in results.baseline_comparison.items():
    diff = comparison['difference']
    print(f"vs {baseline}: {diff:+.1%}")
```

---

## Statistical Methods

### Wilson Score Confidence Intervals

We use the **Wilson score interval** for binomial proportions, which is preferred over the normal approximation because it:

- Works well for small sample sizes (n < 100)
- Handles proportions near 0 or 1
- Produces asymmetric bounds when appropriate

```python
from llm_evaluator import calculate_wilson_ci

# 85% accuracy on 100 samples
lower, upper = calculate_wilson_ci(correct=85, total=100, confidence=0.95)
print(f"95% CI: [{lower:.1%}, {upper:.1%}]")
# Output: 95% CI: [76.7%, 90.8%]
```

**Reference:** Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical inference". JASA.

### Standard Error

```python
from llm_evaluator import calculate_standard_error

se = calculate_standard_error(correct=850, total=1000)
print(f"SE: {se:.4f}")  # 0.0113
```

### Bootstrap Confidence Intervals

For complex metrics (F1, calibration error):

```python
from llm_evaluator import bootstrap_confidence_interval
import numpy as np

scores = [0.8, 0.9, 0.75, 0.85, 0.92, 0.88, 0.79]
lower, upper = bootstrap_confidence_interval(
    scores,
    statistic=np.mean,
    n_bootstrap=10000,
    confidence=0.95
)
print(f"Mean: {np.mean(scores):.1%} (95% CI: [{lower:.1%}, {upper:.1%}])")
```

### McNemar's Test

For comparing two models on the **same** test set:

```python
from llm_evaluator import mcnemar_test

# model_a_correct[i] = 1 if model A got question i correct
# model_b_correct[i] = 1 if model B got question i correct
model_a = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
model_b = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]

statistic, p_value, significant = mcnemar_test(
    model_a,
    model_b,
    alpha=0.05
)

if significant:
    print(f"Significant difference (p={p_value:.4f})")
else:
    print(f"No significant difference (p={p_value:.4f})")
```

**Reference:** McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions or percentages". Psychometrika.

### Effect Size (Cohen's h)

Measure practical significance:

```python
from llm_evaluator import cohens_h

p1 = 0.85  # Model A accuracy
p2 = 0.80  # Model B accuracy
h = cohens_h(p1, p2)

print(f"Cohen's h: {h:.3f}")
# Interpretation:
# h < 0.2: small effect
# 0.2 ≤ h < 0.5: small-medium effect
# 0.5 ≤ h < 0.8: medium-large effect
# h ≥ 0.8: large effect
```

---

## Baseline Comparisons

Compare your model against published results:

```python
from llm_evaluator import ACADEMIC_BASELINES, compare_to_baselines

# Available baselines
print(ACADEMIC_BASELINES.keys())
# ['gpt-4', 'gpt-4-turbo', 'claude-3-opus', 'claude-3-sonnet',
#  'llama-3-70b', 'llama-3-8b', 'mistral-large', 'gemini-pro']

# Your model's results
my_results = {
    'mmlu': 0.68,
    'truthfulqa': 0.55,
    'hellaswag': 0.72
}

# Compare
comparisons = compare_to_baselines(my_results)
for baseline, diff in comparisons.items():
    print(f"vs {baseline}:")
    print(f"  MMLU: {diff['mmlu']:+.1%}")
    print(f"  TruthfulQA: {diff['truthfulqa']:+.1%}")
```

### Built-in Baselines

| Model | MMLU | TruthfulQA | HellaSwag | Source |
|-------|------|------------|-----------|--------|
| GPT-4 | 86.4% | 71.2% | 95.3% | OpenAI (2023) |
| Claude 3 Opus | 86.8% | 73.5% | 95.4% | Anthropic (2024) |
| Llama 3 70B | 79.5% | 65.3% | 88.0% | Meta (2024) |
| Mistral Large | 81.2% | 68.2% | 89.1% | Mistral (2024) |
| Gemini Pro | 79.1% | 64.8% | 84.7% | Google (2024) |

---

## Export Formats

### LaTeX Tables

```python
from llm_evaluator import export_to_latex

results = {
    'llama3.2:1b': {
        'mmlu': 0.68, 'mmlu_ci': (0.65, 0.71),
        'truthfulqa': 0.55, 'truthfulqa_ci': (0.51, 0.59),
        'hellaswag': 0.72, 'hellaswag_ci': (0.69, 0.75)
    },
    'mistral:7b': {
        'mmlu': 0.78, 'mmlu_ci': (0.75, 0.81),
        'truthfulqa': 0.62, 'truthfulqa_ci': (0.58, 0.66),
        'hellaswag': 0.81, 'hellaswag_ci': (0.78, 0.84)
    }
}

latex = export_to_latex(
    results,
    include_ci=True,
    caption="Benchmark comparison of Llama and Mistral models",
    label="tab:benchmark_results"
)

print(latex)
```

**Output:**

```latex
\begin{table}[htbp]
\centering
\caption{Benchmark comparison of Llama and Mistral models}
\label{tab:benchmark_results}
\begin{tabular}{lccc}
\toprule
Model & MMLU & TruthfulQA & HellaSwag \\
\midrule
llama3.2:1b & 68.0$\pm$3.0 & 55.0$\pm$4.0 & 72.0$\pm$3.0 \\
mistral:7b & 78.0$\pm$3.0 & 62.0$\pm$4.0 & 81.0$\pm$3.0 \\
\bottomrule
\end{tabular}
\end{table}
```

### BibTeX Citations

```python
from llm_evaluator import generate_bibtex

bibtex = generate_bibtex()
print(bibtex)
```

**Output:**

```bibtex
@article{hendrycks2021mmlu,
  title={Measuring Massive Multitask Language Understanding},
  author={Hendrycks, Dan and Burns, Collin and Basart, Steven and ...},
  journal={ICLR},
  year={2021}
}

@article{lin2022truthfulqa,
  title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
  author={Lin, Stephanie and Hilton, Jacob and Evans, Owain},
  journal={ACL},
  year={2022}
}

@article{zellers2019hellaswag,
  title={HellaSwag: Can a Machine Really Finish Your Sentence?},
  author={Zellers, Rowan and Holtzman, Ari and ...},
  journal={ACL},
  year={2019}
}
```

### Reproducibility Manifest

```python
from llm_evaluator import generate_reproducibility_manifest

manifest = generate_reproducibility_manifest(
    model_name="llama3.2:1b",
    sample_size=500
)
print(manifest)
```

**Output:**

```json
{
  "model": "llama3.2:1b",
  "sample_size": 500,
  "python_version": "3.11.5",
  "package_version": "2.1.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "random_seed": 42,
  "datasets": {
    "mmlu": "cais/mmlu@validation",
    "truthfulqa": "truthful_qa@validation",
    "hellaswag": "hellaswag@validation"
  }
}
```

---

## Recommended Sample Sizes

| Use Case | Sample Size | Statistical Power | Runtime |
|----------|-------------|-------------------|---------|
| Quick check | 50-100 | Low | 5-10 min |
| Development | 200-300 | Medium | 20-30 min |
| **Publication** | **500+** | **High** | 1-2 hours |
| Full dataset | 24,901 | Maximum | 2-8 hours |

**Recommendation:** Use 500+ samples for publications to achieve narrow confidence intervals.

```python
# For 95% CI width of ±3%
# Need ~1000 samples at 50% accuracy
# Need ~500 samples at 85% accuracy (Wilson score)
```

---

## Power Analysis for Sample Size Planning

**New in v2.3.0:** Built-in power analysis helps researchers determine the minimum sample size needed to detect statistically significant differences between models.

### Using the CLI

```bash
# Default: detect 5% difference at 80% power
llm-eval power

# Detect smaller 2% difference (needs more samples)
llm-eval power --difference 0.02

# Higher power requirement (90%)  
llm-eval power --power 0.90

# Show reference table
llm-eval power --show-table
```

### Power Analysis API

```python
from llm_evaluator.statistical_metrics import (
    power_analysis_sample_size,
    minimum_sample_size_table
)

# Calculate sample size for your study
result = power_analysis_sample_size(
    expected_difference=0.05,  # 5% improvement to detect
    baseline_accuracy=0.75,    # Expected baseline accuracy
    alpha=0.05,                # Significance level
    power=0.80                 # 80% power (standard)
)

print(f"Need {result['n_per_group']:,} samples per model")
print(f"Total: {result['total_n']:,} samples")
print(f"Effect size (Cohen's h): {result['effect_size_h']:.3f}")

# Get benchmark recommendations
for bench, info in result['recommendations'].items():
    print(f"{bench}: use {info['recommended']:,} of {info['available']:,} available")
```

### Reference Table

| Power | 2% diff | 5% diff | 10% diff | 15% diff |
|-------|---------|---------|----------|----------|
| 80%   | 14,312  | 2,184   | 496      | 194      |
| 90%   | 19,160  | 2,924   | 664      | 258      |
| 95%   | 23,694  | 3,616   | 820      | 320      |

*Values are TOTAL samples (divide by 2 for per-model count). Based on baseline accuracy of 75%, α=0.05.*

### Interpreting Results

- **Small differences (2%)**: Require very large samples (>7,000 per model)
- **Medium differences (5%)**: Feasible with MMLU or HellaSwag (~1,000 per model)
- **Large differences (10%+)**: Achievable with most benchmarks (~300 per model)

**Tip:** If you can't achieve the required sample size, consider:
1. Increasing expected effect size (comparing more different models)
2. Combining multiple benchmarks
3. Using 80% power instead of 95%
4. Reporting effect sizes alongside significance

---

## Reproducibility Settings

**New in v2.3.0:** Full reproducibility controls for academic evaluations.

### CLI Options

```bash
# Fully reproducible evaluation
llm-eval benchmark --model llama3.2:1b \
    --seed 42 \
    --temperature 0.0 \
    --sample-size 500

# Academic command defaults to temperature=0.0
llm-eval academic --model llama3.2:1b --seed 42
```

### Python API

```python
from llm_evaluator import ModelEvaluator
from llm_evaluator.providers import OllamaProvider

provider = OllamaProvider(model="llama3.2:1b")
evaluator = ModelEvaluator(provider=provider)

# Fully reproducible evaluation
results = evaluator.evaluate_all_academic(
    sample_size=500,
    seed=42,           # Reproducible sample selection
    temperature=0.0    # Deterministic LLM outputs
)

# Results include reproducibility manifest
print(results.reproducibility_manifest)
```

### Reproducibility Manifest

```json
{
  "model": "llama3.2:1b",
  "sample_size": 500,
  "random_seed": 42,
  "temperature": 0.0,
  "python_version": "3.11.5",
  "package_version": "2.3.0",
  "timestamp": "2024-12-02T10:30:00Z",
  "results_hash": "sha256:abc123..."
}
```

### Best Practices for Reproducibility

1. **Always set a seed**: Use `--seed 42` (or any fixed number)
2. **Use temperature=0**: Minimizes LLM output variance
3. **Record package versions**: Include in your paper's appendix
4. **Share your manifest**: Upload with supplementary materials
5. **Cache responses**: Use `--cache` to store LLM outputs

---

## Complete Academic Example

```python
from llm_evaluator import (
    ModelEvaluator,
    export_to_latex,
    generate_bibtex,
    generate_reproducibility_manifest,
    mcnemar_test
)
from llm_evaluator.providers import OllamaProvider, CachedProvider
import json

# Setup with caching
provider = CachedProvider(OllamaProvider(model="llama3.2:1b"))
evaluator = ModelEvaluator(provider=provider)

# Run academic evaluation
results = evaluator.evaluate_all_academic(
    sample_size=500,
    compare_baselines=True
)

# Print results
print("=" * 60)
print(f"Model: {results.model_name}")
print("=" * 60)
print(f"\nMMLU: {results.mmlu_accuracy:.1%}")
print(f"  95% CI: [{results.mmlu_ci[0]:.1%}, {results.mmlu_ci[1]:.1%}]")
print(f"  SE: ±{results.mmlu_se*100:.2f}%")
print(f"  n = {results.mmlu_n}")

print(f"\nTruthfulQA: {results.truthfulqa_accuracy:.1%}")
print(f"  95% CI: [{results.truthfulqa_ci[0]:.1%}, {results.truthfulqa_ci[1]:.1%}]")

print(f"\nHellaSwag: {results.hellaswag_accuracy:.1%}")
print(f"  95% CI: [{results.hellaswag_ci[0]:.1%}, {results.hellaswag_ci[1]:.1%}]")

print(f"\nAverage: {results.average_accuracy:.1%}")

# Baseline comparisons
print("\n" + "=" * 60)
print("Baseline Comparisons")
print("=" * 60)
for baseline, comparison in results.baseline_comparison.items():
    print(f"\nvs {baseline}:")
    for metric, diff in comparison.items():
        if isinstance(diff, (int, float)):
            print(f"  {metric}: {diff:+.1%}")

# Export for paper
latex = export_to_latex({results.model_name: results.to_dict()})
bibtex = generate_bibtex()
manifest = results.reproducibility_manifest

# Save files
with open("results.tex", "w") as f:
    f.write(latex)
    
with open("citations.bib", "w") as f:
    f.write(bibtex)
    
with open("reproducibility.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("\n✅ Exported: results.tex, citations.bib, reproducibility.json")
```

---

## Paper Template

Use this template for your methods section:

> We evaluated [MODEL] using the LLM Benchmark Toolkit (v2.3.0) on three standard benchmarks: MMLU (Hendrycks et al., 2021), TruthfulQA (Lin et al., 2022), and HellaSwag (Zellers et al., 2019). We sampled 500 questions per benchmark and report accuracy with 95% Wilson score confidence intervals. Statistical significance was assessed using McNemar's test (α = 0.05). For reproducibility, we set the random seed to 42 and temperature to 0.0.

---

## References

1. Hendrycks, D., et al. (2021). "Measuring Massive Multitask Language Understanding." ICLR.
2. Lin, S., et al. (2022). "TruthfulQA: Measuring How Models Mimic Human Falsehoods." ACL.
3. Zellers, R., et al. (2019). "HellaSwag: Can a Machine Really Finish Your Sentence?" ACL.
4. Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical inference." JASA.
5. McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions." Psychometrika.

---

**Navigation:** [← README](../README.md) | [Quick Start](QUICKSTART.md) | [Providers](PROVIDERS.md) | [API Reference](API.md) | [Benchmarks](FULL_BENCHMARKS.md)
