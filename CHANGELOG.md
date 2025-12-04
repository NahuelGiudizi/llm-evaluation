# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.3] - 2025-12-03

### Fixed

- **Cross-Platform Compatibility**:
  - Fixed mypy type checking errors for Windows-only `winreg` module on Linux/macOS
  - Added `TYPE_CHECKING` imports and `type: ignore` comments for platform-specific code
  - Ensures CI/CD pipelines pass on all operating systems (Windows, Linux, macOS)

## [2.3.2] - 2025-12-03

### Added

- **Dashboard UI Enhancements**:
  - Run sorting in Model Comparison: 10 sorting options (date, model name, score, sample size, temperature)
  - Click-to-highlight in charts: Click legend items to highlight specific runs (bar chart opacity, radar chart stroke)
  - Improved Radar Chart: Removed white dots for cleaner visualization

## [2.3.1] - 2025-12-03

### Fixed

- **Dashboard UI Fixes**:
  - Fixed GSM8K showing 0% across all dashboard views (RunDetail, ModelComparison, RunHistory)
  - Fixed Do-Not-Answer showing 0% (added support for `donotanswer_accuracy` field in demo mode)
  - Fixed missing benchmark names in RunDetail chart (CommonsenseQA, BoolQ, SafetyBench, Do-Not-Answer)
  - Fixed intrusive auto-scroll in `/progress` route (now only activates when user is near bottom)
  - Fixed Tailwind CSS build errors caused by `box-shadow` in `@apply` directives

## [2.3.0] - 2025-12-02

### Added

- **Power Analysis Tool** - Calculate required sample sizes for academic rigor:
  - `llm-eval power --difference 0.05` - Calculate sample size for detecting 5% differences
  - `llm-eval power --show-table` - Display reference table for common scenarios
  - Based on Cohen's h effect size and statistical power analysis (Cohen, 1988)
  - Benchmark-specific recommendations with available sample sizes
- **Reproducibility Controls** - Ensure reproducible benchmark runs:
  - `--seed` option for deterministic sample selection
  - `--temperature` option for LLM generation (default 0.0 for determinism)
  - Works with `llm-eval benchmark` and `llm-eval academic` commands
  - Sets both Python's `random.seed()` and `numpy.random.seed()`
- **GSM8K Benchmark** - Grade school math reasoning (8,500 problems)
  - Tests mathematical reasoning capabilities
  - Extracts numerical answers from natural language responses
  - Included in `run_all_benchmarks()` aggregate score
  - Added to Dashboard UI for visual evaluation
- **3 New Providers** - Now supporting 8 LLM providers:
  - **Groq** - Ultra-fast inference (100+ tokens/sec) with Llama, Mixtral
  - **Together.ai** - Access to 100+ open models
  - **Fireworks AI** - Optimized open model inference
- **Docker Support**
  - Multi-stage Dockerfile for minimal image size
  - docker-compose.yml for easy deployment
  - Run benchmarks with `docker run llm-benchmark quick`
  - Launch dashboard with `docker compose up dashboard`
- **Parallel Execution** - 5-10x speedup with `--workers` option
  - `llm-eval benchmark --workers 4` for 4x parallel speedup
  - Works with all providers that support concurrent requests
  - Thread-safe progress bar with live accuracy updates

### Changed

- Updated CLI to support new providers (`--provider groq|together|fireworks`)
- Updated `llm-eval providers` command to show all 8 providers
- Updated `--benchmarks` option to include gsm8k
- README updated with Docker examples, new providers, and GSM8K benchmark
- Now 10 benchmarks totaling 108,000+ questions
- Academic documentation updated with power analysis and reproducibility guides

### Fixed

- Type annotations for strict mypy compliance
- Dashboard GSM8K integration with API endpoints

## [2.2.2] - 2025-12-02

### Fixed

- **Dashboard progress bar** - All benchmarks now correctly update the progress bar
  - Fixed: ARC-Challenge, WinoGrande, CommonsenseQA, BoolQ, SafetyBench, Do-Not-Answer
  - Previously only MMLU, TruthfulQA, and HellaSwag showed progress
- **README documentation** - Corrected dashboard launch command

### Added

- **Duration display** - Run details now show total duration (wall clock time)
- **Inference Time** - New metric showing sum of all benchmark execution times (scientific metric)
  - Displayed in Run Details view alongside Duration
  - More meaningful for reproducibility than wall clock time

## [2.2.1] - 2025-12-02

### Changed

- **Production/Stable Status** - Package now marked as production-ready
- **Full installation by default** - `pip install llm-benchmark-toolkit` now includes everything
- **Improved package metadata** for better discoverability on PyPI
- **Enhanced keywords** for all benchmarks and providers
- **Multi-language support** - English and Spanish classifiers

## [2.2.0] - 2025-12-02

### Added

- **Interactive Web Dashboard**
  - Real-time evaluation progress monitoring
  - Model comparison charts (bar & radar)
  - Queue system for batch evaluations
  - Export to JSON, CSV, LaTeX, BibTeX
  - Scenario viewer with detailed Q&A inspection

- **Comprehensive Test Suite**
  - Added 100+ new tests across all modules
  - Test coverage increased to 79%
  - New test files for benchmarks, CLI, config, providers

- **Code Quality Improvements**
  - All code now passes `isort`, `black`, `ruff`, and `mypy` checks
  - Fixed type annotations throughout codebase
  - Improved import organization

### Changed

- Updated coverage configuration to exclude optional provider modules
- Improved benchmark demo modes for better testability

### Fixed

- Type annotation issues in dashboard and CLI modules
- Import sorting issues across all source files

---

## [2.0.0] - 2025-01-XX

### Added

#### Academic Evaluation Features

- **Statistical Metrics Module** (`statistical_metrics.py`)
  - Wilson score confidence intervals for accurate proportions
  - Bootstrap confidence intervals for complex metrics
  - McNemar's test for statistical significance between models
  - Cohen's h effect size calculation
  - Standard error calculations

- **Academic Baselines** (`academic_baselines.py`)
  - Published baselines for GPT-4, Claude 3 Opus, Llama 3 70B, Gemini Pro, Mistral Large
  - Complete with paper references and URLs
  - `compare_to_baseline()` function for easy comparison

- **Error Analysis** (`error_analysis.py`)
  - `ErrorAnalyzer` class for categorizing errors
  - Expected Calibration Error (ECE) calculation
  - Cohen's Kappa for inter-rater agreement

- **Export Utilities** (`export.py`)
  - `export_to_latex()` - Publication-ready LaTeX tables
  - `generate_bibtex()` - Citations for MMLU, TruthfulQA, HellaSwag
  - `generate_reproducibility_manifest()` - Full reproducibility info
  - `generate_methods_section()` - Template methods text for papers

- **Academic Evaluation Method** (`evaluator.py`)
  - `AcademicEvaluationResults` dataclass with CIs and baselines
  - `ModelEvaluator.evaluate_all_academic()` method

- **CLI Academic Command**
  - `llm-eval academic` command with options:
    - `--output-latex`: Export LaTeX table
    - `--output-bibtex`: Export BibTeX citations
    - `--compare-baselines`: Compare to published baselines

- **Documentation**
  - `docs/ACADEMIC_USAGE.md` - Comprehensive academic usage guide
  - `examples/academic_evaluation.py` - Full working example

### Changed

- All benchmark functions now return confidence intervals (`confidence_interval_95`, `standard_error`, `ci_width`)
- Updated README with academic features section
- Version bumped to 2.0.0

### Fixed

- Improved type safety throughout codebase

---

## [0.4.1] - 2025-01-XX

### Fixed

- Provider signature compatibility issues
- Test stability improvements

## [0.4.0] - 2025-01-XX

### Added

- Initial PyPI release as `llm-benchmark-toolkit`
- , TruthfulQA, HellaSwag benchmarks
- Ollama, OpenAI, Anthropic, HuggingFace providers
- CLI tool with `quick`, `run`, `compare`, `benchmark` commands
- Intelligent caching with `CachedProvider`
- Progress bars with ETA

## [0.3.0] - 2025-01-XX

### Added

- HuggingFace provider
- Visualization module
- Markdown report generation

## [0.2.0] - 2025-01-XX

### Added

- OpenAI and Anthropic providers
- Full benchmark datasets from HuggingFace

## [0.1.0] - 2025-01-XX

### Added

- Initial release
- Basic ModelEvaluator class
- Ollama provider
- Demo benchmarks
