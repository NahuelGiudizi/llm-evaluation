# Migration Guide

## ⚠️ Version 0.x and 2.0-2.2 are deprecated

If you're using versions older than **2.3.0**, please upgrade immediately:

```bash
pip install --upgrade llm-benchmark-toolkit
```

## Breaking Changes

### Version 0.x → 2.x

Major API restructuring. Key changes:

1. **Package renamed**: `llm-evaluator` → `llm-benchmark-toolkit`
2. **CLI commands consolidated**: Now all under `llm-eval`
3. **Dashboard**: New web UI at `llm-eval dashboard`
4. **Python API**: New class-based API

### Quick Migration

**Before (0.x):**
```python
from llm_evaluator import evaluate_model
results = evaluate_model("gpt-3.5-turbo")
```

**After (2.x):**
```python
from llm_evaluator import ModelEvaluator
from llm_evaluator.providers.openai_provider import OpenAIProvider

provider = OpenAIProvider(model="gpt-3.5-turbo")
evaluator = ModelEvaluator(provider)
results = evaluator.run_quick_evaluation()
```

## Version 2.0-2.2 → 2.3+

Minor fixes, no breaking changes. Just upgrade:

```bash
pip install --upgrade llm-benchmark-toolkit
```

## Support

- **Documentation**: [Full Guides](docs/)
- **Issues**: [GitHub Issues](https://github.com/NahuelGiudizi/llm-evaluation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NahuelGiudizi/llm-evaluation/discussions)
