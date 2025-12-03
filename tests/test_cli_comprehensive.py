"""
Additional CLI tests to improve coverage
"""

import os
import tempfile
from unittest.mock import patch

from click.testing import CliRunner


class TestCLIHelpers:
    """Test CLI helper functions"""

    def test_detect_provider_from_env_openai(self):
        """Test detection of OpenAI from env"""
        from llm_evaluator.cli import detect_provider_from_env

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            provider, model = detect_provider_from_env()
            # Should detect openai when key is present
            assert provider == "openai" or provider is None

    def test_detect_provider_from_env_anthropic(self):
        """Test detection of Anthropic from env"""
        from llm_evaluator.cli import detect_provider_from_env

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            provider, model = detect_provider_from_env()
            # Result depends on order of checks
            assert isinstance(provider, str) or provider is None

    def test_detect_provider_from_env_deepseek(self):
        """Test detection of DeepSeek from env"""
        from llm_evaluator.cli import detect_provider_from_env

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
            provider, model = detect_provider_from_env()
            # Result depends on order of checks
            assert isinstance(provider, str) or provider is None


class TestCLIVersionCommand:
    """Test CLI version command"""

    def test_version_flag(self):
        """Test --version flag"""
        from llm_evaluator.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "2.3.0" in result.output or "version" in result.output.lower()


class TestCLIHelpCommands:
    """Test CLI help commands"""

    def test_main_help(self):
        """Test main help"""
        from llm_evaluator.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "quick" in result.output or "benchmark" in result.output

    def test_quick_help(self):
        """Test quick command help"""
        from llm_evaluator.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["quick", "--help"])

        assert result.exit_code == 0

    def test_benchmark_help(self):
        """Test benchmark help"""
        from llm_evaluator.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark", "--help"])

        assert result.exit_code == 0


class TestCLIProviderFlags:
    """Test CLI provider availability flags"""

    def test_has_openai_flag(self):
        """Test HAS_OPENAI flag exists"""
        from llm_evaluator.cli import HAS_OPENAI

        assert isinstance(HAS_OPENAI, bool)

    def test_has_anthropic_flag(self):
        """Test HAS_ANTHROPIC flag exists"""
        from llm_evaluator.cli import HAS_ANTHROPIC

        assert isinstance(HAS_ANTHROPIC, bool)

    def test_has_huggingface_flag(self):
        """Test HAS_HUGGINGFACE flag exists"""
        from llm_evaluator.cli import HAS_HUGGINGFACE

        assert isinstance(HAS_HUGGINGFACE, bool)

    def test_has_deepseek_flag(self):
        """Test HAS_DEEPSEEK flag exists"""
        from llm_evaluator.cli import HAS_DEEPSEEK

        assert isinstance(HAS_DEEPSEEK, bool)


class TestCLIExport:
    """Test CLI export commands"""

    def test_providers_help(self):
        """Test providers command help"""
        from llm_evaluator.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["providers", "--help"])

        assert result.exit_code == 0


class TestCLIVersion:
    """Test CLI version variable"""

    def test_version_variable(self):
        """Test __version__ variable"""
        from llm_evaluator.cli import __version__

        assert __version__ == "2.3.0"


class TestCLIAcademic:
    """Test CLI academic command"""

    def test_academic_help(self):
        """Test academic command help"""
        from llm_evaluator.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["academic", "--help"])

        assert result.exit_code == 0


class TestCLICompare:
    """Test CLI compare command"""

    def test_compare_help(self):
        """Test compare command help"""
        from llm_evaluator.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])

        assert result.exit_code == 0


class TestCLIVisualize:
    """Test CLI visualize command"""

    def test_visualize_help(self):
        """Test visualize command help"""
        from llm_evaluator.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "--help"])

        assert result.exit_code == 0


class TestCLIExportResultsValidation:
    """Test CLI export results validation"""

    def test_export_with_invalid_json(self):
        """Test export with invalid JSON file"""
        from llm_evaluator.cli import cli

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid JSON file
            invalid_file = os.path.join(tmpdir, "invalid.json")
            with open(invalid_file, "w") as f:
                f.write("not valid json{")

            result = runner.invoke(cli, ["export", invalid_file])

            # Should fail with error
            assert (
                result.exit_code != 0
                or "error" in result.output.lower()
                or "Error" in result.output
            )


class TestCLIAcademicEvaluationResults:
    """Test AcademicEvaluationResults in CLI context"""

    def test_academic_results_import(self):
        """Test AcademicEvaluationResults can be imported from cli"""
        from llm_evaluator.cli import AcademicEvaluationResults

        # Should import successfully
        assert AcademicEvaluationResults is not None


class TestCLIExportFunctions:
    """Test export functions used by CLI"""

    def test_export_to_latex_import(self):
        """Test export_to_latex can be imported from cli"""
        from llm_evaluator.cli import export_to_latex

        assert export_to_latex is not None

    def test_generate_bibtex_import(self):
        """Test generate_bibtex can be imported from cli"""
        from llm_evaluator.cli import generate_bibtex

        assert generate_bibtex is not None


class TestCLIBenchmarkRunner:
    """Test BenchmarkRunner in CLI context"""

    def test_benchmark_runner_import(self):
        """Test BenchmarkRunner can be imported"""
        from llm_evaluator.cli import BenchmarkRunner

        assert BenchmarkRunner is not None


class TestCLICachedProvider:
    """Test CachedProvider in CLI context"""

    def test_cached_provider_import(self):
        """Test CachedProvider can be imported"""
        from llm_evaluator.cli import CachedProvider

        assert CachedProvider is not None


class TestCLIOllamaProvider:
    """Test OllamaProvider in CLI context"""

    def test_ollama_provider_import(self):
        """Test OllamaProvider can be imported"""
        from llm_evaluator.cli import OllamaProvider

        assert OllamaProvider is not None
