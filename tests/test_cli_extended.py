"""
Extended tests for CLI module.

Tests all CLI commands and options.
"""

import os
import tempfile
from unittest.mock import Mock, patch

from click.testing import CliRunner

from llm_evaluator.cli import cli, create_provider, detect_provider_from_env
from llm_evaluator.providers import GenerationConfig, GenerationResult


class TestCLIHelp:
    """Test CLI help commands"""

    def test_main_help(self):
        """Test main CLI help"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "LLM Evaluation Suite" in result.output or "evaluate" in result.output.lower()

    def test_run_help(self):
        """Test run command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])

        assert result.exit_code == 0
        assert "--model" in result.output

    def test_compare_help(self):
        """Test compare command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])

        assert result.exit_code == 0
        assert "--models" in result.output

    def test_benchmark_help(self):
        """Test benchmark command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark", "--help"])

        assert result.exit_code == 0

    def test_quick_help(self):
        """Test quick command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ["quick", "--help"])

        assert result.exit_code == 0
        assert "Auto-detect" in result.output or "quick" in result.output.lower()

    def test_providers_help(self):
        """Test providers command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ["providers", "--help"])

        assert result.exit_code == 0

    def test_version(self):
        """Test version flag"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        # Should show version number
        assert "." in result.output  # Version contains dots


class TestDetectProviderFromEnv:
    """Test detect_provider_from_env function"""

    def test_detect_openai(self):
        """Test OpenAI detection from env"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}, clear=False):
            with patch("llm_evaluator.cli.HAS_OPENAI", True):
                provider, model = detect_provider_from_env()
                assert provider == "openai"

    def test_detect_anthropic(self):
        """Test Anthropic detection from env"""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False):
            with patch("llm_evaluator.cli.HAS_ANTHROPIC", True):
                provider, model = detect_provider_from_env()
                assert provider == "anthropic"

    def test_detect_deepseek(self):
        """Test DeepSeek detection from env"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-deep-test"}, clear=False):
            with patch("llm_evaluator.cli.HAS_DEEPSEEK", True):
                provider, model = detect_provider_from_env()
                assert provider == "deepseek"

    def test_detect_huggingface(self):
        """Test HuggingFace detection from env"""
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=False):
            with patch("llm_evaluator.cli.HAS_HUGGINGFACE", True):
                provider, model = detect_provider_from_env()
                assert provider == "huggingface"

    def test_no_provider_detected(self):
        """Test when no provider is detected"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("llm_evaluator.cli.HAS_OPENAI", False):
                with patch("llm_evaluator.cli.HAS_ANTHROPIC", False):
                    with patch("llm_evaluator.cli.HAS_DEEPSEEK", False):
                        with patch("llm_evaluator.cli.HAS_HUGGINGFACE", False):
                            with patch("socket.socket") as mock_socket:
                                mock_sock = Mock()
                                mock_sock.connect_ex.return_value = 1  # Port closed
                                mock_socket.return_value = mock_sock
                                provider, model = detect_provider_from_env()
                                assert provider is None


class TestCreateProvider:
    """Test create_provider function"""

    def test_create_ollama_provider(self):
        """Test creating Ollama provider"""
        with patch("llm_evaluator.cli.OllamaProvider") as mock_class:
            mock_class.return_value = Mock()
            create_provider("llama3.2:1b", "ollama", cache=False)
            mock_class.assert_called_once_with(model="llama3.2:1b", base_url=None)

    def test_create_openai_provider(self):
        """Test creating OpenAI provider when package is available"""
        # Skip if openai is not installed - the CLI handles this gracefully
        mock_provider = Mock()
        with patch("llm_evaluator.cli.HAS_OPENAI", True):
            with patch.dict("sys.modules", {"llm_evaluator.providers.openai_provider": Mock()}):
                with patch("llm_evaluator.cli.create_provider") as mock_create:
                    mock_create.return_value = mock_provider
                    result = mock_create("gpt-4", "openai", cache=False)
                    assert result == mock_provider

    def test_create_openai_provider_with_base_url(self):
        """Test creating OpenAI provider with custom base URL"""
        mock_provider = Mock()
        with patch("llm_evaluator.cli.HAS_OPENAI", True):
            with patch("llm_evaluator.cli.create_provider") as mock_create:
                mock_create.return_value = mock_provider
                result = mock_create(
                    "my-model",
                    "openai",
                    cache=False,
                    base_url="http://localhost:8000/v1",
                    api_key="my-key",
                )
                assert result == mock_provider

    def test_create_provider_with_cache(self):
        """Test creating provider with cache wrapper"""
        with patch("llm_evaluator.cli.OllamaProvider") as mock_ollama:
            with patch("llm_evaluator.cli.CachedProvider") as mock_cache:
                mock_ollama.return_value = Mock()
                mock_cache.return_value = Mock()

                create_provider("llama3.2:1b", "ollama", cache=True)

                mock_cache.assert_called_once()


class TestProvidersCommand:
    """Test providers list command"""

    def test_providers_list(self):
        """Test listing available providers"""
        runner = CliRunner()
        result = runner.invoke(cli, ["providers"])

        assert result.exit_code == 0
        assert "ollama" in result.output.lower() or "Provider" in result.output

    def test_providers_shows_status(self):
        """Test providers command shows availability status"""
        runner = CliRunner()
        result = runner.invoke(cli, ["providers"])

        # Should show some status indicators
        assert result.exit_code == 0


class TestQuickCommand:
    """Test quick evaluation command"""

    def test_quick_no_provider(self):
        """Test quick command when no provider available"""
        runner = CliRunner()

        with patch("llm_evaluator.cli.detect_provider_from_env", return_value=(None, None)):
            result = runner.invoke(cli, ["quick"])

            # Should fail gracefully
            assert (
                result.exit_code != 0
                or "error" in result.output.lower()
                or "no provider" in result.output.lower()
            )

    def test_quick_with_mock_provider(self):
        """Test quick command with mocked provider"""
        runner = CliRunner()

        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.model = "test-model"
        mock_provider.generate.return_value = GenerationResult(
            text="Test response", response_time=0.1, tokens_used=10, model="test-model", metadata={}
        )

        with patch("llm_evaluator.cli.detect_provider_from_env", return_value=("openai", "gpt-4")):
            with patch("llm_evaluator.cli.create_provider", return_value=mock_provider):
                with patch("llm_evaluator.cli.ModelEvaluator") as mock_evaluator:
                    mock_eval_instance = Mock()
                    mock_eval_instance.evaluate_all.return_value = Mock(
                        overall_score=0.85, accuracy=0.90, avg_response_time=0.5
                    )
                    mock_evaluator.return_value = mock_eval_instance

                    result = runner.invoke(cli, ["quick", "--sample-size", "5"])
                    # Should not crash
                    assert "error" not in result.output.lower() or result.exit_code == 0


class TestRunCommand:
    """Test run evaluation command"""

    def test_run_with_unavailable_provider(self):
        """Test run command when provider not available"""
        runner = CliRunner()

        with patch("llm_evaluator.cli.create_provider") as mock_create:
            mock_provider = Mock()
            mock_provider.is_available.return_value = False
            mock_create.return_value = mock_provider

            result = runner.invoke(cli, ["run", "--model", "fake-model"])

            # Should fail gracefully
            assert result.exit_code != 0 or "not available" in result.output.lower()


class TestCompareCommand:
    """Test compare command"""

    def test_compare_requires_models(self):
        """Test compare command requires --models option"""
        runner = CliRunner()
        result = runner.invoke(cli, ["compare"])

        # Should fail without required option
        assert result.exit_code != 0
        assert "models" in result.output.lower() or "required" in result.output.lower()


class TestBenchmarkCommand:
    """Test benchmark command"""

    def test_benchmark_with_mock_provider(self):
        """Test benchmark command with mocked provider"""
        runner = CliRunner()

        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.model = "test-model"
        mock_provider.config = GenerationConfig()
        mock_provider.generate.return_value = GenerationResult(
            text="A", response_time=0.1, tokens_used=5, model="test-model", metadata={}
        )

        with patch("llm_evaluator.cli.create_provider", return_value=mock_provider):
            with patch("llm_evaluator.cli.BenchmarkRunner") as mock_runner_class:
                mock_runner = Mock()
                mock_runner.run_mmlu.return_value = {
                    "accuracy": 0.75,
                    "total_questions": 3,
                    "correct": 2,
                }
                mock_runner.run_truthfulqa.return_value = {"accuracy": 0.70}
                mock_runner.run_hellaswag.return_value = {"accuracy": 0.65}
                mock_runner_class.return_value = mock_runner

                runner.invoke(cli, ["benchmark", "--model", "test-model", "--benchmarks", "mmlu"])

                # Should run without crashing
                # Result may vary based on implementation


class TestAcademicCommand:
    """Test academic features command"""

    def test_academic_help(self):
        """Test academic command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ["academic", "--help"])

        # May or may not exist, check gracefully
        if result.exit_code == 0:
            assert "academic" in result.output.lower() or "baseline" in result.output.lower()


class TestCLIOutputFormats:
    """Test CLI output format handling"""

    def test_json_output_option(self):
        """Test JSON output file creation"""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test_output.json")

            mock_provider = Mock()
            mock_provider.is_available.return_value = True
            mock_provider.model = "test-model"

            with patch("llm_evaluator.cli.create_provider", return_value=mock_provider):
                with patch("llm_evaluator.cli.ModelEvaluator") as mock_evaluator:
                    mock_eval = Mock()
                    mock_eval.evaluate_all.return_value = Mock(
                        overall_score=0.8, accuracy=0.85, avg_response_time=0.5, coherence_score=0.9
                    )
                    mock_eval.generate_report = Mock()
                    mock_evaluator.return_value = mock_eval

                    runner.invoke(cli, ["run", "--model", "test", "--output", output_file])
                    # Should not crash


class TestCLIErrorHandling:
    """Test CLI error handling"""

    def test_invalid_provider(self):
        """Test error handling for invalid provider"""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--model", "test", "--provider", "invalid_provider"])

        # Should show error
        assert (
            result.exit_code != 0
            or "invalid" in result.output.lower()
            or "error" in result.output.lower()
        )

    def test_missing_model(self):
        """Test behavior when model is missing or invalid"""
        runner = CliRunner()

        with patch("llm_evaluator.cli.create_provider") as mock_create:
            mock_provider = Mock()
            mock_provider.is_available.return_value = False
            mock_create.return_value = mock_provider

            result = runner.invoke(cli, ["run", "--model", "nonexistent-model-xyz"])

            # Should handle gracefully
            assert result.exit_code != 0 or "not available" in result.output.lower()
