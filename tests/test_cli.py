"""
Tests for CLI module

Tests the command-line interface using Click's testing utilities.
"""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from llm_evaluator.cli import (
    cli,
    create_provider,
    detect_provider_from_env,
    echo_error,
    echo_info,
    echo_success,
    echo_warning,
)


class TestDetectProviderFromEnv:
    """Test auto-detection of providers from environment variables"""

    def test_detect_openai(self):
        """Test detection of OpenAI from OPENAI_API_KEY"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}, clear=True):
            provider, model = detect_provider_from_env()
            assert provider == "openai"
            assert model == "gpt-4o-mini"

    def test_detect_anthropic(self):
        """Test detection of Anthropic from ANTHROPIC_API_KEY"""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
            provider, model = detect_provider_from_env()
            assert provider == "anthropic"
            assert model == "claude-3-5-sonnet-20241022"

    def test_detect_deepseek(self):
        """Test detection of DeepSeek from DEEPSEEK_API_KEY"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-ds-test"}, clear=True):
            provider, model = detect_provider_from_env()
            assert provider == "deepseek"
            assert model == "deepseek-chat"

    def test_detect_huggingface_hf_token(self):
        """Test detection of HuggingFace from HF_TOKEN"""
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test123"}, clear=True):
            provider, model = detect_provider_from_env()
            assert provider == "huggingface"
            assert "llama" in model.lower()

    def test_detect_huggingface_api_key(self):
        """Test detection of HuggingFace from HUGGINGFACE_API_KEY"""
        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "hf_test123"}, clear=True):
            provider, model = detect_provider_from_env()
            assert provider == "huggingface"

    def test_detect_ollama_running(self):
        """Test detection of Ollama when server is running"""
        with patch.dict(os.environ, {}, clear=True):
            # Mock socket connection and urllib request
            with patch("socket.socket") as mock_socket:
                mock_sock_instance = Mock()
                mock_sock_instance.connect_ex.return_value = 0  # Port is open
                mock_socket.return_value = mock_sock_instance

                with patch("urllib.request.urlopen") as mock_urlopen:
                    # Mock response for /api/tags
                    mock_response = Mock()
                    mock_response.__enter__ = Mock(return_value=mock_response)
                    mock_response.__exit__ = Mock(return_value=False)
                    mock_response.read.return_value = b'{"models": [{"name": "llama3.2:1b"}]}'
                    mock_urlopen.return_value = mock_response

                    provider, model = detect_provider_from_env()
                    assert provider == "ollama"
                    assert model == "llama3.2:1b"

    def test_detect_none_when_no_provider(self):
        """Test returns None when no provider is available"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("socket.socket") as mock_socket:
                mock_sock_instance = Mock()
                mock_sock_instance.connect_ex.return_value = 1  # Port closed
                mock_socket.return_value = mock_sock_instance

                provider, model = detect_provider_from_env()
                assert provider is None
                assert model is None

    def test_priority_openai_over_others(self):
        """Test that OpenAI has priority when multiple keys exist"""
        env = {
            "OPENAI_API_KEY": "sk-openai",
            "ANTHROPIC_API_KEY": "sk-ant",
            "DEEPSEEK_API_KEY": "sk-ds",
        }
        with patch.dict(os.environ, env, clear=True):
            provider, _ = detect_provider_from_env()
            assert provider == "openai"


class TestCreateProvider:
    """Test provider creation"""

    def test_create_ollama_provider(self):
        """Test creating Ollama provider"""
        with patch("llm_evaluator.cli.OllamaProvider") as mock_provider:
            mock_instance = Mock()
            mock_provider.return_value = mock_instance

            provider = create_provider("llama3.2:1b", "ollama", cache=False)

            mock_provider.assert_called_once_with(model="llama3.2:1b", base_url=None)
            assert provider == mock_instance

    def test_create_ollama_provider_with_base_url(self):
        """Test creating Ollama provider with custom base URL"""
        with patch("llm_evaluator.cli.OllamaProvider") as mock_provider:
            mock_instance = Mock()
            mock_provider.return_value = mock_instance

            provider = create_provider(
                "llama3.2:1b", "ollama", cache=False, base_url="http://localhost:8080"
            )

            mock_provider.assert_called_once_with(
                model="llama3.2:1b", base_url="http://localhost:8080"
            )
            assert provider == mock_instance

    def test_create_provider_with_cache(self):
        """Test creating provider with caching enabled"""
        with patch("llm_evaluator.cli.OllamaProvider") as mock_ollama:
            with patch("llm_evaluator.cli.CachedProvider") as mock_cached:
                mock_base = Mock()
                mock_ollama.return_value = mock_base
                mock_cached_instance = Mock()
                mock_cached.return_value = mock_cached_instance

                provider = create_provider("llama3.2:1b", "ollama", cache=True)

                mock_cached.assert_called_once_with(mock_base)
                assert provider == mock_cached_instance

    def test_create_openai_provider_not_installed(self):
        """Test error when OpenAI not installed"""
        with patch("llm_evaluator.cli.HAS_OPENAI", False):
            with pytest.raises(SystemExit):
                create_provider("gpt-4", "openai")

    def test_create_auto_provider_success(self):
        """Test auto provider detection and creation"""
        with patch("llm_evaluator.cli.detect_provider_from_env") as mock_detect:
            mock_detect.return_value = ("ollama", "llama3.2:1b")
            with patch("llm_evaluator.cli.OllamaProvider") as mock_ollama:
                mock_instance = Mock()
                mock_ollama.return_value = mock_instance

                create_provider("auto", "auto", cache=False)

                mock_ollama.assert_called_once()

    def test_create_auto_provider_no_detection(self):
        """Test error when auto detection fails"""
        with patch("llm_evaluator.cli.detect_provider_from_env") as mock_detect:
            mock_detect.return_value = (None, None)

            with pytest.raises(SystemExit):
                create_provider("auto", "auto")


class TestEchoFunctions:
    """Test colored echo functions"""

    def test_echo_success(self, capsys):
        """Test success message output"""
        echo_success("Test message")
        # Function outputs to click.echo which may not be captured by capsys
        # Just verify no exception is raised

    def test_echo_error(self, capsys):
        """Test error message output"""
        echo_error("Error message")

    def test_echo_warning(self, capsys):
        """Test warning message output"""
        echo_warning("Warning message")

    def test_echo_info(self, capsys):
        """Test info message output"""
        echo_info("Info message")


class TestCLICommands:
    """Test CLI commands using Click's CliRunner"""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_cli_version(self, runner):
        """Test --version flag"""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "llm-eval" in result.output
        assert "2." in result.output  # Version 2.x

    def test_cli_help(self, runner):
        """Test --help flag"""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "LLM Evaluation Suite" in result.output
        assert "quick" in result.output
        assert "run" in result.output
        assert "compare" in result.output
        assert "benchmark" in result.output
        assert "academic" in result.output

    def test_providers_command(self, runner):
        """Test providers command"""
        result = runner.invoke(cli, ["providers"])
        assert result.exit_code == 0
        assert "Available Providers" in result.output
        assert "ollama" in result.output
        assert "Environment Variables" in result.output

    def test_quick_command_no_provider(self, runner):
        """Test quick command when no provider is available"""
        with patch("llm_evaluator.cli.detect_provider_from_env") as mock_detect:
            mock_detect.return_value = (None, None)

            result = runner.invoke(cli, ["quick"])

            assert result.exit_code == 1
            assert "No provider detected" in result.output

    def test_quick_command_success(self, runner):
        """Test quick command with mocked provider"""
        with patch("llm_evaluator.cli.detect_provider_from_env") as mock_detect:
            mock_detect.return_value = ("ollama", "llama3.2:1b")

            with patch("llm_evaluator.cli.create_provider") as mock_create:
                mock_provider = Mock()
                mock_provider.is_available.return_value = True
                mock_create.return_value = mock_provider

                with patch("llm_evaluator.cli.BenchmarkRunner") as mock_runner:
                    mock_runner_instance = Mock()
                    mock_runner.return_value = mock_runner_instance
                    mock_runner_instance.run_mmlu_sample.return_value = {"mmlu_accuracy": 0.5}
                    mock_runner_instance.run_truthfulqa_sample.return_value = {
                        "truthfulness_score": 0.4
                    }
                    mock_runner_instance.run_hellaswag_sample.return_value = {
                        "hellaswag_accuracy": 0.6
                    }

                    result = runner.invoke(cli, ["quick"])

                    assert result.exit_code == 0
                    assert "QUICK EVALUATION" in result.output
                    assert "RESULTS" in result.output

    def test_run_command_help(self, runner):
        """Test run command help"""
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--provider" in result.output
        assert "--cache" in result.output

    def test_compare_command_help(self, runner):
        """Test compare command help"""
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0
        assert "--models" in result.output

    def test_benchmark_command_help(self, runner):
        """Test benchmark command help"""
        result = runner.invoke(cli, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "--benchmarks" in result.output
        assert "--sample-size" in result.output
        assert "--full" in result.output

    def test_academic_command_help(self, runner):
        """Test academic command help"""
        result = runner.invoke(cli, ["academic", "--help"])
        assert result.exit_code == 0
        assert "--output-latex" in result.output
        assert "--output-bibtex" in result.output
        assert "--compare-baselines" in result.output
        assert "statistical rigor" in result.output

    def test_visualize_command_help(self, runner):
        """Test visualize command help"""
        result = runner.invoke(cli, ["visualize", "--help"])
        assert result.exit_code == 0
        assert "visualizations" in result.output.lower()


class TestCLIIntegration:
    """Integration tests for CLI commands"""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_run_with_unavailable_provider(self, runner):
        """Test run command when provider is not available"""
        with patch("llm_evaluator.cli.create_provider") as mock_create:
            mock_provider = Mock()
            mock_provider.is_available.return_value = False
            mock_create.return_value = mock_provider

            result = runner.invoke(cli, ["run", "--model", "test", "--provider", "ollama"])

            assert result.exit_code == 1
            assert "not available" in result.output

    def test_benchmark_unknown_benchmark(self, runner):
        """Test benchmark command with unknown benchmark name"""
        with patch("llm_evaluator.cli.create_provider") as mock_create:
            mock_provider = Mock()
            mock_provider.is_available.return_value = True
            mock_create.return_value = mock_provider

            with patch("llm_evaluator.cli.BenchmarkRunner") as mock_runner:
                mock_runner_instance = Mock()
                mock_runner.return_value = mock_runner_instance

                result = runner.invoke(
                    cli, ["benchmark", "--model", "test", "--benchmarks", "unknown_benchmark"]
                )

                assert "Unknown benchmark" in result.output

    def test_quick_with_output_file(self, runner):
        """Test quick command saves results to file"""
        with patch("llm_evaluator.cli.detect_provider_from_env") as mock_detect:
            mock_detect.return_value = ("ollama", "llama3.2:1b")

            with patch("llm_evaluator.cli.create_provider") as mock_create:
                mock_provider = Mock()
                mock_provider.is_available.return_value = True
                mock_create.return_value = mock_provider

                with patch("llm_evaluator.cli.BenchmarkRunner") as mock_runner:
                    mock_runner_instance = Mock()
                    mock_runner.return_value = mock_runner_instance
                    mock_runner_instance.run_mmlu_sample.return_value = {"mmlu_accuracy": 0.5}
                    mock_runner_instance.run_truthfulqa_sample.return_value = {
                        "truthfulness_score": 0.4
                    }
                    mock_runner_instance.run_hellaswag_sample.return_value = {
                        "hellaswag_accuracy": 0.6
                    }

                    with runner.isolated_filesystem():
                        result = runner.invoke(cli, ["quick", "--output", "results.json"])

                        assert result.exit_code == 0
                        assert Path("results.json").exists()

                        data = json.loads(Path("results.json").read_text())
                        assert "model" in data
                        assert "results" in data
