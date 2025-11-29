"""Pytest configuration and fixtures for LLM Evaluation tests"""

import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response"""
    mock = Mock()
    mock.generate.return_value = {
        "response": "This is a test response",
        "total_duration": 1500000000,  # 1.5 seconds in nanoseconds
        "eval_count": 50,  # tokens generated
    }
    return mock


@pytest.fixture
def sample_evaluation_results():
    """Sample evaluation results for testing"""
    return {
        "model": "test-model",
        "performance": {
            "avg_response_time": 1.5,
            "tokens_per_second": 150.0,
            "total_requests": 10,
        },
        "quality": {
            "accuracy": 0.85,
            "coherence_score": 0.90,
            "hallucination_rate": 0.05,
        },
        "benchmarks": {
            "mmlu": {"score": 0.72, "total": 50, "correct": 36},
            "truthfulqa": {"score": 0.68, "total": 30, "correct": 20},
            "hellaswag": {"score": 0.75, "total": 40, "correct": 30},
        },
    }


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing"""
    return [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "What is 2+2?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
    ]
