"""
LLM Benchmark Dashboard

Web-based dashboard for running and visualizing LLM evaluations.
"""

from .app import create_app, run_dashboard

__all__ = ["create_app", "run_dashboard"]
