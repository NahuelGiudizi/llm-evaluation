"""Entry point for running the dashboard as a module.

Usage:
    python -m llm_evaluator.dashboard
    python -m llm_evaluator.dashboard --port 8080
    python -m llm_evaluator.dashboard --host 0.0.0.0 --port 8765
"""

import argparse
import sys


def main() -> None:
    """Run the dashboard server."""
    parser = argparse.ArgumentParser(
        description="LLM Benchmark Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m llm_evaluator.dashboard              # Start on localhost:8765
    python -m llm_evaluator.dashboard --port 8080  # Custom port
    python -m llm_evaluator.dashboard --host 0.0.0.0  # Allow external connections
        """,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind to (default: 8765)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup for --help
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    print("\n🚀 Starting LLM Benchmark Dashboard...")
    print(f"   URL: http://{args.host}:{args.port}")
    print("   Press Ctrl+C to stop\n")

    uvicorn.run(
        "llm_evaluator.dashboard.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
