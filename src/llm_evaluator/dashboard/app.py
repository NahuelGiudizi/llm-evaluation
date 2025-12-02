"""
Dashboard Application

Self-contained FastAPI application for the LLM Benchmark Dashboard.
"""

import json
import logging
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from threading import Timer
from typing import Any, Dict, List, Optional

# Check for required dependencies
try:
    from fastapi import FastAPI, HTTPException, Query  # noqa: F401
    from fastapi.middleware.cors import CORSMiddleware  # noqa: F401
    from fastapi.responses import FileResponse, JSONResponse  # noqa: F401
    from fastapi.staticfiles import StaticFiles  # noqa: F401

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    import uvicorn  # noqa: F401

    HAS_UVICORN = True
except ImportError:
    HAS_UVICORN = False

logger = logging.getLogger(__name__)

# Paths
PACKAGE_DIR = Path(__file__).parent
SRC_DIR = PACKAGE_DIR.parent
BASE_DIR = SRC_DIR.parent.parent  # llm-evaluation root
STATIC_DIR = PACKAGE_DIR / "static"  # Built UI files (included in package)

# Default outputs directory (shared with CLI via config)
DEFAULT_OUTPUTS_DIR = Path.home() / ".llm-benchmark" / "outputs"


def check_dependencies() -> list[str]:
    """Check if dashboard dependencies are installed"""
    missing: list[str] = []
    if not HAS_FASTAPI:
        missing.append("fastapi")
    if not HAS_UVICORN:
        missing.append("uvicorn")

    try:
        import sse_starlette  # noqa: F401
    except ImportError:
        missing.append("sse-starlette")

    try:
        import psutil  # noqa: F401
    except ImportError:
        missing.append("psutil")

    return missing


def create_app(outputs_dir: Optional[Path] = None) -> "FastAPI":
    """
    Create the FastAPI application for the dashboard.

    Args:
        outputs_dir: Directory to store evaluation outputs (default: ~/.llm-benchmark/outputs)

    Returns:
        Configured FastAPI application
    """
    missing = check_dependencies()
    if missing:
        raise ImportError(
            f"Dashboard requires additional dependencies: {', '.join(missing)}\n"
            f"Install with: pip install llm-benchmark-toolkit[dashboard]"
        )

    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from sse_starlette.sse import EventSourceResponse

    # Use provided outputs dir or default
    outputs_path = outputs_dir or DEFAULT_OUTPUTS_DIR
    outputs_path.mkdir(parents=True, exist_ok=True)

    # Import runner (lazy import to avoid circular deps)
    from .models import (
        Benchmark,
        DeleteRunsRequest,
        DeleteRunsResponse,
        Preset,
        QueueRequest,
        QueueStatus,
        RunRequest,
        RunResponse,
        RunsResponse,
        RunStatus,
        RunSummary,
    )
    from .runner import EvaluationRunner

    # Create FastAPI app
    app = FastAPI(
        title="LLM Benchmark Dashboard",
        description="Web dashboard for LLM evaluation",
        version="2.1.0",
    )

    # CORS for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize runner
    runner = EvaluationRunner(outputs_path)

    # =========================================================================
    # API Endpoints
    # =========================================================================

    @app.get("/api/health")
    async def health_check() -> Dict[str, str]:
        return {"status": "healthy", "version": "2.1.0"}

    @app.get("/api/models")
    async def get_models() -> Any:
        """Get available models from all providers"""
        from .model_discovery import discover_models

        return discover_models()

    @app.get("/api/model-info/{provider}/{model:path}")
    async def get_model_info(provider: str, model: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        try:
            if provider == "ollama":
                import ollama

                try:
                    info = ollama.show(model)
                    details = info.get("details", {})
                    model_info = info.get("model_info", {})

                    # Extract architecture information
                    architecture = {}

                    # Parameter count - try multiple sources
                    param_count = details.get("parameter_size", "")
                    if not param_count and model_info:
                        # Try to find parameter count from model_info keys
                        for key in model_info:
                            if "parameter" in key.lower() and "count" in key.lower():
                                param_count = str(model_info[key])
                                break
                    if param_count:
                        architecture["parameter_count"] = param_count

                    # Quantization level
                    quant = details.get("quantization_level", "")
                    if quant:
                        architecture["quantization"] = quant

                    # Family/format
                    family = details.get("family", "")
                    if family:
                        architecture["family"] = family

                    # Format
                    fmt = details.get("format", "")
                    if fmt:
                        architecture["format"] = fmt

                    # Extract from model_info if available
                    if model_info:
                        # Common Ollama model_info keys
                        arch_keys = {
                            "embedding_length": ["llama.embedding_length", "embedding_length"],
                            "attention_heads": [
                                "llama.attention.head_count",
                                "attention_heads",
                                "num_attention_heads",
                            ],
                            "layers": ["llama.block_count", "block_count", "num_hidden_layers"],
                            "context_length": [
                                "llama.context_length",
                                "context_length",
                                "max_position_embeddings",
                            ],
                            "vocab_size": ["llama.vocab_size", "vocab_size"],
                        }

                        for target_key, source_keys in arch_keys.items():
                            for source_key in source_keys:
                                if source_key in model_info:
                                    if target_key == "context_length":
                                        architecture["context_length"] = model_info[source_key]
                                    else:
                                        architecture[target_key] = model_info[source_key]
                                    break

                    # Build response
                    result: Dict[str, Any] = {
                        "model": model,
                        "provider": provider,
                    }

                    if architecture:
                        result["architecture"] = architecture

                    # Context info
                    if "context_length" in architecture:
                        result["context"] = {"context_length": architecture.pop("context_length")}

                    # Include raw details for debugging
                    result["raw_details"] = details

                    return result
                except Exception as e:
                    logger.warning(f"Could not get Ollama model info: {e}")
                    return {"model": model, "provider": provider, "error": str(e)}
            else:
                # For other providers, return basic info
                return {"model": model, "provider": provider}
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/benchmarks")
    async def get_benchmarks() -> List[Benchmark]:
        """Get available benchmarks"""
        return [
            # Core benchmarks
            Benchmark(
                id="mmlu",
                name="MMLU",
                description="Massive Multitask Language Understanding - 14K questions across 57 subjects",
                category="Knowledge",
                questions_count=14042,
            ),
            Benchmark(
                id="truthfulqa",
                name="TruthfulQA",
                description="Tests model truthfulness on 817 questions designed to cause false answers",
                category="Truthfulness",
                questions_count=817,
            ),
            Benchmark(
                id="hellaswag",
                name="HellaSwag",
                description="Commonsense reasoning with 10K scenarios",
                category="Reasoning",
                questions_count=10042,
            ),
            # Knowledge benchmarks
            Benchmark(
                id="arc",
                name="ARC-Challenge",
                description="AI2 Reasoning Challenge - 2.5K science questions requiring multi-step reasoning",
                category="Knowledge",
                questions_count=2590,
            ),
            Benchmark(
                id="winogrande",
                name="WinoGrande",
                description="Pronoun resolution requiring commonsense - 44K fill-in-the-blank scenarios",
                category="Reasoning",
                questions_count=44000,
            ),
            Benchmark(
                id="commonsenseqa",
                name="CommonsenseQA",
                description="Commonsense knowledge questions requiring background knowledge - 12K questions",
                category="Knowledge",
                questions_count=12247,
            ),
            Benchmark(
                id="boolq",
                name="BoolQ",
                description="Boolean questions from real Google queries - 16K yes/no questions",
                category="Knowledge",
                questions_count=15942,
            ),
            # Security benchmarks
            Benchmark(
                id="safetybench",
                name="SafetyBench",
                description="Safety and ethics evaluation - 11K safety-related questions",
                category="Security",
                questions_count=11000,
            ),
            Benchmark(
                id="donotanswer",
                name="Do-Not-Answer",
                description="Harmful prompt refusal detection - 939 prompts models should refuse",
                category="Security",
                questions_count=939,
            ),
        ]

    @app.get("/api/presets")
    async def get_presets() -> List[Preset]:
        """Get evaluation presets"""
        return [
            Preset(
                id="quick",
                name="Quick Test",
                description="Fast sanity check with MMLU only",
                sample_size=10,
                benchmarks=["mmlu"],
                category="Speed",
            ),
            Preset(
                id="knowledge",
                name="Knowledge",
                description="Academic knowledge evaluation",
                sample_size=30,
                benchmarks=["mmlu", "arc", "commonsenseqa"],
                category="Knowledge",
            ),
            Preset(
                id="reasoning",
                name="Reasoning",
                description="Logical and commonsense reasoning",
                sample_size=30,
                benchmarks=["hellaswag", "winogrande", "boolq"],
                category="Reasoning",
            ),
            Preset(
                id="safety",
                name="Safety & Trust",
                description="Truthfulness and safety evaluation",
                sample_size=30,
                benchmarks=["truthfulqa", "safetybench", "donotanswer"],
                category="Safety",
            ),
            Preset(
                id="full",
                name="Full Suite",
                description="All 9 benchmarks - comprehensive evaluation",
                sample_size=50,
                benchmarks=[
                    "mmlu",
                    "truthfulqa",
                    "hellaswag",
                    "arc",
                    "winogrande",
                    "commonsenseqa",
                    "boolq",
                    "safetybench",
                    "donotanswer",
                ],
                category="Complete",
            ),
        ]

    @app.post("/api/run")
    async def start_run(request: RunRequest) -> RunResponse:
        """Start a new evaluation run"""
        sample_size = request.sample_size or 20

        run_id = runner.start_run(
            model=request.model,
            provider=request.provider,
            benchmarks=request.benchmarks,
            preset="custom",
            sample_size=sample_size,
            inference_settings=(
                request.inference_settings.model_dump() if request.inference_settings else None
            ),
            base_url=request.base_url,
            api_key=request.api_key,
        )

        return RunResponse(
            run_id=run_id,
            status=RunStatus.PENDING,
            message=f"Started evaluation of {request.model}",
        )

    @app.get("/api/runs")
    async def get_runs(limit: int = Query(50, ge=1, le=200)) -> RunsResponse:
        """Get all evaluation runs"""
        runs = runner.get_runs()

        summaries = []
        for r in runs[:limit]:
            # Calculate scores
            results = r.get("results", {})
            scores = []
            for name, data in results.items():
                if isinstance(data, dict):
                    score = (
                        data.get("mmlu_accuracy")
                        or data.get("truthfulness_score")
                        or data.get("hellaswag_accuracy")
                    )
                    if score is not None:
                        scores.append(score)

            overall_score = sum(scores) / len(scores) if scores else None

            # Parse dates
            started = r.get("started_at")
            completed = r.get("completed_at")
            if isinstance(started, str):
                started = datetime.fromisoformat(started)
            if isinstance(completed, str):
                completed = datetime.fromisoformat(completed)

            duration = None
            if started and completed:
                duration = (completed - started).total_seconds()

            summaries.append(
                RunSummary(
                    run_id=r["run_id"],
                    model=r.get("model", "unknown"),
                    provider=r.get("provider", "unknown"),
                    status=r.get("status", RunStatus.COMPLETED),
                    started_at=started if isinstance(started, datetime) else datetime.now(),
                    completed_at=completed,
                    duration_seconds=duration,
                    benchmarks=r.get("benchmarks", []),
                    preset=r.get("preset"),
                    sample_size=r.get("sample_size"),
                    overall_score=overall_score,
                    results=results,
                    system_info=r.get("system_info", {}),
                    inference_settings=r.get("inference_settings", {}),
                    file_path=r.get("file_path"),
                    filename=r.get("filename"),
                )
            )

        return RunsResponse(runs=summaries, total=len(runs))

    @app.get("/api/run/{run_id}")
    async def get_run(run_id: str) -> Any:
        """Get details of a specific run"""
        run = runner.get_run(run_id)
        if not run:
            # Try loading from disk
            json_file = outputs_path / f"{run_id}.json"
            if json_file.exists():
                with open(json_file) as f:
                    data = json.load(f)
                # Normalize status to simple string
                if "status" in data:
                    status = str(data["status"]).lower()
                    if "completed" in status or "complete" in status:
                        data["status"] = "completed"
                    elif "running" in status:
                        data["status"] = "running"
                    elif "failed" in status or "error" in status:
                        data["status"] = "failed"
                    elif "cancelled" in status or "canceled" in status:
                        data["status"] = "cancelled"
                    elif "pending" in status:
                        data["status"] = "pending"
                return data
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.get("/api/run/{run_id}/scenarios")
    async def get_scenarios(
        run_id: str,
        benchmark: Optional[str] = None,
        filter: Optional[str] = None,
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
    ) -> Dict[str, Any]:
        """Get evaluated scenarios for a run"""
        json_file = outputs_path / f"{run_id}.json"
        if not json_file.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        with open(json_file) as f:
            data = json.load(f)

        results = data.get("results", {})
        all_scenarios = []

        for bench_name, bench_data in results.items():
            if benchmark and bench_name.lower() != benchmark.lower():
                continue
            scenarios = bench_data.get("scenarios", [])
            for s in scenarios:
                s["benchmark"] = bench_name
                all_scenarios.append(s)

        if filter == "correct":
            all_scenarios = [s for s in all_scenarios if s.get("is_correct", False)]
        elif filter == "incorrect":
            all_scenarios = [s for s in all_scenarios if not s.get("is_correct", False)]

        total = len(all_scenarios)
        total_pages = (total + page_size - 1) // page_size
        start = (page - 1) * page_size
        end = start + page_size

        correct_count = sum(1 for s in all_scenarios if s.get("is_correct", False))

        return {
            "run_id": run_id,
            "benchmark": benchmark,
            "filter": filter,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "correct_count": correct_count,
            "incorrect_count": total - correct_count,
            "scenarios": all_scenarios[start:end],
        }

    @app.post("/api/run/{run_id}/cancel")
    async def cancel_run(run_id: str) -> Dict[str, str]:
        """Cancel a running evaluation"""
        success = runner.cancel_run(run_id)
        if not success:
            raise HTTPException(status_code=404, detail="Run not found or already completed")
        return {"run_id": run_id, "status": "cancelled"}

    @app.delete("/api/run/{run_id}")
    async def delete_run(run_id: str) -> Dict[str, str]:
        """Delete a single evaluation run"""
        json_file = outputs_path / f"{run_id}.json"
        if not json_file.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        try:
            json_file.unlink()
            # Also remove from in-memory cache if present
            runner.remove_run(run_id)
            return {"run_id": run_id, "status": "deleted"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete run: {str(e)}")

    @app.delete("/api/runs")
    async def delete_runs(request: DeleteRunsRequest) -> DeleteRunsResponse:
        """Delete multiple evaluation runs"""
        deleted: list[str] = []
        failed: list[str] = []

        for run_id in request.run_ids:
            json_file = outputs_path / f"{run_id}.json"
            try:
                if json_file.exists():
                    json_file.unlink()
                    runner.remove_run(run_id)
                    deleted.append(run_id)
                else:
                    failed.append(run_id)
            except Exception:
                failed.append(run_id)

        return DeleteRunsResponse(
            deleted=deleted,
            failed=failed,
            message=f"Deleted {len(deleted)} runs, {len(failed)} failed",
        )

    # =========================================================================
    # Queue Endpoints
    # =========================================================================

    @app.post("/api/queue")
    async def start_queue(request: QueueRequest) -> Dict[str, Any]:
        """Start a queue of sequential evaluations"""
        queue_id = runner.start_queue(request.runs)
        return {
            "queue_id": queue_id,
            "status": "started",
            "total": len(request.runs),
            "message": f"Started queue with {len(request.runs)} evaluations",
        }

    @app.get("/api/queue/status")
    async def get_queue_status() -> Optional[QueueStatus]:
        """Get current queue status"""
        status = runner.get_queue_status()
        if not status:
            return None
        return status

    @app.delete("/api/queue")
    async def cancel_queue() -> Dict[str, str]:
        """Cancel the current queue"""
        success = runner.cancel_queue()
        if not success:
            raise HTTPException(status_code=404, detail="No active queue")
        return {"status": "cancelled", "message": "Queue cancelled"}

    @app.get("/api/queue/progress")
    async def get_queue_progress() -> Any:
        """SSE endpoint for real-time queue progress"""

        async def event_generator() -> Any:
            async for event in runner.subscribe_queue_progress():
                yield {"event": "queue_progress", "data": event.model_dump_json()}

        return EventSourceResponse(event_generator())

    @app.get("/api/run/{run_id}/progress")
    async def get_progress(run_id: str) -> Any:
        """SSE endpoint for real-time progress"""

        async def event_generator() -> Any:
            async for event in runner.subscribe_progress(run_id):
                yield {"event": "progress", "data": event.model_dump_json()}

        return EventSourceResponse(event_generator())

    @app.get("/api/run/{run_id}/logs")
    async def get_logs(run_id: str) -> Dict[str, Any]:
        """Get logs for a run"""
        logs = runner.get_logs(run_id)
        return {"run_id": run_id, "logs": logs}

    # =========================================================================
    # Export Endpoints
    # =========================================================================

    @app.get("/api/run/{run_id}/export/json")
    async def export_json(run_id: str) -> Any:
        """Export run results as JSON"""
        from fastapi.responses import Response

        json_file = outputs_path / f"{run_id}.json"
        if not json_file.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        with open(json_file) as f:
            data = json.load(f)

        # Add export metadata
        from ..export import generate_reproducibility_manifest

        export_data = {
            "results": data,
            "manifest": generate_reproducibility_manifest(
                config=data.get("config", {}),
                results=data.get("results", {}),
            ),
        }

        json_content = json.dumps(export_data, indent=2, default=str)

        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{run_id}_results.json"'},
        )

    @app.get("/api/run/{run_id}/export/latex")
    async def export_latex(run_id: str) -> Any:
        """Export run results as LaTeX table"""
        from fastapi.responses import Response

        json_file = outputs_path / f"{run_id}.json"
        if not json_file.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        with open(json_file) as f:
            data = json.load(f)

        from ..export import export_to_latex

        # Prepare results in expected format
        model_name = data.get("model", "Model")
        results_data = data.get("results", {})

        # Convert to format expected by export_to_latex
        formatted_results: dict[str, dict[str, Any]] = {model_name: {}}
        benchmarks_used = []

        for bench_name, bench_data in results_data.items():
            benchmarks_used.append(bench_name.lower())
            # Extract score
            score = None
            if "score" in bench_data:
                score = bench_data["score"]
            elif "accuracy" in bench_data:
                score = bench_data["accuracy"]
            elif f"{bench_name}_accuracy" in bench_data:
                score = bench_data[f"{bench_name}_accuracy"]
            elif "correct" in bench_data and "questions_tested" in bench_data:
                if bench_data["questions_tested"] > 0:
                    score = bench_data["correct"] / bench_data["questions_tested"]

            if score is not None:
                formatted_results[model_name][bench_name.lower()] = score
                # Add CI if available
                ci = bench_data.get("confidence_interval") or bench_data.get("ci")
                if ci:
                    formatted_results[model_name][f"{bench_name.lower()}_ci"] = tuple(ci)

        latex_content = export_to_latex(
            results=formatted_results,
            include_ci=True,
            caption=f"Benchmark Results for {model_name}",
            label=f"tab:{run_id}",
            benchmarks=benchmarks_used if benchmarks_used else None,
        )

        # Wrap in complete LaTeX document for easy compilation
        full_document = (
            r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage[colorlinks=true,allcolors=blue]{hyperref}

\begin{document}

"""
            + latex_content
            + r"""

\end{document}
"""
        )

        return Response(
            content=full_document,
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{run_id}_table.tex"'},
        )

    @app.get("/api/run/{run_id}/export/bibtex")
    async def export_bibtex(run_id: str) -> Any:
        """Export BibTeX citation and references"""
        from fastapi.responses import Response

        json_file = outputs_path / f"{run_id}.json"
        if not json_file.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        with open(json_file) as f:
            data = json.load(f)

        from ..export import generate_bibtex, generate_references_bibtex

        # Generate citation for this evaluation
        eval_metadata = {
            "version": "2.1.0",
            "date": data.get("started_at", "")[:10] if data.get("started_at") else "",
            "models_evaluated": [data.get("model", "unknown")],
            "n_samples": data.get("sample_size", "N/A"),
            "github_url": "https://github.com/NahuelGiudizi/llm-evaluation",
        }

        bibtex_content = f"""% Citation for this evaluation
{generate_bibtex(eval_metadata)}

% Standard benchmark references
{generate_references_bibtex()}
"""

        return Response(
            content=bibtex_content,
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{run_id}_references.bib"'},
        )

    @app.get("/api/run/{run_id}/export/csv")
    async def export_csv(run_id: str) -> Any:
        """Export run results as CSV"""
        import csv
        import io

        from fastapi.responses import Response

        json_file = outputs_path / f"{run_id}.json"
        if not json_file.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        with open(json_file) as f:
            data = json.load(f)

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "Benchmark",
                "Score (%)",
                "Correct",
                "Total",
                "CI Lower",
                "CI Upper",
                "Time (s)",
            ]
        )

        results_data = data.get("results", {})
        for bench_name, bench_data in results_data.items():
            # Extract score
            score = None
            if "score" in bench_data:
                score = bench_data["score"]
            elif "accuracy" in bench_data:
                score = bench_data["accuracy"]
            elif "correct" in bench_data and "questions_tested" in bench_data:
                if bench_data["questions_tested"] > 0:
                    score = bench_data["correct"] / bench_data["questions_tested"]

            ci = bench_data.get("confidence_interval") or bench_data.get("ci")
            ci_lower = ci[0] * 100 if ci else ""
            ci_upper = ci[1] * 100 if ci else ""

            writer.writerow(
                [
                    bench_name.upper(),
                    f"{score * 100:.2f}" if score else "",
                    bench_data.get("correct", ""),
                    bench_data.get("questions_tested") or bench_data.get("scenarios_tested", ""),
                    f"{ci_lower:.2f}" if ci_lower else "",
                    f"{ci_upper:.2f}" if ci_upper else "",
                    (
                        f"{bench_data.get('elapsed_time', ''):.2f}"
                        if bench_data.get("elapsed_time")
                        else ""
                    ),
                ]
            )

        csv_content = output.getvalue()

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{run_id}_results.csv"'},
        )

    # =========================================================================
    # Static Files (UI)
    # =========================================================================

    # Try to serve built UI if available
    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")
    else:

        @app.get("/")
        async def root() -> Any:
            return JSONResponse(
                {
                    "message": "LLM Benchmark Dashboard API",
                    "docs": "/docs",
                    "note": "UI not bundled. Run in development mode or build UI first.",
                }
            )

    return app


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 8888,
    outputs_dir: Optional[Path] = None,
    open_browser: bool = True,
    reload: bool = False,
) -> None:
    """
    Run the dashboard server.

    Args:
        host: Host to bind to (default: 127.0.0.1)
        port: Port to run on (default: 8888)
        outputs_dir: Directory to store outputs (default: ~/.llm-benchmark/outputs)
        open_browser: Whether to open browser automatically
        reload: Enable hot reload for development
    """
    missing = check_dependencies()
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("   Install with: pip install llm-benchmark-toolkit[dashboard]")
        sys.exit(1)

    import uvicorn

    # Create outputs dir
    outputs_path = outputs_dir or DEFAULT_OUTPUTS_DIR
    outputs_path.mkdir(parents=True, exist_ok=True)

    url = f"http://{host}:{port}"

    print(
        f"""
╔══════════════════════════════════════════════════════════════╗
║           ⚡ LLM Benchmark Dashboard v2.1.0 ⚡                ║
╠══════════════════════════════════════════════════════════════╣
║  Dashboard: {url:<47} ║
║  API Docs:  {url}/docs{' ' * 40} ║
║  Outputs:   {str(outputs_path)[:47]:<47} ║
╚══════════════════════════════════════════════════════════════╝
    """
    )

    # Open browser after short delay
    if open_browser:
        Timer(1.5, lambda: webbrowser.open(url)).start()

    # Create app with custom outputs dir
    os.environ["LLM_BENCHMARK_OUTPUTS"] = str(outputs_path)

    uvicorn.run(
        "llm_evaluator.dashboard.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
        log_level="info",
    )
