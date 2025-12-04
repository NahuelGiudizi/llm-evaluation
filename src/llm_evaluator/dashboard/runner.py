"""
Evaluation Runner

Manages background evaluation jobs with progress tracking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

from .models import ProgressEvent, QueueItemStatus, QueueStatus, RunStatus

if TYPE_CHECKING:
    from .models import QueueItem

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Manages evaluation runs with progress tracking"""

    def __init__(self, outputs_dir: Path) -> None:
        self.outputs_dir = outputs_dir
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Active runs tracking
        self._runs: dict[str, dict[str, Any]] = {}
        self._progress: dict[str, ProgressEvent] = {}
        self._subscribers: dict[str, list[asyncio.Queue[ProgressEvent]]] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.RLock()  # Use RLock to allow reentrant locking

        # Queue management
        self._queue_id: Optional[str] = None
        self._queue_items: list["QueueItem"] = []
        self._queue_status: list[QueueItemStatus] = []
        self._queue_current_index: int = 0
        self._queue_started_at: Optional[datetime] = None
        self._queue_cancelled: bool = False
        self._queue_subscribers: list[asyncio.Queue[QueueStatus]] = []

    def start_run(
        self,
        model: str,
        provider: str,
        benchmarks: list[str],
        preset: str,
        sample_size: int,
        inference_settings: Optional[dict[str, Any]] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        """Start a new evaluation run in the background"""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        output_file = self.outputs_dir / f"{run_id}.json"

        with self._lock:
            self._runs[run_id] = {
                "run_id": run_id,
                "model": model,
                "provider": provider,
                "benchmarks": benchmarks,
                "preset": preset,
                "sample_size": sample_size,
                "status": RunStatus.PENDING,
                "started_at": datetime.now(),
                "completed_at": None,
                "results": {},
                "artifacts": [],
                "logs": [],
                "inference_settings": inference_settings or {},
                "file_path": str(output_file.resolve()),
                "filename": output_file.name,
                "base_url": base_url,
                "system_info": self._get_system_info(),
            }

            self._progress[run_id] = ProgressEvent(
                run_id=run_id,
                status=RunStatus.PENDING,
                progress=0,
                questions_total=sample_size * len(benchmarks),
            )

            self._subscribers[run_id] = []

        # Start evaluation in background thread
        thread = threading.Thread(
            target=self._run_evaluation,
            args=(run_id, model, provider, benchmarks, sample_size, base_url, api_key),
            daemon=True,
        )
        thread.start()

        return run_id

    def _run_evaluation(
        self,
        run_id: str,
        model: str,
        provider: str,
        benchmarks: list[str],
        sample_size: int,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Run evaluation in background thread"""
        try:
            self._update_status(run_id, RunStatus.RUNNING)
            self._add_log(run_id, f"Starting evaluation: {model} on {', '.join(benchmarks)}")

            # Build CLI command
            output_file = self.outputs_dir / f"{run_id}.json"
            cmd = [
                sys.executable,
                "-m",
                "llm_evaluator.cli",
                "benchmark",
                "--model",
                model,
                "--provider",
                provider,
                "--benchmarks",
                ",".join(benchmarks),
                "--sample-size",
                str(sample_size),
                "--output",
                str(output_file),
            ]

            # Add base_url if provided
            if base_url:
                cmd.extend(["--base-url", base_url])

            # Add api_key if provided
            if api_key:
                cmd.extend(["--api-key", api_key])

            self._add_log(run_id, f"Command: {' '.join(cmd)}")

            # Run with UTF-8 encoding for Windows compatibility
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                encoding="utf-8",
                errors="replace",
            )

            with self._lock:
                self._processes[run_id] = process

            total_questions = sample_size * len(benchmarks)
            questions_done = 0
            current_benchmark = benchmarks[0] if benchmarks else None

            # Parse output for progress
            if process.stdout is not None:
                for line in iter(process.stdout.readline, ""):
                    if not line:
                        break

                    line = line.strip()
                    self._add_log(run_id, line)

                    # Parse progress from tqdm output
                    if "%" in line:
                        try:
                            pct_str = line.split("%")[0].split()[-1]
                            pct = float(pct_str)

                            # Detect which benchmark is running
                            bench_idx = 0
                            if "MMLU" in line:
                                current_benchmark = "mmlu"
                                bench_idx = benchmarks.index("mmlu") if "mmlu" in benchmarks else 0
                            elif "TruthfulQA" in line:
                                current_benchmark = "truthfulqa"
                                bench_idx = (
                                    benchmarks.index("truthfulqa")
                                    if "truthfulqa" in benchmarks
                                    else 0
                                )
                            elif "HellaSwag" in line:
                                current_benchmark = "hellaswag"
                                bench_idx = (
                                    benchmarks.index("hellaswag")
                                    if "hellaswag" in benchmarks
                                    else 0
                                )
                            elif "ARC" in line:
                                current_benchmark = "arc"
                                bench_idx = benchmarks.index("arc") if "arc" in benchmarks else 0
                            elif "WinoGrande" in line:
                                current_benchmark = "winogrande"
                                bench_idx = (
                                    benchmarks.index("winogrande")
                                    if "winogrande" in benchmarks
                                    else 0
                                )
                            elif "CommonsenseQA" in line:
                                current_benchmark = "commonsenseqa"
                                bench_idx = (
                                    benchmarks.index("commonsenseqa")
                                    if "commonsenseqa" in benchmarks
                                    else 0
                                )
                            elif "BoolQ" in line:
                                current_benchmark = "boolq"
                                bench_idx = (
                                    benchmarks.index("boolq") if "boolq" in benchmarks else 0
                                )
                            elif "SafetyBench" in line:
                                current_benchmark = "safetybench"
                                bench_idx = (
                                    benchmarks.index("safetybench")
                                    if "safetybench" in benchmarks
                                    else 0
                                )
                            elif "Do-Not-Answer" in line:
                                current_benchmark = "donotanswer"
                                bench_idx = (
                                    benchmarks.index("donotanswer")
                                    if "donotanswer" in benchmarks
                                    else 0
                                )

                            bench_progress = pct / 100
                            overall = (bench_idx + bench_progress) / len(benchmarks) * 100
                            questions_done = int(total_questions * overall / 100)

                            self._update_progress(
                                run_id,
                                progress=overall,
                                current_benchmark=current_benchmark,
                                questions_completed=questions_done,
                                questions_total=total_questions,
                            )
                        except (ValueError, IndexError):
                            pass

                    # Parse accuracy
                    if "Accuracy:" in line or "Score:" in line:
                        try:
                            score_str = line.split(":")[-1].strip().replace("%", "")
                            score = float(score_str) / 100
                            self._update_partial_metric(
                                run_id, current_benchmark or "unknown", score
                            )
                        except (ValueError, IndexError):
                            pass

            process.wait()

            if process.returncode == 0:
                if output_file.exists():
                    with open(output_file) as f:
                        results = json.load(f)
                    self._runs[run_id]["results"] = results
                    self._runs[run_id]["artifacts"].append(output_file.name)

                self._update_status(run_id, RunStatus.COMPLETED)
                self._update_progress(run_id, progress=100)
                self._add_log(run_id, "Evaluation completed successfully")

                # Save metadata AFTER updating status and adding final log
                self._save_run_metadata(run_id, output_file, "completed")
            else:
                self._update_status(run_id, RunStatus.FAILED)
                self._add_log(run_id, f"Evaluation failed with code {process.returncode}")
                # Save even failed runs
                self._save_run_metadata(run_id, output_file, "failed")

        except Exception as e:
            logger.exception(f"Evaluation failed: {e}")
            self._update_status(run_id, RunStatus.FAILED, error=str(e))
            self._add_log(run_id, f"Error: {e}")

        finally:
            self._runs[run_id]["completed_at"] = datetime.now()
            with self._lock:
                self._processes.pop(run_id, None)

    def cancel_run(self, run_id: str) -> bool:
        """Cancel an active evaluation run"""
        with self._lock:
            if run_id not in self._runs:
                return False

            run = self._runs[run_id]
            if run["status"] not in [RunStatus.PENDING, RunStatus.RUNNING]:
                return False

            process = self._processes.get(run_id)

        if process:
            try:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            except Exception as e:
                logger.warning(f"Error terminating process for {run_id}: {e}")

        self._update_status(run_id, RunStatus.CANCELLED)
        self._add_log(run_id, "Run cancelled by user")

        return True

    def _update_status(self, run_id: str, status: RunStatus, error: Optional[str] = None) -> None:
        """Update run status and notify subscribers"""
        with self._lock:
            self._runs[run_id]["status"] = status
            self._progress[run_id].status = status
            if error:
                self._progress[run_id].error = error

        self._notify_subscribers(run_id)

    def _update_progress(
        self,
        run_id: str,
        progress: float,
        current_benchmark: Optional[str] = None,
        questions_completed: int = 0,
        questions_total: int = 0,
    ) -> None:
        """Update progress and notify subscribers"""
        with self._lock:
            event = self._progress[run_id]
            event.progress = progress
            if current_benchmark:
                event.current_benchmark = current_benchmark
            event.questions_completed = questions_completed
            event.questions_total = questions_total

            if progress > 0:
                started = self._runs[run_id]["started_at"]
                elapsed = (datetime.now() - started).total_seconds()
                total_estimated = elapsed / (progress / 100)
                event.eta_seconds = int(total_estimated - elapsed)

        self._notify_subscribers(run_id)

    def _update_partial_metric(self, run_id: str, benchmark: str, score: float) -> None:
        """Update partial metrics"""
        with self._lock:
            self._progress[run_id].partial_metrics[benchmark] = score
        self._notify_subscribers(run_id)

    def _add_log(self, run_id: str, message: str) -> None:
        """Add log message"""
        with self._lock:
            timestamp = datetime.now().isoformat()
            self._runs[run_id]["logs"].append({"timestamp": timestamp, "message": message})

    def _notify_subscribers(self, run_id: str) -> None:
        """Notify all subscribers of progress update"""
        # Capture data under lock BEFORE scheduling async work
        with self._lock:
            if run_id not in self._subscribers:
                return
            event = self._progress.get(run_id)
            if not event:
                return
            # Copy the subscriber list to avoid issues
            subscribers = list(self._subscribers[run_id])

        async def _notify() -> None:
            for queue in subscribers:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(_notify(), loop)
        except RuntimeError:
            pass

    async def subscribe_progress(self, run_id: str) -> AsyncGenerator[ProgressEvent, None]:
        """Subscribe to progress updates for a run"""
        queue: asyncio.Queue[ProgressEvent] = asyncio.Queue(maxsize=100)

        with self._lock:
            if run_id not in self._subscribers:
                self._subscribers[run_id] = []
            self._subscribers[run_id].append(queue)

            if run_id in self._progress:
                yield self._progress[run_id]

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                    yield event

                    if event.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
                        break
                except asyncio.TimeoutError:
                    if run_id in self._progress:
                        yield self._progress[run_id]
        finally:
            with self._lock:
                if run_id in self._subscribers:
                    self._subscribers[run_id].remove(queue)

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        """Get run details"""
        with self._lock:
            return self._runs.get(run_id)

    def _save_run_metadata(
        self, run_id: str, output_file: Path, final_status: str = "completed"
    ) -> None:
        """Save complete run data with metadata to JSON file"""
        run = self._runs.get(run_id)
        if not run:
            return

        started_at = run.get("started_at")
        completed_at = run.get("completed_at") or datetime.now()

        # Get status value properly - handle RunStatus enum
        status: str = final_status
        run_status = run.get("status")
        if run_status is not None and hasattr(run_status, "value"):
            status = run_status.value
        elif isinstance(run_status, str):
            status = run_status

        # Calculate duration in seconds
        duration_seconds: float | None = None
        if started_at and completed_at:
            if isinstance(completed_at, datetime) and isinstance(started_at, datetime):
                duration_seconds = (completed_at - started_at).total_seconds()

        complete_data: dict[str, Any] = {
            "run_id": run_id,
            "model": run.get("model"),
            "provider": run.get("provider"),
            "benchmarks": run.get("benchmarks", []),
            "preset": run.get("preset"),
            "sample_size": run.get("sample_size"),
            "status": status,
            "started_at": started_at.isoformat() if started_at else None,
            "completed_at": (
                completed_at.isoformat() if isinstance(completed_at, datetime) else completed_at
            ),
            "duration_seconds": duration_seconds,
            "inference_settings": run.get("inference_settings", {}),
            "results": run.get("results", {}),
            "logs": run.get("logs", []),
            "system_info": self._get_system_info(),
        }

        with open(output_file, "w") as f:
            json.dump(complete_data, f, indent=2)

    def _get_system_info(self) -> dict[str, Any]:
        """Collect system information for reproducibility"""
        # Get processor name - try to get the friendly name on Windows
        processor_name = platform.processor()
        if platform.system() == "Windows":
            try:
                import winreg  # type: ignore[import-not-found]

                key = winreg.OpenKey(  # type: ignore[attr-defined]
                    winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"  # type: ignore[attr-defined]
                )
                processor_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()  # type: ignore[attr-defined]
                winreg.CloseKey(key)  # type: ignore[attr-defined]
            except Exception:
                pass  # Fall back to platform.processor()

        info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "processor": processor_name,
            "machine": platform.machine(),
        }

        try:
            import psutil

            info["cpu_count"] = psutil.cpu_count(logical=False)  # Physical cores
            info["cpu_count_logical"] = psutil.cpu_count(logical=True)  # Logical (threads)
            mem = psutil.virtual_memory()
            info["ram_total_gb"] = round(mem.total / (1024**3), 2)
            info["ram_available_gb"] = round(mem.available / (1024**3), 2)
        except ImportError:
            pass

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
                if gpu_info:
                    info["gpu"] = gpu_info
        except Exception:
            pass

        return info

    def get_runs(self) -> list[dict[str, Any]]:
        """Get all runs (from memory and disk)"""
        runs = []

        with self._lock:
            runs.extend(list(self._runs.values()))

        for json_file in self.outputs_dir.glob("run_*.json"):
            run_id = json_file.stem
            if not any(r["run_id"] == run_id for r in runs):
                try:
                    with open(json_file) as f:
                        data = json.load(f)

                    if "model" in data:
                        started_at = data.get("started_at")
                        completed_at = data.get("completed_at")
                        runs.append(
                            {
                                "run_id": data.get("run_id", run_id),
                                "model": data.get("model", "unknown"),
                                "provider": data.get("provider", "unknown"),
                                "benchmarks": data.get(
                                    "benchmarks", list(data.get("results", {}).keys())
                                ),
                                "preset": data.get("preset", "custom"),
                                "sample_size": data.get("sample_size", 0),
                                "status": RunStatus.COMPLETED,
                                "started_at": (
                                    datetime.fromisoformat(started_at)
                                    if started_at
                                    else datetime.now()
                                ),
                                "completed_at": (
                                    datetime.fromisoformat(completed_at)
                                    if completed_at
                                    else datetime.now()
                                ),
                                "results": data.get("results", {}),
                                "inference_settings": data.get("inference_settings", {}),
                                "system_info": data.get("system_info", {}),
                                "artifacts": [json_file.name],
                                "logs": [],
                                "file_path": str(json_file.resolve()),
                                "filename": json_file.name,
                            }
                        )
                    else:
                        benchmark_names = [
                            k for k in data.keys() if k not in ["timestamp", "system_info"]
                        ]
                        runs.append(
                            {
                                "run_id": run_id,
                                "model": "unknown (legacy)",
                                "provider": "unknown",
                                "benchmarks": benchmark_names,
                                "preset": "custom",
                                "sample_size": 0,
                                "status": RunStatus.COMPLETED,
                                "started_at": datetime.now(),
                                "completed_at": datetime.now(),
                                "results": data,
                                "system_info": data.get("system_info", {}),
                                "artifacts": [json_file.name],
                                "logs": [],
                                "file_path": str(json_file.resolve()),
                                "filename": json_file.name,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to load run {run_id}: {e}")

        runs.sort(key=lambda r: r.get("started_at", datetime.min), reverse=True)

        return runs

    def get_logs(self, run_id: str) -> list[dict[str, str]]:
        """Get logs for a run"""
        with self._lock:
            if run_id in self._runs:
                logs: list[dict[str, str]] = self._runs[run_id].get("logs", [])
                return logs
        return []

    def remove_run(self, run_id: str) -> bool:
        """Remove a run from memory cache"""
        with self._lock:
            if run_id in self._runs:
                del self._runs[run_id]
            if run_id in self._progress:
                del self._progress[run_id]
            if run_id in self._subscribers:
                del self._subscribers[run_id]
            return True

    # =========================================================================
    # Queue Management
    # =========================================================================

    def start_queue(self, runs: list["QueueItem"]) -> str:
        """Start a queue of sequential evaluations"""
        queue_id = f"queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        with self._lock:
            self._queue_id = queue_id
            self._queue_items = runs
            self._queue_current_index = 0
            self._queue_started_at = datetime.now()
            self._queue_cancelled = False

            # Initialize status for each item
            self._queue_status = [
                QueueItemStatus(
                    index=i,
                    model=item.model,
                    provider=item.provider,
                    benchmarks=item.benchmarks,
                    sample_size=item.sample_size,
                    status=RunStatus.PENDING,
                    inference_settings=(
                        item.inference_settings.model_dump() if item.inference_settings else None
                    ),
                )
                for i, item in enumerate(runs)
            ]

        # Start queue execution in background
        thread = threading.Thread(
            target=self._run_queue,
            daemon=True,
        )
        thread.start()

        return queue_id

    def _run_queue(self) -> None:
        """Execute queue items sequentially"""
        for i, item in enumerate(self._queue_items):
            if self._queue_cancelled:
                break

            with self._lock:
                self._queue_current_index = i
                self._queue_status[i].status = RunStatus.RUNNING

            self._notify_queue_subscribers()

            # Start the run and wait for completion
            start_time = datetime.now()
            run_id = self.start_run(
                model=item.model,
                provider=item.provider,
                benchmarks=item.benchmarks,
                preset="queue",
                sample_size=item.sample_size,
                inference_settings=(
                    item.inference_settings.model_dump() if item.inference_settings else None
                ),
                base_url=item.base_url,
                api_key=item.api_key,
            )

            with self._lock:
                self._queue_status[i].run_id = run_id

            # Wait for run to complete
            while True:
                if self._queue_cancelled:
                    self.cancel_run(run_id)
                    break

                with self._lock:
                    run = self._runs.get(run_id, {})
                    status = run.get("status", RunStatus.PENDING)

                if status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
                    break

                import time

                time.sleep(1)

            # Update queue item status
            duration = (datetime.now() - start_time).total_seconds()
            with self._lock:
                run = self._runs.get(run_id, {})
                status = run.get("status", RunStatus.FAILED)
                self._queue_status[i].status = status
                self._queue_status[i].duration_seconds = duration

                # Calculate score from results
                results = run.get("results", {})
                scores = []
                for bench_data in results.values():
                    if isinstance(bench_data, dict):
                        score = (
                            bench_data.get("mmlu_accuracy")
                            or bench_data.get("truthfulness_score")
                            or bench_data.get("hellaswag_accuracy")
                        )
                        if score is not None:
                            scores.append(score)
                if scores:
                    self._queue_status[i].score = sum(scores) / len(scores)

                if status == RunStatus.FAILED:
                    self._queue_status[i].error = run.get("error", "Unknown error")

            self._notify_queue_subscribers()

        # Mark queue as complete
        with self._lock:
            self._queue_id = None

    def get_queue_status(self) -> Optional[QueueStatus]:
        """Get current queue status"""
        # Fast path - check without lock first
        if not self._queue_id:
            return None

        with self._lock:
            if not self._queue_id:
                return None

            # Calculate overall status
            statuses = [item.status for item in self._queue_status]
            if all(s == RunStatus.COMPLETED for s in statuses):
                overall_status = RunStatus.COMPLETED
            elif any(s == RunStatus.RUNNING for s in statuses):
                overall_status = RunStatus.RUNNING
            elif any(s == RunStatus.FAILED for s in statuses):
                overall_status = RunStatus.FAILED
            elif self._queue_cancelled:
                overall_status = RunStatus.CANCELLED
            else:
                overall_status = RunStatus.PENDING

            # Calculate ETA
            eta_seconds: Optional[int] = None
            if self._queue_started_at and self._queue_current_index > 0:
                elapsed = (datetime.now() - self._queue_started_at).total_seconds()
                avg_per_item = elapsed / self._queue_current_index
                remaining = len(self._queue_items) - self._queue_current_index
                eta_seconds = int(avg_per_item * remaining)

            return QueueStatus(
                queue_id=self._queue_id,
                status=overall_status,
                items=self._queue_status.copy(),
                current_index=self._queue_current_index,
                total=len(self._queue_items),
                started_at=self._queue_started_at,
                eta_seconds=eta_seconds,
            )

    def cancel_queue(self) -> bool:
        """Cancel the current queue"""
        with self._lock:
            if not self._queue_id:
                return False
            self._queue_cancelled = True

        # Cancel current run if any
        current_item = self._queue_status[self._queue_current_index]
        if current_item.run_id:
            self.cancel_run(current_item.run_id)

        return True

    def _notify_queue_subscribers(self) -> None:
        """Notify all queue subscribers"""
        status = self.get_queue_status()
        if not status:
            return

        # Copy subscriber list under lock BEFORE scheduling async work
        with self._lock:
            subscribers = list(self._queue_subscribers)

        async def _notify() -> None:
            for queue in subscribers:
                try:
                    queue.put_nowait(status)
                except asyncio.QueueFull:
                    pass

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(_notify(), loop)
        except RuntimeError:
            pass

    async def subscribe_queue_progress(self) -> AsyncGenerator[QueueStatus, None]:
        """Subscribe to queue progress updates"""
        queue: asyncio.Queue[QueueStatus] = asyncio.Queue(maxsize=100)

        with self._lock:
            self._queue_subscribers.append(queue)

        # Send initial status (outside lock to avoid deadlock)
        status = self.get_queue_status()
        if status:
            yield status

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                    yield event

                    if event.status in (
                        RunStatus.COMPLETED,
                        RunStatus.FAILED,
                        RunStatus.CANCELLED,
                    ):
                        break
                except asyncio.TimeoutError:
                    status = self.get_queue_status()
                    if status:
                        yield status
                    else:
                        break
        finally:
            with self._lock:
                if queue in self._queue_subscribers:
                    self._queue_subscribers.remove(queue)
