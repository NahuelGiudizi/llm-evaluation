"""
Pydantic models for the Dashboard API
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class RunStatus(str, Enum):
    """Status of an evaluation run"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Preset(BaseModel):
    """Evaluation preset configuration"""

    id: str
    name: str
    description: str
    sample_size: Optional[int] = None
    benchmarks: list[str] = []
    category: str = "General"


class Benchmark(BaseModel):
    """Benchmark configuration"""

    id: str
    name: str
    description: str
    category: str = "General"
    questions_count: int = 0


class ModelInfo(BaseModel):
    """Information about an available model"""

    id: str
    name: str
    provider: str
    size: Optional[str] = None
    modified: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response for /models endpoint"""

    ollama: list[ModelInfo] = []
    openai: list[ModelInfo] = []
    anthropic: list[ModelInfo] = []
    deepseek: list[ModelInfo] = []
    huggingface: list[ModelInfo] = []


class InferenceSettings(BaseModel):
    """Inference hyperparameters"""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    max_tokens: int = 2048
    seed: Optional[int] = 42


class RunRequest(BaseModel):
    """Request to start a new evaluation run"""

    model: str
    provider: str = "ollama"
    benchmarks: list[str] = ["mmlu", "truthfulqa", "hellaswag"]
    preset: str = "quick"
    sample_size: Optional[int] = None
    inference_settings: Optional[InferenceSettings] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class RunResponse(BaseModel):
    """Response when starting a run"""

    run_id: str
    status: RunStatus
    message: str


class ProgressEvent(BaseModel):
    """Progress event for SSE"""

    run_id: str
    status: RunStatus
    progress: float = 0
    current_benchmark: Optional[str] = None
    current_step: Optional[str] = None
    questions_completed: int = 0
    questions_total: int = 0
    partial_metrics: dict[str, float] = {}
    eta_seconds: Optional[int] = None
    error: Optional[str] = None


class RunSummary(BaseModel):
    """Summary for run history list"""

    run_id: str
    model: str
    provider: str
    status: RunStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    benchmarks: list[str]
    preset: Optional[str] = None
    sample_size: Optional[int] = None
    overall_score: Optional[float] = None
    results: dict[str, Any] = {}
    system_info: dict[str, Any] = {}
    inference_settings: dict[str, Any] = {}
    file_path: Optional[str] = None
    filename: Optional[str] = None


class RunsResponse(BaseModel):
    """Response for /runs endpoint"""

    runs: list[RunSummary]
    total: int


class DeleteRunsRequest(BaseModel):
    """Request to delete multiple runs"""

    run_ids: list[str]


class DeleteRunsResponse(BaseModel):
    """Response after deleting runs"""

    deleted: list[str]
    failed: list[str]
    message: str


class QueueItem(BaseModel):
    """Single item in the evaluation queue"""

    model: str
    provider: str = "ollama"
    benchmarks: list[str] = ["mmlu", "truthfulqa", "hellaswag"]
    sample_size: int = 100
    inference_settings: Optional[InferenceSettings] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class QueueRequest(BaseModel):
    """Request to start a queue of evaluations"""

    runs: list[QueueItem]


class QueueItemStatus(BaseModel):
    """Status of a single queue item"""

    index: int
    model: str
    provider: str
    benchmarks: list[str]
    sample_size: int
    status: RunStatus
    run_id: Optional[str] = None
    score: Optional[float] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


class QueueStatus(BaseModel):
    """Status of the entire queue"""

    queue_id: str
    status: RunStatus
    items: list[QueueItemStatus]
    current_index: int
    total: int
    started_at: Optional[datetime] = None
    eta_seconds: Optional[int] = None
