# LLM Benchmark Toolkit Docker Image
# Multi-stage build for minimal image size

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for caching)
COPY pyproject.toml setup.py README.md ./
COPY src/ ./src/

# Build wheel
RUN pip install --no-cache-dir build && \
    python -m build --wheel

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 benchuser

# Copy wheel from builder
COPY --from=builder /app/dist/*.whl /tmp/

# Install the package with all providers
RUN pip install --no-cache-dir /tmp/*.whl[dev] && \
    pip install --no-cache-dir openai anthropic huggingface-hub && \
    rm /tmp/*.whl

# Create directories for outputs
RUN mkdir -p /app/outputs /app/data && \
    chown -R benchuser:benchuser /app

# Switch to non-root user
USER benchuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/benchuser

# Expose dashboard port
EXPOSE 8888

# Default command shows help
ENTRYPOINT ["llm-eval"]
CMD ["--help"]

# ============================================
# Usage Examples:
# ============================================
# 
# Build the image:
#   docker build -t llm-benchmark .
#
# Quick evaluation with OpenAI:
#   docker run -e OPENAI_API_KEY=$OPENAI_API_KEY llm-benchmark quick
#
# Run specific benchmarks:
#   docker run -e OPENAI_API_KEY=$OPENAI_API_KEY llm-benchmark benchmark \
#     --model gpt-4o-mini --benchmarks mmlu,truthfulqa -s 50
#
# Launch dashboard:
#   docker run -p 8888:8888 -e OPENAI_API_KEY=$OPENAI_API_KEY \
#     llm-benchmark dashboard --host 0.0.0.0
#
# Compare models:
#   docker run -e OPENAI_API_KEY=$OPENAI_API_KEY llm-benchmark compare \
#     --models gpt-4o-mini,gpt-3.5-turbo --provider openai
#
# With Groq (ultra-fast):
#   docker run -e GROQ_API_KEY=$GROQ_API_KEY llm-benchmark quick \
#     --model llama-3.1-8b-instant --provider groq
#
# Mount volume for persistent results:
#   docker run -v $(pwd)/outputs:/app/outputs \
#     -e OPENAI_API_KEY=$OPENAI_API_KEY \
#     llm-benchmark run --model gpt-4o-mini -o /app/outputs/results.json
