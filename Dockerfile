# DT4LC Backend Dockerfile
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies for rasterio, OpenCV, and ML models
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

WORKDIR /app

# Copy all source code first (hatchling needs dta/__init__.py for version)
COPY pyproject.toml README.md ./
COPY dta/ ./dta/
COPY server/ ./server/

# Install dependencies
RUN uv pip install --system -e ".[server]"

# Create directories for uploads and cache
RUN mkdir -p /tmp/dt4lc_uploads /app/.cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Run the server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
