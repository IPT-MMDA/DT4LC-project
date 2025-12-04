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

# Install GDAL Python bindings matching system libgdal version
RUN pip install gdal==$(gdal-config --version)

# Install uv for fast package management
RUN pip install uv

WORKDIR /app

# --- Dependency caching layer ---
# Copy only files needed for dependency resolution
COPY pyproject.toml README.md ./
COPY uv.lock* ./

# Create minimal stub for hatchling version extraction (cached layer)
RUN mkdir -p dta server && \
    echo '__version__ = "0.0.0"' > dta/__init__.py && \
    touch server/__init__.py

# Install dependencies (this layer is cached unless pyproject.toml/uv.lock change)
RUN uv pip install --system -e ".[server]"

# --- Source code layer (changes frequently) ---
# Copy actual source code (invalidates only this layer on code changes)
COPY dta/ ./dta/
COPY server/ ./server/

# Cache directories are created at runtime under resources/.cache/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Run the server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
