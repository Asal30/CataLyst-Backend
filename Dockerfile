# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for PyTorch and image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install PyTorch CPU-only first (much smaller ~200MB vs 915MB CUDA wheel)
RUN pip install --no-cache-dir --timeout 120 \
    torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies (skip torch packages already installed)
RUN pip install --no-cache-dir --timeout 120 \
    "numpy<2.0" \
    fastapi==0.115.0 \
    python-multipart==0.0.9 \
    aiofiles==23.2.1 \
    pillow \
    scikit-learn \
    scipy \
    pandas \
    matplotlib

# Install production server
RUN pip install --no-cache-dir gunicorn uvicorn[standard]

# Copy application code
COPY app/ ./app/

# Create directories for uploads and outputs
RUN mkdir -p uploads outputs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with gunicorn and uvicorn workers
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120", "app.main:app"]
