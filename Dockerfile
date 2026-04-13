# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for PyTorch and image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install PyTorch CPU-only first (matching requirements.txt versions)
RUN pip install --no-cache-dir --timeout 120 \
    torch==2.10.0+cpu torchvision==0.25.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install all Python dependencies from requirements.txt
RUN pip install --no-cache-dir --timeout 120 -r requirements.txt

# Install production server (gunicorn already included in requirements if needed)
RUN pip install --no-cache-dir gunicorn

# Copy application code
COPY app/ ./app/

# Create directories for uploads, outputs, and logs
RUN mkdir -p uploads outputs logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with gunicorn and uvicorn workers
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120", "app.main:app"]
