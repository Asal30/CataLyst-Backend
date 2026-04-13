import os

# Set OpenBLAS memory limits BEFORE any torch/numpy imports
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["GOTO_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from fastapi.staticfiles import StaticFiles
import time

app = FastAPI(
    title="CataLyst API",
    description="Backend API for cataract screening using mobile images",
    version="1.0"
)

# CORS middleware — must come before other middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"\n{'='*60}")
    print(f"REQUEST: {request.method} {request.url}")
    print(f"{'='*60}\n")

    start_time = time.time()

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        print(f"Response status: {response.status_code} - Time: {process_time:.3f}s")
        return response
    except Exception as e:
        print(f"MIDDLEWARE ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Register routes
app.include_router(router)

# Serve output heatmap/overlay images  →  GET /outputs/<filename>
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Serve original uploaded images  →  GET /uploads/<filename>
# Required so the frontend GradCamOverlay can fetch the base image.
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
def root():
    return {"message": "CataLyst backend is running"}

# Touch: helps uvicorn --reload refresh modules reliably on Windows (v2).
