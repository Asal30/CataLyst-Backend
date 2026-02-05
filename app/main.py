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

# CORS middleware - MUST come before other middleware
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
    print(f"Headers: {request.headers}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        print(f"Response status: {response.status_code} - Time: {process_time}s")
        return response
    except Exception as e:
        print(f"MIDDLEWARE CAUGHT ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Register routes
app.include_router(router)

# Mount static files for outputs
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.get("/")
def root():
    return {"message": "CataLyst backend is running"}