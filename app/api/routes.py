from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from uuid import uuid4
import shutil
import os
import numpy as np
from app.services.preprocess import preprocess_image
from app.services.inference import predict_cataract
from app.services.gradcam import generate_gradcam
from app.services.analyze import analyze_image
from app.services.model_selector import run_inference

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/health")
def health_check():
    return {"status": "Backend is healthy"}

@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    file_ext = file.filename.split(".")[-1]
    file_name = f"{uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, file_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "Image uploaded successfully",
        "image_id": file_name
    }

@router.post("/predict")
def predict(image_id: str, source: str = "mobile"):
    try:
        image_path = f"uploads/{image_id}"

        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")

        # Use real preprocessing and inference
        image_array = preprocess_image(image_path, source)
        result = run_inference(image_array, source)

        # Sanitize numpy types for JSON serialization
        def sanitize(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(v) for v in obj]
            return obj

        response = {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "confidence_level": result["confidence_level"],
            "cataract_type": result.get("cataract_type"),
            "severity": result.get("severity"),
            "explanation": result["explanation"],
            "medical_disclaimer": "This is not a medical diagnosis. Please consult an eye specialist."
        }

        return sanitize(response)
        
    except Exception as e:
        print(f"ERROR IN /PREDICT: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/explain")
def explain(image_id: str):
    image_path = f"uploads/{image_id}"

    if not os.path.exists(image_path):
        return {"error": "Image not found"}

    gradcam_path = generate_gradcam(image_path, image_id)

    return {
        "message": "Explanation generated successfully",
        "gradcam_url": f"/outputs/gradcam_{image_id}",
        "explanation_text": (
            "Highlighted regions indicate areas that influenced the system's decision. "
            "Brighter regions suggest possible lens opacity."
        )
    }

@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    source: str = Form("mobile")
):
    file_ext = file.filename.split(".")[-1]
    image_id = f"{uuid4()}.{file_ext}"
    image_path = f"uploads/{image_id}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_image(image_path, image_id, source)
    result["image_id"] = image_id
    result["image_source"] = source

    return result