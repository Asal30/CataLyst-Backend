from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from uuid import uuid4
import shutil
import os
import numpy as np
from typing import Optional
from app.services.preprocess import preprocess_image
from app.services.gradcam import generate_gradcam
from app.services.analyze import analyze_image
from app.services.model_selector import run_inference
from app.utils.prediction_logger import log_prediction, get_prediction_logs

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
def predict(image_id: str, source: str = "cbm"):
    try:
        image_path = f"uploads/{image_id}"

        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")

        image_array = preprocess_image(image_path, source)
        result = run_inference(image_array, source)

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
            "prediction": result.get("prediction"),
            "is_cataract": result.get("is_cataract"),
            "presence_score": result.get("presence_score"),
            "presence_confidence": result.get("presence_confidence"),
            "NO": result.get("NO"),
            "NC": result.get("NC"),
            "CO": result.get("CO"),
            "PSC": result.get("PSC"),
            "dominant_concept": result.get("dominant_concept"),
            "overall_score": result.get("overall_score"),
            "overall_severity": result.get("overall_severity"),
            "cataract_type": result.get("cataract_type"),
            "cataract_type_all_scores": result.get("cataract_type_all_scores"),
            "concepts": result.get("concepts"),
            "explanation": result.get("explanation"),
            "treatment_action": result.get("treatment", {}).get("action"),
            "treatment_recommendation": result.get("treatment", {}).get("recommendation"),
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
    source: str = Form("cbm")
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

@router.post("/log-prediction")
def log_prediction_endpoint(
    image_id: str,
    source: str = "cbm",
    ground_truth_type: Optional[str] = None,
    ground_truth_severity: Optional[str] = None,
    notes: Optional[str] = None
):
    """
    Log a prediction result for evaluation purposes.
    This does not interfere with the main prediction endpoint.
    """
    try:
        # Get the prediction result (assuming it was already run)
        image_path = f"uploads/{image_id}"

        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")

        # Run prediction to get result
        image_array = preprocess_image(image_path, source)
        prediction_result = run_inference(image_array, source)

        # Log the prediction
        log_prediction(
            prediction_result=prediction_result,
            source=source,
            ground_truth_type=ground_truth_type,
            ground_truth_severity=ground_truth_severity,
            notes=notes
        )

        return {
            "message": "Prediction logged successfully",
            "log_file": "logs/prediction_logs.csv"
        }

    except Exception as e:
        print(f"ERROR IN /log-prediction: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prediction-logs")
def get_logs(limit: int = 50):
    """Retrieve recent prediction logs for analysis."""
    try:
        logs = get_prediction_logs(limit=limit)
        return {
            "logs": logs,
            "total_returned": len(logs),
            "log_file": "logs/prediction_logs.csv"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))