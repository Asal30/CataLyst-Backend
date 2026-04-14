from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from uuid import uuid4
import shutil
import os
import numpy as np
import time
from typing import Optional

from app.services.preprocess import preprocess_image
from app.services.gradcam import generate_cbm_concept_gradcams
from app.services.model_selector import run_inference, get_model
from app.utils.prediction_logger import log_prediction, get_prediction_logs

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _sanitize(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _build_predict_response(result: dict) -> dict:
    return {
        "prediction": {
            "label": result.get("prediction"),
            "is_cataract": result.get("is_cataract"),
            "cataract_probability": result.get("cataract_probability"),
            "cataract_probability_percent": result.get("cataract_probability_percent"),
            "decision_confidence_percent": result.get("decision_confidence_percent"),
            "presence_score": result.get("presence_score"),
            "presence_margin": result.get("presence_margin"),
            "presence_threshold": result.get("presence_threshold"),
        },
        "concepts": {
            "raw_scaled_0_to_1": result.get("raw_concepts_scaled"),
            "scores_0_to_5": {
                "NO": result.get("NO"),
                "NC": result.get("NC"),
                "CO": result.get("CO"),
                "PSC": result.get("PSC"),
            },
            "details": result.get("concepts"),
            "dominant_concept": result.get("dominant_concept"),
            "concept_confidences": result.get("concept_confidences"),
        },
        "interpretation": {
            "overall_severity_score": result.get("overall_severity_score"),
            "overall_severity_label": result.get("overall_severity_label"),
            "overall_score": result.get("overall_score"),
            "overall_severity": result.get("overall_severity"),
            "severity_method": result.get("severity_method"),
            "cataract_type": result.get("cataract_type"),
            "primary_cataract_type": result.get("primary_cataract_type"),
            "mixed_subtypes": result.get("mixed_subtypes"),
            "cataract_type_margin": result.get("cataract_type_margin"),
            "cataract_type_all_scores": result.get("cataract_type_all_scores"),
            "explanation": result.get("explanation"),
            "explanation_text": result.get("explanation_text"),
            "treatment_action": result.get("treatment", {}).get("action"),
            "treatment_recommendation": result.get("treatment", {}).get("recommendation"),
        },
        "visuals": {
            "gradcam_path": result.get("gradcam_path"),
            "raw_heatmap_path": result.get("raw_heatmap_path"),
            "center_prior_gradcam_path": result.get("center_prior_gradcam_path"),
            "heuristic_overlay_path": result.get("heuristic_overlay_path"),
            "gradcam_paths": result.get("gradcam_paths"),
            "heatmap_paths": result.get("heatmap_paths"),
            "heuristic_overlay_paths": result.get("heuristic_overlay_paths"),
            "gradcam_run_id": result.get("gradcam_run_id"),
            "gradcam_error": result.get("gradcam_error"),
            "highlight_circle": result.get("highlight_circle"),
            "highlight_circle_meta": result.get("highlight_circle_meta"),
            "visual_method": result.get("visual_method"),
            "visual_explanations": result.get("visual_explanations"),
            "visual_explanation_note": result.get("visual_explanation_note"),
        },
        "meta": {
            "interpretation_version": result.get("interpretation_version"),
            "model_name": result.get("model_name"),
            "model_version": result.get("model_version"),
            "preprocessing": result.get("preprocessing"),
            "original_image_url": result.get("original_image_url"),
            "medical_disclaimer": "This is not a medical diagnosis. Please consult an eye specialist.",
        },
    }


def _predict_from_image_path(image_path: str, source: str = "cbm") -> dict:
    image_array = preprocess_image(image_path, source)
    result = run_inference(image_array, source)
    return _sanitize(_build_predict_response(result))


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
        "image_id": file_name,
    }


@router.post("/predict")
def predict(image_id: str, source: str = "cbm"):
    try:
        image_path = os.path.join(UPLOAD_DIR, image_id)

        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")

        return _predict_from_image_path(image_path, source)

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR IN /predict: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain")
def explain(image_id: str, source: str = "cbm"):
    image_path = os.path.join(UPLOAD_DIR, image_id)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        image_array = preprocess_image(image_path, source)
        model = get_model(source)
        gradcam_result = generate_cbm_concept_gradcams(image_array, model)

        if gradcam_result.get("gradcam_error"):
            raise HTTPException(
                status_code=500,
                detail=f"Grad-CAM failed: {gradcam_result['gradcam_error']}",
            )

        return {
            "message": "Explanation generated successfully",
            "image_id": image_id,
            "visuals": {
                "gradcam_url": gradcam_result.get("gradcam_path"),
                "raw_heatmap_url": gradcam_result.get("raw_heatmap_path"),
                "center_prior_gradcam_url": gradcam_result.get("center_prior_gradcam_path"),
                "heuristic_overlay_url": gradcam_result.get("heuristic_overlay_path"),
                "gradcam_paths": gradcam_result.get("gradcam_paths"),
                "heatmap_paths": gradcam_result.get("heatmap_paths"),
                "heuristic_overlay_paths": gradcam_result.get("heuristic_overlay_paths"),
                "dominant_concept": gradcam_result.get("dominant_concept"),
                "concept_confidences": gradcam_result.get("concept_confidences"),
                "gradcam_run_id": gradcam_result.get("gradcam_run_id"),
                "highlight_circle": gradcam_result.get("highlight_circle"),
                "highlight_circle_meta": gradcam_result.get("highlight_circle_meta"),
                "visual_method": gradcam_result.get("visual_method"),
            },
            "explanation_text": (
                "The primary visual explanation is the raw Grad-CAM overlay. "
                "Any circle-based overlay is an auxiliary heuristic aid and not the direct neural explanation."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR IN /explain: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    source: str = Form("cbm"),
):
    try:
        start_time = time.time()

        file_ext = file.filename.split(".")[-1]
        image_id = f"{uuid4()}.{file_ext}"
        image_path = os.path.join(UPLOAD_DIR, image_id)

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = _predict_from_image_path(image_path, source)

        result["meta"]["image_id"] = image_id
        result["meta"]["image_source"] = source
        result["meta"]["inference_time_sec"] = round(time.time() - start_time, 3)
        result["meta"]["original_image_url"] = f"/uploads/{image_id}"

        return _sanitize(result)

    except Exception as e:
        print(f"ERROR IN /analyze: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/log-prediction")
def log_prediction_endpoint(
    image_id: str,
    source: str = "cbm",
    ground_truth_type: Optional[str] = None,
    ground_truth_severity: Optional[str] = None,
    notes: Optional[str] = None,
):
    try:
        image_path = os.path.join(UPLOAD_DIR, image_id)

        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")

        image_array = preprocess_image(image_path, source)
        prediction_result = run_inference(image_array, source)

        log_prediction(
            prediction_result=prediction_result,
            source=source,
            ground_truth_type=ground_truth_type,
            ground_truth_severity=ground_truth_severity,
            notes=notes,
        )

        return {
            "message": "Prediction logged successfully",
            "log_file": "logs/prediction_logs.csv",
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR IN /log-prediction: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prediction-logs")
def get_logs(limit: int = 50):
    try:
        logs = get_prediction_logs(limit=limit)
        return {
            "logs": logs,
            "total_returned": len(logs),
            "log_file": "logs/prediction_logs.csv",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))