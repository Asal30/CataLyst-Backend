import os

# Set OpenBLAS memory allocation before importing torch
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["GOTO_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConceptBottleneckModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.features = nn.Sequential(*list(backbone.children())[:-1])

        self.shared = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.concept_head = nn.Linear(128, 4)
        self.presence_head = nn.Linear(128, 1)  # logits

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        shared = self.shared(x)

        concepts = torch.sigmoid(self.concept_head(shared))   # 0-1
        presence_logits = self.presence_head(shared)          # logits

        return concepts, presence_logits


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _load_model_checkpoint(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        key = k.replace("module.", "") if k.startswith("module.") else k
        cleaned_state_dict[key] = v

    model = ConceptBottleneckModel()
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    print("CBM checkpoint loaded")
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    model.to(DEVICE)
    model.eval()
    return model


cbm_model = _load_model_checkpoint(MODEL_PATH)


def _build_gradcam_visual_explanations(concepts_scaled, concept_confidences):
    region_hints = {
        "NO": "central nucleus region",
        "NC": "nuclear color/intensity region in central lens",
        "CO": "peripheral cortical spokes",
        "PSC": "posterior subcapsular region near the visual axis",
    }

    explanations = []
    concept_names = ["NO", "NC", "CO", "PSC"]

    for idx, name in enumerate(concept_names):
        raw_score = float(concept_confidences.get(name, 0.0))
        score_0_to_5 = float(concepts_scaled[idx] * 5.0)

        if score_0_to_5 < 1.0:
            severity_pattern = "Normal"
        elif score_0_to_5 < 2.0:
            severity_pattern = "Mild"
        elif score_0_to_5 < 3.5:
            severity_pattern = "Moderate"
        else:
            severity_pattern = "Severe"

        explanations.append({
            "concept": name,
            "activation_region": region_hints[name],
            "raw_model_score": round(raw_score, 4),
            "severity_score": round(score_0_to_5, 2),
            "severity_pattern": severity_pattern,
            "text": (
                f"For {name}, the primary Grad-CAM overlay highlights the {region_hints[name]} "
                f"(raw score={raw_score:.3f}, concept-inspired severity={score_0_to_5:.2f}/5). "
                f"Any separate circle overlay should be interpreted only as an auxiliary localization aid."
            ),
        })

    return explanations


def _to_pil(image_array: np.ndarray) -> Image.Image:
    arr = np.asarray(image_array)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    return Image.fromarray(arr).convert("RGB")


def predict_cbm(image_array: np.ndarray) -> dict:
    pil_img = _to_pil(image_array)
    tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        concepts_output, presence_logits = cbm_model(tensor)

    concepts_raw = concepts_output.detach().cpu().numpy().flatten()
    concepts_clipped = np.clip(concepts_raw, 0.0, 1.0)

    presence_logit = float(presence_logits.detach().cpu().numpy().flatten()[0])
    presence_prob = float(torch.sigmoid(presence_logits).detach().cpu().numpy().flatten()[0])

    result = {
        "raw_model_output": {
            "concepts_raw": concepts_raw.tolist(),
            "concepts_clipped": concepts_clipped.tolist(),
            "presence_logit": presence_logit,
            "presence_prob": presence_prob,
        }
    }

    from app.utils.postprocess import process_cbm_output
    processed_result = process_cbm_output(concepts_clipped, presence_prob)
    result.update(processed_result)

    result["model_name"] = "ConceptBottleneckModel_ResNet18"
    result["model_version"] = "cbm_backend_v3"
    result["preprocessing"] = {
        "resize": [224, 224],
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }

    try:
        from app.services.gradcam import generate_cbm_concept_gradcams

        gradcam_result = generate_cbm_concept_gradcams(np.array(pil_img), cbm_model)

        result["gradcam_path"] = gradcam_result.get("gradcam_path")
        result["raw_heatmap_path"] = gradcam_result.get("raw_heatmap_path")
        result["center_prior_gradcam_path"] = gradcam_result.get("center_prior_gradcam_path")
        result["heuristic_overlay_path"] = gradcam_result.get("heuristic_overlay_path")

        result["gradcam_paths"] = gradcam_result.get("gradcam_paths", {})
        result["heatmap_paths"] = gradcam_result.get("heatmap_paths", {})
        result["heuristic_overlay_paths"] = gradcam_result.get("heuristic_overlay_paths", {})

        result["concept_confidences"] = gradcam_result.get("concept_confidences", {
            "NO": round(float(concepts_clipped[0]), 4),
            "NC": round(float(concepts_clipped[1]), 4),
            "CO": round(float(concepts_clipped[2]), 4),
            "PSC": round(float(concepts_clipped[3]), 4),
        })

        result["dominant_concept"] = gradcam_result.get("dominant_concept", result.get("dominant_concept"))
        result["gradcam_run_id"] = gradcam_result.get("gradcam_run_id")
        result["gradcam_error"] = gradcam_result.get("gradcam_error")

        result["highlight_circle"] = gradcam_result.get("highlight_circle")
        result["highlight_circle_meta"] = gradcam_result.get("highlight_circle_meta")
        result["visual_method"] = gradcam_result.get("visual_method")
        result["visual_explanations"] = _build_gradcam_visual_explanations(
            concepts_clipped,
            result["concept_confidences"],
        )

    except Exception as e:
        print(f"GradCAM generation failed: {e}")

        result["gradcam_path"] = None
        result["raw_heatmap_path"] = None
        result["center_prior_gradcam_path"] = None
        result["heuristic_overlay_path"] = None

        result["gradcam_paths"] = {}
        result["heatmap_paths"] = {}
        result["heuristic_overlay_paths"] = {}

        result["concept_confidences"] = {
            "NO": round(float(concepts_clipped[0]), 4),
            "NC": round(float(concepts_clipped[1]), 4),
            "CO": round(float(concepts_clipped[2]), 4),
            "PSC": round(float(concepts_clipped[3]), 4),
        }

        result["gradcam_run_id"] = None
        result["gradcam_error"] = str(e)
        result["highlight_circle"] = None
        result["highlight_circle_meta"] = None
        result["visual_method"] = None
        result["visual_explanations"] = _build_gradcam_visual_explanations(
            concepts_clipped,
            result["concept_confidences"],
        )

    return result