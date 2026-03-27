import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

torch.set_num_threads(1)
DEVICE = torch.device("cpu")

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
        self.presence_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)

        shared = self.shared(x)

        concepts = torch.sigmoid(self.concept_head(shared))
        presence = torch.sigmoid(self.presence_head(shared))

        return concepts, presence

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

cbm_model = ConceptBottleneckModel()

try:
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        elif isinstance(checkpoint, nn.Module):
            cbm_model  = checkpoint
            state_dict = None
        else:
            raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")

        if state_dict is not None:
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                key = k.replace("module.", "") if k.startswith("module.") else k
                cleaned_state_dict[key] = v

            cbm_model.load_state_dict(cleaned_state_dict, strict=False)

        cbm_model.to(DEVICE)
        cbm_model.eval()
        print("CBM model loaded successfully")
    else:
        print("CBM model file not found, using uninitialized model")
        cbm_model.to(DEVICE)
        cbm_model.eval()

except Exception as e:
    print(f"Failed to load CBM model: {e}")
    import traceback
    traceback.print_exc()
    print("Using uninitialized model for testing")
    cbm_model.to(DEVICE)
    cbm_model.eval()


# Preprocessing  (must match training transforms exactly)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# Internal helpers

def _severity_from_score(score_0_to_5: float) -> str:
    if score_0_to_5 < 1.0:
        return "Normal"
    if score_0_to_5 < 2.0:
        return "Mild"
    if score_0_to_5 < 3.5:
        return "Moderate"
    return "Severe"


def _build_gradcam_visual_explanations(concepts_scaled, concept_confidences):
    region_hints = {
        "NO":  "central nucleus region",
        "NC":  "nuclear color/intensity region in central lens",
        "CO":  "peripheral cortical spokes",
        "PSC": "posterior subcapsular region near the visual axis",
    }

    explanations    = []
    concept_names   = ["NO", "NC", "CO", "PSC"]

    for idx, name in enumerate(concept_names):
        raw_conf     = float(concept_confidences.get(name, 0.0))
        score_0_to_5 = float(concepts_scaled[idx] * 5.0)
        explanations.append({
            "concept":          name,
            "activation_region": region_hints[name],
            "confidence":       round(raw_conf, 4),
            "severity_score":   round(score_0_to_5, 2),
            "severity_pattern": _severity_from_score(score_0_to_5),
            "text": (
                f"{name} activates mostly in the {region_hints[name]} "
                f"(confidence={raw_conf:.3f}, severity={score_0_to_5:.2f}/5)."
            ),
        })

    return explanations


# Main prediction entry point

def predict_cbm(image_array: np.ndarray) -> dict:
    print("========== PREDICT_CBM CALLED ==========")
    print(f"Input shape: {image_array.shape}, dtype: {image_array.dtype}, "
          f"min/max: {image_array.min()}/{image_array.max()}")

    # Drop batch dimension if present  (1, H, W, C) → (H, W, C)
    if image_array.ndim == 4:
        print("Removing batch dimension...")
        image_array = image_array[0]

    print(f"Shape after squeeze: {image_array.shape}")

    # Convert float [0,1] → uint8 [0,255] for PIL
    if image_array.max() <= 1.0:
        image_uint8 = (image_array * 255).clip(0, 255).astype(np.uint8)
    else:
        image_uint8 = image_array.clip(0, 255).astype(np.uint8)

    print(f"uint8 min/max: {image_uint8.min()}/{image_uint8.max()}")

    img    = Image.fromarray(image_uint8).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    print(f"Tensor shape: {tensor.shape}")

    # Inference (no grad — fast path)
    print("Running CBM inference...")
    with torch.no_grad():
        concepts_output, presence_output = cbm_model(tensor)

    print(f"Concepts output: {concepts_output}")
    print(f"Presence output: {presence_output}")

    concepts_raw = concepts_output.detach().cpu().numpy().flatten()     # (4,)
    presence_raw = float(presence_output.detach().cpu().numpy().flatten()[0])

    concepts_clipped = np.clip(concepts_raw, 0.0, 1.0)

    print(f"Concepts [0,1]: {concepts_clipped}")
    print(f"Presence:       {presence_raw}")

    # Postprocess 
    from app.utils.postprocess import process_cbm_output
    result = process_cbm_output(concepts_clipped, presence_raw)

    # Grad-CAM (runs a second forward pass WITH gradients)
    try:
        from app.services.gradcam import generate_cbm_concept_gradcams

        gradcam_result = generate_cbm_concept_gradcams(image_uint8, cbm_model)

        result["gradcam_path"]        = gradcam_result.get("gradcam_path")
        result["gradcam_paths"]       = gradcam_result.get("gradcam_paths", {})

        # the raw jet-colourmap images (separate from the blended overlays).
        result["heatmap_paths"]       = gradcam_result.get("heatmap_paths", {})

        result["concept_confidences"] = gradcam_result.get("concept_confidences", {})
        result["gradcam_run_id"]      = gradcam_result.get("gradcam_run_id")
        result["gradcam_error"]       = gradcam_result.get("gradcam_error")
        result["highlight_circle"]      = gradcam_result.get("highlight_circle")
        result["highlight_circle_meta"] = gradcam_result.get("highlight_circle_meta")
        result["visual_explanations"] = _build_gradcam_visual_explanations(
            concepts_clipped,
            result["concept_confidences"],
        )

    except Exception as e:
        print(f"GradCAM generation failed: {e}")
        import traceback; traceback.print_exc()

        # Graceful fallback — inference result is still usable without heatmaps
        result["gradcam_path"]        = None
        result["gradcam_paths"]       = {}
        result["heatmap_paths"]       = {}   # FIXED: always present in response
        result["concept_confidences"] = {
            "NO":  round(float(concepts_clipped[0]), 4),
            "NC":  round(float(concepts_clipped[1]), 4),
            "CO":  round(float(concepts_clipped[2]), 4),
            "PSC": round(float(concepts_clipped[3]), 4),
        }
        result["gradcam_run_id"]      = None
        result["gradcam_error"]       = str(e)
        result["highlight_circle"]      = None
        result["highlight_circle_meta"] = None
        result["visual_explanations"] = _build_gradcam_visual_explanations(
            concepts_clipped,
            result["concept_confidences"],
        )

    print(f"Final result keys: {list(result.keys())}")
    print("========== PREDICT_CBM DONE ==========")

    return result