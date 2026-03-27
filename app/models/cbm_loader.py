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
        presence = torch.sigmoid(self.presence_head(shared))  # ✅ FIXED

        return concepts, presence
    
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
            cbm_model = checkpoint
        else:
            raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")

        if state_dict is not None:
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    cleaned_state_dict[k.replace("module.", "")] = v
                else:
                    cleaned_state_dict[k] = v

            cbm_model.load_state_dict(cleaned_state_dict, strict=False)

        cbm_model.to(DEVICE)
        cbm_model.eval()
        print("CBM model loaded successfully")
    else:
        print("CBM model file not found, using uninitialized model")
        cbm_model.to(DEVICE)
        cbm_model.eval()

except Exception as e:
    print(f"✗ Failed to load CBM model: {e}")
    cbm_model.to(DEVICE)
    cbm_model.eval()

except Exception as e:
    print(f"Failed to load CBM model: {e}")
    import traceback
    traceback.print_exc()
    print("Using uninitialized model for testing")
    cbm_model.to(DEVICE)
    cbm_model.eval()


# -------------------------
# Preprocessing
# -------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def predict_cbm(image_array):
    print("========== PREDICT_CBM CALLED ==========")
    print(f"Input shape: {image_array.shape}")
    print(f"Input dtype: {image_array.dtype}")
    print(f"Input min/max: {image_array.min()}/{image_array.max()}")

    # Handle (1, H, W, C) input from preprocess_image
    if len(image_array.shape) == 4:
        print("Removing batch dimension...")
        image_array = image_array[0]

    print(f"Shape after squeeze: {image_array.shape}")
    
    # Convert 0-1 float back to uint8 for PIL
    print("Converting to uint8...")
    if image_array.max() <= 1.0:
        image_uint8 = (image_array * 255).astype(np.uint8)
    else:
        image_uint8 = image_array.astype(np.uint8)
    print(f"uint8 min/max: {image_uint8.min()}/{image_uint8.max()}")
    
    print("Creating PIL image...")
    img = Image.fromarray(image_uint8).convert("RGB")
    
    print("Applying transforms...")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    print(f"Tensor shape: {tensor.shape}")

    print("Running CBM inference...")
    with torch.no_grad():
        concepts_output, presence_output = cbm_model(tensor)
    
    print(f"Concepts output: {concepts_output}")
    print(f"Presence output: {presence_output}")
    
    # Unpack outputs
    concepts_raw = concepts_output.cpu().numpy().flatten()  # 4 concept scores
    presence_raw = presence_output.cpu().numpy().flatten()[0]  # Single presence score
    
    print(f"Raw concepts: {concepts_raw}")
    print(f"Raw presence: {presence_raw}")
    
    # Process concepts: clip to [0,1] and scale to [0,5]
    concepts_clipped = np.clip(concepts_raw, 0, 1)
    
    print(f"Concepts normalized [0,1]: {concepts_clipped}")
    print(f"Presence: {presence_raw}")
    
    # Process output through postprocessing
    from app.utils.postprocess import process_cbm_output
    result = process_cbm_output(concepts_clipped, presence_raw)
    
    # Optional Grad-CAM attachment
    try:
        from app.services.gradcam import generate_gradcam_from_image_array
        import uuid

        output_name = f"gradcam_cbm_{uuid.uuid4().hex}.jpg"
        gradcam_path = generate_gradcam_from_image_array(image_uint8, cbm_model, output_name=output_name)
        result["gradcam_path"] = gradcam_path
    except Exception as e:
        print(f"GradCAM generation failed for CBM model: {e}")
        result["gradcam_path"] = None

    print(f"Final result: {result}")
    print("========== PREDICT_CBM DONE ==========")
    
    return result
