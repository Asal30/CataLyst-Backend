import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import torch
torch.set_num_threads(1)

DEVICE = torch.device("cpu")

# -------------------------
# Model architecture
# -------------------------
class ConceptModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.concept_head = nn.Linear(512, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.concept_head(x)

# -------------------------
# Load model
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "concept_slitlamp.pt")
slitlamp_model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
slitlamp_model.eval()

try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            cleaned_state_dict[k.replace("backbone.", "")] = v
        else:
            cleaned_state_dict[k] = v

    slitlamp_model.load_state_dict(cleaned_state_dict, strict=False)
    slitlamp_model.eval()
    print("Slit-lamp PyTorch model loaded successfully.")

except Exception as e:
    print(f"Failed to load slit-lamp model: {e}")
    slitlamp_model = None

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

# -------------------------
# Prediction function
# -------------------------
def predict_slitlamp(image_array):

    if slitlamp_model is None:
        raise RuntimeError("Slit-lamp model is not loaded")

    # Handle (1, H, W, 3) input from preprocess_image
    if len(image_array.shape) == 4:
        image_array = image_array[0]
    
    # Convert 0-1 float back to uint8 for PIL
    if image_array.max() <= 1.0:
        image_uint8 = (image_array * 255).astype(np.uint8)
    else:
        image_uint8 = image_array.astype(np.uint8)
    img = Image.fromarray(image_uint8).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = slitlamp_model(tensor)

    scores = output.cpu().numpy().reshape(-1)

    from app.utils.postprocess import process_model_output
    return process_model_output(scores)
