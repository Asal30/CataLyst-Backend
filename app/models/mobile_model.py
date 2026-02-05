import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

DEVICE = torch.device("cpu")

# -------------------------
# Model architecture
# -------------------------
class ConceptModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        # Assuming same architecture as slitlamp for now
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
MODEL_PATH = os.path.join(BASE_DIR, "models", "concept_mobile_final.hdf5")

mobile_model = ConceptModel().to(DEVICE)

try:
    if os.path.exists(MODEL_PATH):
        # Try loading state dict
        # Note: If saved as full model, this might fail, so we try specific logic
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        if isinstance(checkpoint, dict):
             # Handle state dict
             cleaned_state_dict = {}
             for k, v in checkpoint.items():
                if k.startswith("backbone."):
                    cleaned_state_dict[k.replace("backbone.", "")] = v
                else:
                    cleaned_state_dict[k] = v
             mobile_model.load_state_dict(cleaned_state_dict, strict=False)
        elif isinstance(checkpoint, nn.Module):
             mobile_model = checkpoint
        
        mobile_model.eval()
        print("Mobile PyTorch model loaded successfully.")
    else:
        print(f"Mobile model file not found at: {MODEL_PATH}")
        mobile_model = None

except Exception as e:
    print(f"Failed to load mobile model: {e}")
    mobile_model = None

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

def predict_mobile(image_array):
    """
    image_array: numpy array (H, W, 3) OR (1, H, W, 3) from preprocess_image
    returns: structured inference result
    """
    print("========== PREDICT_MOBILE CALLED ==========")
    print(f"Input shape: {image_array.shape}")
    print(f"Input dtype: {image_array.dtype}")
    print(f"Input min/max: {image_array.min()}/{image_array.max()}")
    
    if mobile_model is None:
        print("ERROR: Mobile model is None!")
        raise RuntimeError("Mobile model is not loaded")

    # Handle (1, H, W, C) input from preprocess_image
    if len(image_array.shape) == 4:
        print("Removing batch dimension...")
        image_array = image_array[0]

    print(f"Shape after squeeze: {image_array.shape}")
    
    # Convert 0-1 float back to uint8 for PIL
    print("Converting to uint8...")
    image_uint8 = (image_array * 255).astype(np.uint8)
    print(f"uint8 min/max: {image_uint8.min()}/{image_uint8.max()}")
    
    print("Creating PIL image...")
    img = Image.fromarray(image_uint8).convert("RGB")
    
    print("Applying transforms...")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    print(f"Tensor shape: {tensor.shape}")

    print("Running model inference...")
    with torch.no_grad():
        output = mobile_model(tensor)
    
    print(f"Model output: {output}")
    
    # Convert to standard list
    scores = output.cpu().numpy().reshape(-1)
    print(f"Scores: {scores}")

    from app.utils.postprocess import process_model_output
    print("Processing output...")
    result = process_model_output(scores)
    print(f"Final result: {result}")
    print("========== PREDICT_MOBILE DONE ==========")
    
    return result
