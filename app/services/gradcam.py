from PIL import Image, ImageOps
import os
import numpy as np
import torch
import torchvision.transforms as T

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from PIL import Image, ImageOps
import os
import numpy as np
import torch
import torchvision.transforms as T

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_gradcam(image_path, image_id):
    from PIL import Image
    import numpy as np

    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    output_name = f"gradcam_{image_id}"

    base_image = Image.open(image_path).convert("RGBA")
    width, height = base_image.size

    overlay = Image.new("RGBA", base_image.size, (255, 0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(overlay)

    ellipse_box = [
        width * 0.25,
        height * 0.25,
        width * 0.75,
        height * 0.75
    ]
    draw.ellipse(ellipse_box, fill=(255, 0, 0, 60))

    result = Image.alpha_composite(base_image, overlay)

    output_path = os.path.join(OUTPUT_DIR, output_name)
    result.convert("RGB").save(output_path)

    return output_path


def _get_last_conv_layer(model):
    # ResNet path for last conv layer
    if hasattr(model, "layer4") and len(model.layer4) > 0:
        layer = model.layer4[-1]
        if hasattr(layer, "conv2"):
            return layer.conv2
    # Fallback: scan for last conv2d
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def generate_gradcam_from_image_array(image_array, model, output_name=None):
    """Run Grad-CAM on a model and an input image array.

    Args:
        image_array: numpy array shape (H,W,3) or (1,H,W,3), values in [0,1] or [0,255]
        model: torch model (eval mode recommended)
        output_name: optional filename (without path)

    Returns:
        absolute file path of saved gradcam overlay image
    """
    if model is None:
        raise ValueError("Model is required for gradcam generation")

    if isinstance(image_array, np.ndarray):
        arr = image_array.copy()
    else:
        raise ValueError("image_array must be numpy ndarray")

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.max() <= 1.0:
        arr = (arr * 255.0).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    pil_img = Image.fromarray(arr).convert("RGB")

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(pil_img).unsqueeze(0)

    target_layer = _get_last_conv_layer(model)
    if target_layer is None:
        raise RuntimeError("Unable to locate last conv layer for Grad-CAM")

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    model.eval()
    model.zero_grad()

    out = model(tensor)
    if isinstance(out, tuple) or isinstance(out, list):
        concepts = out[0]
    else:
        concepts = out

    if concepts is None or concepts.dim() != 2 or concepts.size(1) < 1:
        fh.remove(); bh.remove()
        raise RuntimeError("Unexpected model output shape for Grad-CAM")

    concept_scores = concepts[0]

    # Get dominant concept for Grad-CAM targeting using reference code logic
    from app.utils.postprocess import get_dominant_concept_index
    concepts_np = concept_scores.cpu().numpy()
    dominant_concept_idx = get_dominant_concept_index(concepts_np)

    score = concept_scores[dominant_concept_idx]

    # Backward on dominant concept only
    score.backward(retain_graph=False)

    if gradients is None or activations is None:
        fh.remove(); bh.remove()
        raise RuntimeError("Gradients or activations not captured")

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = cam.squeeze(0).squeeze(0).cpu().numpy()

    # Normalize CAM
    if cam.max() != cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    cam_image = Image.fromarray(np.uint8(cam * 255)).resize(pil_img.size, Image.BILINEAR)
    cam_image = cam_image.convert("RGBA")

    heatmap = Image.new("RGBA", pil_img.size)
    heatmap_data = np.array(cam_image)
    heatmap_data[..., 0] = 255
    heatmap_data[..., 1] = 0
    heatmap_data[..., 2] = 0
    heatmap_data[..., 3] = (heatmap_data[..., 3] * 0.6).astype(np.uint8)
    heatmap = Image.fromarray(heatmap_data)

    base = pil_img.convert("RGBA")
    overlay = Image.alpha_composite(base, heatmap)

    fname = output_name or f"gradcam_{np.random.randint(1_000_000)}.jpg"
    output_path = os.path.join(OUTPUT_DIR, fname)
    overlay.convert("RGB").save(output_path)

    fh.remove(); bh.remove()
    return output_path

