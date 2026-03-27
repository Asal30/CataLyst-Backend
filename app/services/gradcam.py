import os
import uuid
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CBM_CONCEPTS = ["NO", "NC", "CO", "PSC"]


# INTERNAL HELPERS

def _get_target_layer(model):
    # Always pick the last Conv2d layer from model tree.
    # This avoids selecting pooling/classifier blocks that produce flat CAMs.
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError(
            "Could not find any Conv2d layer in the model. "
            "Pass target_layer explicitly."
        )
    return last_conv


def _normalize_input_image(image_array: np.ndarray) -> np.ndarray:
    
    if not isinstance(image_array, np.ndarray):
        raise ValueError("image_array must be a numpy ndarray")

    arr = image_array.copy()

    # Drop the batch dimension if present
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(f"Expected HWC array, got shape {arr.shape}")

    # Convert float [0,1] → uint8 [0,255]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    return arr


def _build_model_transform() -> T.Compose:

    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def _cam_to_heatmap_and_overlay(pil_img: Image.Image, cam: np.ndarray):
    W, H = pil_img.size  # PIL gives (width, height)

    # 1. Resize CAM to original image resolution
    cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)

    # 2. Build a binary mask from CAM and localize the strongest blob.
    thr = 0.55
    bin_mask = (cam_resized >= thr).astype(np.uint8)  # 0/1

    # Smooth/close to form a single blob and reduce noise.
    bin_mask = cv2.GaussianBlur(bin_mask.astype(np.float32), (9, 9), 0)
    bin_mask = (bin_mask >= 0.20).astype(np.uint8)
    kernel = np.ones((9, 9), np.uint8)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

    # Fallback (typical lens center)
    cx0, cy0 = W // 2, H // 2
    r0 = int(min(W, H) * 0.12)

    # 3. Pick best component (prefer larger + closer to center)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    best_label = None
    best_score = None

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 250:
            continue
        cx, cy = centroids[label]
        dist = float(np.sqrt((cx - cx0) ** 2 + (cy - cy0) ** 2))
        center_bonus = 1.0 / (1.0 + dist / (0.4 * min(W, H)))
        score = float(area) * center_bonus
        if best_score is None or score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        circle = {"cx": int(cx0), "cy": int(cy0), "r": int(r0)}
    else:
        component_mask = (labels == best_label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            circle = {"cx": int(cx0), "cy": int(cy0), "r": int(r0)}
        else:
            cnt = max(contours, key=cv2.contourArea)
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            r = max(float(r), float(r0))
            circle = {"cx": int(round(cx)), "cy": int(round(cy)), "r": int(round(r))}

    # 4. Draw circle (red outline + mild fill). Do NOT tint whole image.
    original_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)

    # Prefer detecting the "white filled circle with a dark border" (your tip).
    detected_circle, detected_score, detect_method = _detect_white_circle_with_dark_ring(original_rgb)
    if detected_circle is not None and detected_score >= 12.0:
        circle = detected_circle
        circle_meta = {
            "source": "ring_detect",
            "method": detect_method,
            "ring_detect_score": round(float(detected_score), 3),
        }
    else:
        circle_meta = {
            "source": "gradcam_blob",
            "method": detect_method,
            "ring_detect_score": round(float(detected_score), 3),
        }

    overlay_rgb = original_rgb.copy()

    fill = overlay_rgb.copy()
    cv2.circle(fill, (circle["cx"], circle["cy"]), circle["r"], (255, 0, 0), thickness=-1)  # RGB red
    overlay_rgb = cv2.addWeighted(overlay_rgb, 0.78, fill, 0.22, 0)
    cv2.circle(overlay_rgb, (circle["cx"], circle["cy"]), circle["r"], (255, 0, 0), thickness=5)

    heatmap_rgb = overlay_rgb.copy()
    return heatmap_rgb, overlay_rgb, circle, circle_meta


def _build_center_prior(height: int, width: int) -> np.ndarray:
    """
    Build a smooth center-weighting prior in [0,1].
    This nudges Grad-CAM toward the lens center when activations are diffuse.
    """
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    dist_sq = xx * xx + yy * yy

    # Gaussian-like center emphasis; sigma controls spread.
    sigma = 0.55
    prior = np.exp(-dist_sq / (2.0 * sigma * sigma))
    prior = (prior - prior.min()) / (prior.max() - prior.min() + 1e-8)
    return prior.astype(np.float32)


def _score_ring_candidate(gray: np.ndarray, x: int, y: int, r: int, cx0: int, cy0: int) -> float:
    H, W = gray.shape[:2]
    if r <= 0:
        return 0.0
    if x - r < 0 or y - r < 0 or x + r >= W or y + r >= H:
        return 0.0

    inner_r = max(int(r * 0.70), 1)
    ring_r1 = max(int(r * 0.85), inner_r + 1)
    ring_r2 = max(int(r * 1.05), ring_r1 + 1)

    mask_inner = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask_inner, (x, y), inner_r, 255, thickness=-1)

    mask_ring = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask_ring, (x, y), ring_r2, 255, thickness=-1)
    cv2.circle(mask_ring, (x, y), ring_r1, 0, thickness=-1)

    inner_mean = float(cv2.mean(gray, mask=mask_inner)[0])
    ring_mean = float(cv2.mean(gray, mask=mask_ring)[0])

    contrast = inner_mean - ring_mean
    if contrast <= 0:
        return 0.0

    dist = float(np.hypot(x - cx0, y - cy0))
    center_bonus = 1.0 / (1.0 + dist / (0.35 * min(W, H)))
    return float(contrast * center_bonus)


def _detect_ring_via_contours(gray: np.ndarray) -> tuple[dict | None, float]:
    """
    Detect bright circular interior with a darker ring using contour + minEnclosingCircle.
    Works better than Hough on some slit-lamp images.
    """
    H, W = gray.shape[:2]
    cx0, cy0 = W // 2, H // 2

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -5
    )

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 800:
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        x, y, r = int(round(x)), int(round(y)), int(round(r))

        min_r = int(min(W, H) * 0.05)
        max_r = int(min(W, H) * 0.28)
        if r < min_r or r > max_r:
            continue

        peri = float(cv2.arcLength(cnt, True))
        if peri <= 1e-6:
            continue
        circularity = float(4.0 * np.pi * area / (peri * peri))
        if circularity < 0.55:
            continue

        score = _score_ring_candidate(gray, x, y, r, cx0, cy0)
        if score <= 0:
            continue
        score = score * (0.5 + 0.5 * circularity)

        if score > best_score:
            best_score = score
            best = {"cx": x, "cy": y, "r": r}

    return best, float(best_score)


def _detect_ring_via_hough(gray: np.ndarray) -> tuple[dict | None, float]:
    H, W = gray.shape[:2]
    cx0, cy0 = W // 2, H // 2

    min_r = int(min(W, H) * 0.06)
    max_r = int(min(W, H) * 0.22)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min(W, H) * 0.12),
        param1=90,
        param2=22,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return None, 0.0

    best = None
    best_score = 0.0

    for x, y, r in np.round(circles[0]).astype(int):
        score = _score_ring_candidate(gray, int(x), int(y), int(r), cx0, cy0)
        if score > best_score:
            best_score = score
            best = {"cx": int(x), "cy": int(y), "r": int(r)}

    return best, float(best_score)


def _detect_ring_via_edge_hough(gray: np.ndarray) -> tuple[dict | None, float]:
    """
    Detect circular border using Canny edges + HoughCircles.
    This often matches a black circular border better than raw intensity Hough.
    """
    H, W = gray.shape[:2]
    cx0, cy0 = W // 2, H // 2

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    min_r = int(min(W, H) * 0.06)
    max_r = int(min(W, H) * 0.28)

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min(W, H) * 0.12),
        param1=50,
        param2=18,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return None, 0.0

    best = None
    best_score = 0.0

    for x, y, r in np.round(circles[0]).astype(int):
        score = _score_ring_candidate(gray, int(x), int(y), int(r), cx0, cy0)
        if score > best_score:
            best_score = score
            best = {"cx": int(x), "cy": int(y), "r": int(r)}

    return best, float(best_score)


def _detect_white_circle_with_dark_ring(original_rgb: np.ndarray) -> tuple[dict | None, float, str]:
    """
    Two-stage ring detector:
    1) contour-based (often best for black-ring / white-fill targets)
    2) Hough fallback

    Returns: (circle_dict, score, method)
    """
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.5)

    c1, s1 = _detect_ring_via_contours(gray)
    if c1 is not None and s1 > 0:
        return c1, s1, "contour"

    c2, s2 = _detect_ring_via_hough(gray)
    if c2 is not None and s2 > 0:
        return c2, s2, "hough_gray"

    c3, s3 = _detect_ring_via_edge_hough(gray)
    if c3 is not None and s3 > 0:
        return c3, s3, "hough_edge"

    return None, 0.0, "none"


def _save_image(arr: np.ndarray, path: str) -> None:
    Image.fromarray(arr).save(path)


def _url_for_path(path: str) -> str:

    # Normalise separators and strip any leading dot/slash
    normalised = path.replace("\\", "/").lstrip("./")
    # Ensure exactly one leading slash
    return "/" + normalised


# PUBLIC API

def generate_cbm_concept_gradcams(image_array: np.ndarray, model: torch.nn.Module,
                                   target_layer: torch.nn.Module = None) -> dict:

    if model is None:
        raise ValueError("A model is required for Grad-CAM generation.")

    # 1. Prepare image
    arr     = _normalize_input_image(image_array)
    pil_img = Image.fromarray(arr).convert("RGB")
    tensor  = _build_model_transform()(pil_img).unsqueeze(0)  # (1,3,224,224)

    # 2. Identify target layer
    if target_layer is None:
        target_layer = _get_target_layer(model)

    # 3. Register hook
    saved = {"activations": None}

    def fwd_hook(module, inp, out):
        saved["activations"] = out  # keep live tensor and store grad on it
        out.retain_grad()

    fh = target_layer.register_forward_hook(fwd_hook)

    model.eval()
    run_id = uuid.uuid4().hex

    try:
        # Single forward to select dominant concept and capture activations
        out = model(tensor)
        concept_scores = out[0] if isinstance(out, (tuple, list)) else out
        if concept_scores.dim() != 2 or concept_scores.size(1) < len(CBM_CONCEPTS):
            raise RuntimeError(
                f"Unexpected model output shape {concept_scores.shape}. "
                f"Expected (1, {len(CBM_CONCEPTS)}) or larger."
            )

        scores_np = concept_scores[0].detach().cpu().numpy()
        concept_confidences = {
            CBM_CONCEPTS[i]: round(float(scores_np[i]), 4)
            for i in range(len(CBM_CONCEPTS))
        }

        dominant_concept_idx = int(np.argmax(scores_np))
        dominant_name = CBM_CONCEPTS[dominant_concept_idx]

        # Gradient for dominant concept with respect to captured activations.
        model.zero_grad()
        score = concept_scores[0, dominant_concept_idx]
        acts = saved["activations"]
        grads = None
        if acts is not None:
            grads = torch.autograd.grad(
                outputs=score,
                inputs=acts,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # CAM from dominant concept gradient signal
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = cam.squeeze().detach().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Increase contrast so highlighted regions are visible.
        cam = np.power(cam, 0.6)

        # Apply mild center prior: keeps Grad-CAM model-driven, but encourages
        # focus around the lens center for slit-lamp images.
        center_prior = _build_center_prior(cam.shape[0], cam.shape[1])
        cam = 0.75 * cam + 0.25 * center_prior
        cam = np.clip(cam, 0.0, 1.0)

        heatmap_rgb, overlay_rgb, circle, circle_meta = _cam_to_heatmap_and_overlay(pil_img, cam)

        # Save the OVERLAY as the primary Grad-CAM image, since it's easier to interpret.
        overlay_fname = f"gradcam_cbm_{run_id}.jpg"
        overlay_path = os.path.join(OUTPUT_DIR, overlay_fname)
        _save_image(overlay_rgb, overlay_path)
        overlay_url = _url_for_path(overlay_path)

        return {
            "gradcam_path":        overlay_url,
            "gradcam_paths":       {dominant_name: overlay_url},
            "heatmap_paths":       {dominant_name: overlay_url},
            "concept_confidences": concept_confidences,
            "dominant_concept":    dominant_name,
            "gradcam_run_id":      run_id,
            "gradcam_error":       None,
            "highlight_circle":    circle,
            "highlight_circle_meta": circle_meta,
        }

    finally:
        fh.remove()


def generate_gradcam_from_image_array(image_array: np.ndarray,
                                       model: torch.nn.Module,
                                       output_name: str = None) -> str | None:

    result = generate_cbm_concept_gradcams(image_array, model)
    if result.get("gradcam_path"):
        return result["gradcam_path"]

    # Fallback: just save the original image so the caller gets something back
    if output_name:
        arr  = _normalize_input_image(image_array)
        path = os.path.join(OUTPUT_DIR, output_name)
        _save_image(arr, path)
        return _url_for_path(path)

    return None