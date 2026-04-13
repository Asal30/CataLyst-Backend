from PIL import Image
import numpy as np

def preprocess_image(image_path, source="cbm"):
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image, dtype=np.uint8)
    return image_array