from PIL import Image
import numpy as np


def preprocess_image(image_path, source="mobile", target_size=(224, 224)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array