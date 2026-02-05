from app.models.slitlamp_model import predict_slitlamp
from app.models.mobile_model import predict_mobile


def run_inference(image_array, source: str):
    source = source.lower().strip()

    print(f"Inference source selected: {source}")

    if source == "slitlamp":
        print("Using SLIT-LAMP model...")
        return predict_slitlamp(image_array)

    elif source == "mobile":
        print("Using MOBILE model...")
        return predict_mobile(image_array)

    else:
        raise ValueError(f"Unknown image source: {source}")