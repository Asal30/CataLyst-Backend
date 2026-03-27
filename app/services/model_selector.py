from app.models.cbm_loader import predict_cbm

def run_inference(image_array, source: str = "cbm"):
    source = source.lower().strip()

    if source == "cbm":
        return predict_cbm(image_array)
    else:
        raise ValueError(f"Invalid source: {source}. Only 'cbm' is supported.")