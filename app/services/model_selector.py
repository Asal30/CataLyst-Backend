from app.models.cbm_loader import predict_cbm, cbm_model

def run_inference(image_array, source: str = "cbm"):
    source = source.lower().strip()

    if source == "cbm":
        return predict_cbm(image_array)
    else:
        raise ValueError(f"Invalid source: {source}. Only 'cbm' is supported.")


def get_model(source: str = "cbm"):
    source = source.lower().strip()

    if source == "cbm":
        return cbm_model
    else:
        raise ValueError(f"Invalid source: {source}. Only 'cbm' is supported.")