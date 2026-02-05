import random

def predict_cataract(preprocessed_image):
    """
    Dummy prediction logic.
    This will be replaced by the trained model later.
    """

    confidence = round(random.uniform(0.6, 0.95), 2)

    if confidence > 0.75:
        prediction = "Cataract Detected"
        explanation = "Cloudy patterns detected in the lens region"
    else:
        prediction = "No Cataract Detected"
        explanation = "Lens appears clear with no significant opacity"

    return prediction, confidence, explanation
