import numpy as np

def process_model_output(output_tensor):
    
    scores = np.array(output_tensor).flatten()
    
    if len(scores) != 4:
        raise ValueError(f"Expected 4 output scores, got {len(scores)}: {scores}")

    no_score = scores[0]
    nc_score = scores[1]
    co_score = scores[2]
    psc_score = scores[3]

    cataract_map = {
        "Nuclear Cataract": nc_score,
        "Cortical Cataract": co_score,
        "Posterior Subcapsular Cataract": psc_score
    }

    # Find the strongest cataract signal
    best_type = max(cataract_map, key=cataract_map.get)
    best_score = cataract_map[best_type]

    is_cataract = best_score > no_score

    if best_score < 1.5:
        severity = "Mild"
    elif 1.5 <= best_score <= 2.5:
        severity = "Moderate"
    else:
        severity = "Severe"
    
    relevant_score = best_score if is_cataract else no_score
    
    MAX_SCORE = 4.0  # based on training distribution
    confidence_val = (relevant_score / MAX_SCORE) * 100
    confidence_val = max(0.0, min(confidence_val, 100.0))
    confidence_percent = round(confidence_val, 1)

    if confidence_percent >= 85:
        conf_level = "High confidence"
    elif confidence_percent >= 70:
        conf_level = "Moderate confidence"
    else:
        conf_level = "Low confidence"

    # --- 4. RESULT CONSTRUCTION ---
    if is_cataract:
        prediction = "Cataract Detected"
        explanation = (
            f"The model detected signs of {best_type} ({severity}). "
            f"The analysis indicates {conf_level.lower()} in this result based on feature strength."
        )
        final_type = best_type
        final_severity = severity
    else:
        prediction = "No Cataract Detected"
        explanation = (
            "The lens appears clear. The model found stronger evidence for a normal eye "
            "than for any cataract type."
        )
        final_type = None
        final_severity = None

    return {
        "prediction": prediction,
        "cataract_type": final_type,
        "severity": final_severity,
        "confidence": confidence_percent,
        "confidence_level": conf_level,
        "explanation": explanation,
        "raw_scores": {
            "NO": float(no_score),
            "NC": float(nc_score),
            "CO": float(co_score),
            "PSC": float(psc_score)
        }
    }
