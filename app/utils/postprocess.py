import numpy as np


def get_severity_label(score):
    if score < 1.0:
        return "Normal"
    elif score < 2.0:
        return "Mild"
    elif score < 3.5:
        return "Moderate"
    elif score <= 5.0:
        return "Severe"
    else:
        return "Not in range of LOCS III"


def get_confidence(score):
    boundaries = np.array([1, 2, 3, 4])
    return float(np.min(np.abs(score - boundaries)))


def analyze_concepts(concepts_scaled):
    if isinstance(concepts_scaled, (list, tuple)):
        concepts_scaled = np.array(concepts_scaled)
    
    names = ["NO", "NC", "CO", "PSC"]
    results = {}
    
    for i, name in enumerate(names):
        score = float(concepts_scaled[i] * 5.0)
        results[name] = {
            "score": round(score, 2),
            "severity": get_severity_label(score),
            "confidence": round(get_confidence(score), 3)
        }
    
    return results


def detect_cataract_type(concepts_scaled):
    concepts_scaled = concepts_scaled * 5.0
    if isinstance(concepts_scaled, (list, tuple)):
        concepts_scaled = np.array(concepts_scaled)
    
    NO, NC, CO, PSC = concepts_scaled
    
    nuclear_score = (NO + NC) / 2.0
    
    type_scores = {
        "Nuclear": nuclear_score,
        "Cortical": CO,
        "PSC": PSC
    }
    
    detected_type = max(type_scores, key=type_scores.get)
    confidence = type_scores[detected_type] / 5.0
    
    return {
        "type": detected_type,
        "confidence": round(float(confidence), 3),
        "all_scores": {k: round(float(v), 2) for k, v in type_scores.items()}
    }


def compute_overall_severity(concepts_scaled):
    concepts_scaled = concepts_scaled * 5.0
    if isinstance(concepts_scaled, (list, tuple)):
        concepts_scaled = np.array(concepts_scaled)
    
    weights = np.array([0.35, 0.35, 0.15, 0.15])
    weighted_score = np.sum(concepts_scaled * weights)
    max_score = np.max(concepts_scaled)
    
    final_score = 0.6 * max_score + 0.4 * weighted_score
    
    return {
        "score": round(float(final_score), 2),
        "severity": get_severity_label(final_score)
    }


def generate_text_explanation(concepts_scaled, is_cataract):
    concepts_scaled = concepts_scaled * 5.0
    if isinstance(concepts_scaled, (list, tuple)):
        concepts_scaled = np.array(concepts_scaled)
    
    NO, NC, CO, PSC = concepts_scaled
    explanation = []
    type_info = detect_cataract_type(concepts_scaled)
    main_type = type_info["type"]
    
    if main_type == "Nuclear":
        explanation.append(
            f"High Nuclear Opalescence (NO={NO:.2f}) and/or Nuclear Color (NC={NC:.2f}) "
            f"indicate increased opacity in the central lens region, consistent with Nuclear Cataract."
        )
    
    if main_type == "Cortical":
        explanation.append(
            f"Elevated Cortical Opacity (CO={CO:.2f}) suggests peripheral spoke-like opacities, "
            f"which are characteristic of Cortical Cataract."
        )
    
    if main_type == "PSC":
        explanation.append(
            f"High Posterior Subcapsular score (PSC={PSC:.2f}) indicates opacity near the back of the lens, "
            f"which strongly affects central vision and is typical of PSC cataracts."
        )
    
    if len(explanation) == 0:
        explanation.append(
            "All cataract-related features are within low ranges, suggesting no significant cataract formation."
        )
    
    return explanation


def generate_treatment_suggestion(concepts_scaled):
    concepts_scaled = concepts_scaled * 5.0
    if isinstance(concepts_scaled, (list, tuple)):
        concepts_scaled = np.array(concepts_scaled)
    
    max_score = np.max(concepts_scaled)
    
    if max_score < 2.0:
        return {
            "action": "No immediate treatment required",
            "recommendation": "Regular monitoring is advised. Maintain eye health and routine check-ups."
        }
    elif max_score < 3.0:
        return {
            "action": "Non-surgical management",
            "recommendation": "Consider updating eyeglass prescription, improving lighting conditions, and periodic monitoring."
        }
    elif max_score < 4.0:
        return {
            "action": "Clinical evaluation recommended",
            "recommendation": "Consult an ophthalmologist. Cataract progression may begin affecting daily activities."
        }
    else:
        return {
            "action": "Surgical intervention likely required",
            "recommendation": "Cataract surgery should be considered, especially if vision impairment affects quality of life."
        }


def get_dominant_concept_index(concepts_scaled):
    concepts_scaled = concepts_scaled * 5.0
    if isinstance(concepts_scaled, (list, tuple)):
        concepts_scaled = np.array(concepts_scaled)
    
    NO, NC, CO, PSC = concepts_scaled
    
    nuclear_score = max(NO, NC)
    scores = [nuclear_score, CO, PSC]
    idx = int(np.argmax(scores))
    
    if idx == 0:
        return 0
    elif idx == 1:
        return 2
    else:
        return 3


def process_cbm_output(concepts_scaled, presence_score):
    if len(concepts_scaled) != 4:
        raise ValueError(f"Expected 4 concept scores, got {len(concepts_scaled)}")
    
    NO, NC, CO, PSC = concepts_scaled
    
    overall_severity = compute_overall_severity(concepts_scaled)
    severity_score = overall_severity["score"]
    is_cataract = (float(presence_score) > 0.5) or (severity_score >= 2.0)
    presence_confidence = round(abs(presence_score - 0.5) * 2, 3)
    
    concepts_analysis = analyze_concepts(concepts_scaled)
    overall_severity = compute_overall_severity(concepts_scaled)
    cataract_type = detect_cataract_type(concepts_scaled)
    explanation_list = generate_text_explanation(concepts_scaled, is_cataract)
    treatment = generate_treatment_suggestion(concepts_scaled)
    dominant_concept_idx = get_dominant_concept_index(concepts_scaled)
    
    concept_names = ["NO", "NC", "CO", "PSC"]
    dominant_concept_name = concept_names[dominant_concept_idx]
    
    prediction = "Cataract Detected" if is_cataract else "No Cataract Detected"
    
    return {
        "prediction": prediction,
        "is_cataract": bool(is_cataract),
        "presence_score": float(presence_score),
        "presence_confidence": presence_confidence,
        
        "concepts": concepts_analysis,
        "NO": round(float(NO * 5.0), 2),
        "NC": round(float(NC * 5.0), 2),
        "CO": round(float(CO * 5.0), 2),
        "PSC": round(float(PSC * 5.0), 2),
        
        "overall_score": overall_severity["score"],
        "overall_severity": overall_severity["severity"],
        "cataract_type": cataract_type["type"],
        "cataract_type_confidence": cataract_type["confidence"],
        "cataract_type_all_scores": cataract_type["all_scores"],
        "detected_type_score": cataract_type["all_scores"][cataract_type["type"]],
        
        "dominant_concept": dominant_concept_name,
        "dominant_concept_index": dominant_concept_idx,
        
        "explanation": explanation_list,
        "treatment": treatment,
        
        "explanation_text": " ".join(explanation_list),
        "treatment_suggestion": treatment["action"],
        "treatment_recommendation": treatment["recommendation"]
    }

