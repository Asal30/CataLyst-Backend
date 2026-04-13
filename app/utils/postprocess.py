import numpy as np

CONCEPT_NAMES = ["NO", "NC", "CO", "PSC"]

TYPE_MIX_THRESHOLD = 0.25
PRESENCE_THRESHOLD = 0.50
PRESENCE_BORDERLINE_MARGIN = 0.08


def _to_numpy_1d(concepts_scaled):
    arr = np.array(concepts_scaled, dtype=float).reshape(-1)
    if len(arr) != 4:
        raise ValueError(f"Expected 4 concept scores, got {len(arr)}")
    return np.clip(arr, 0.0, 1.0)


def get_severity_label(score):
    if score < 1.0:
        return "Normal"
    elif score < 2.0:
        return "Mild"
    elif score < 3.5:
        return "Moderate"
    elif score <= 5.0:
        return "Severe"
    return "Out of Range"


def get_boundary_distance(score):
    boundaries = np.array([1.0, 2.0, 3.5, 5.0], dtype=float)
    return float(np.min(np.abs(score - boundaries)))


def analyze_concepts(concepts_scaled):
    arr = _to_numpy_1d(concepts_scaled)
    scores_0_to_5 = arr * 5.0

    results = {}
    for i, name in enumerate(CONCEPT_NAMES):
        score = float(scores_0_to_5[i])
        results[name] = {
            "score": round(score, 2),
            "severity": get_severity_label(score),
            "boundary_distance": round(get_boundary_distance(score), 3),
        }
    return results


def compute_type_scores(concepts_scaled):
    arr = _to_numpy_1d(concepts_scaled)
    NO, NC, CO, PSC = arr * 5.0

    nuclear_score = (NO + NC) / 2.0
    return {
        "Nuclear": float(nuclear_score),
        "Cortical": float(CO),
        "PSC": float(PSC),
    }


def detect_cataract_type(concepts_scaled):
    type_scores = compute_type_scores(concepts_scaled)
    ranked = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)

    top_type, top_score = ranked[0]
    second_type, second_score = ranked[1]
    margin = float(top_score - second_score)

    if margin < TYPE_MIX_THRESHOLD:
        detected_type = "Mixed"
        mixed_subtypes = [top_type, second_type]
        detected_type_score = top_score
    else:
        detected_type = top_type
        mixed_subtypes = []
        detected_type_score = top_score

    return {
        "type": detected_type,
        "primary_type": top_type,
        "mixed_subtypes": mixed_subtypes,
        "type_margin": round(margin, 3),
        "all_scores": {k: round(float(v), 2) for k, v in type_scores.items()},
        "detected_type_score": round(float(detected_type_score), 2),
    }


def compute_overall_severity(concepts_scaled):
    arr = _to_numpy_1d(concepts_scaled)
    scores_0_to_5 = arr * 5.0

    weights = np.array([0.35, 0.35, 0.15, 0.15], dtype=float)
    weighted_score = float(np.sum(scores_0_to_5 * weights))
    max_score = float(np.max(scores_0_to_5))

    final_score = 0.6 * max_score + 0.4 * weighted_score

    return {
        "score": round(final_score, 2),
        "severity": get_severity_label(final_score),
        "method": "rule_based_weighted_max",
        "weights": {
            "NO": 0.35,
            "NC": 0.35,
            "CO": 0.15,
            "PSC": 0.15,
        },
    }


def get_dominant_concept(concepts_scaled):
    arr = _to_numpy_1d(concepts_scaled)
    idx = int(np.argmax(arr))
    return {
        "name": CONCEPT_NAMES[idx],
        "index": idx,
        "score": round(float(arr[idx] * 5.0), 2),
    }


def generate_text_explanation(concepts_scaled, presence_score, is_cataract):
    arr = _to_numpy_1d(concepts_scaled)
    NO, NC, CO, PSC = arr * 5.0

    explanation = []
    type_info = detect_cataract_type(arr)
    overall = compute_overall_severity(arr)

    presence_score = float(presence_score)
    presence_margin = abs(presence_score - PRESENCE_THRESHOLD)

    if presence_margin < PRESENCE_BORDERLINE_MARGIN:
        explanation.append(
            f"Presence prediction is borderline (presence score={presence_score:.3f}), "
            f"so this result should be interpreted cautiously."
        )

    if not is_cataract:
        explanation.append(
            "Concept-inspired scores remain in the low range, suggesting no strong cataract pattern in this image."
        )
        return explanation

    if type_info["type"] == "Mixed":
        sub_a, sub_b = type_info["mixed_subtypes"]
        explanation.append(
            f"Concept-inspired scores suggest a mixed cataract pattern, mainly {sub_a} and {sub_b}, "
            f"because subtype scores are close "
            f"(Nuclear={type_info['all_scores']['Nuclear']}, "
            f"Cortical={type_info['all_scores']['Cortical']}, "
            f"PSC={type_info['all_scores']['PSC']})."
        )
    elif type_info["primary_type"] == "Nuclear":
        explanation.append(
            f"Nuclear-related concept scores are most elevated (NO={NO:.2f}, NC={NC:.2f}), "
            f"suggesting a stronger nuclear-type pattern."
        )
    elif type_info["primary_type"] == "Cortical":
        explanation.append(
            f"Cortical opacity score is most elevated (CO={CO:.2f}), "
            f"suggesting a stronger cortical-type pattern."
        )
    elif type_info["primary_type"] == "PSC":
        explanation.append(
            f"Posterior subcapsular score is most elevated (PSC={PSC:.2f}), "
            f"suggesting a stronger PSC-type pattern."
        )

    explanation.append(
        f"Overall severity is estimated using rule-based aggregation of concept-inspired scores, "
        f"giving {overall['severity']} severity (score={overall['score']:.2f}/5)."
    )

    return explanation


def generate_treatment_suggestion(concepts_scaled):
    arr = _to_numpy_1d(concepts_scaled)
    max_score = float(np.max(arr * 5.0))

    if max_score < 2.0:
        return {
            "action": "No immediate treatment required",
            "recommendation": "Regular monitoring is advised. Maintain eye health and continue routine check-ups."
        }
    elif max_score < 3.0:
        return {
            "action": "Non-surgical management",
            "recommendation": "Consider updating eyeglass prescription, improving lighting conditions, and periodic monitoring."
        }
    elif max_score < 4.0:
        return {
            "action": "Clinical evaluation recommended",
            "recommendation": "Consult an ophthalmologist. Cataract progression may be affecting daily visual function."
        }
    else:
        return {
            "action": "Surgical evaluation may be required",
            "recommendation": "An ophthalmology review is recommended, especially if vision problems affect daily activities."
        }


def process_cbm_output(concepts_scaled, presence_score):
    arr = _to_numpy_1d(concepts_scaled)
    presence_score = float(np.clip(presence_score, 0.0, 1.0))

    concepts_analysis = analyze_concepts(arr)
    overall_severity = compute_overall_severity(arr)
    cataract_type = detect_cataract_type(arr)
    dominant_concept = get_dominant_concept(arr)

    severity_score = overall_severity["score"]

    # Main task = presence, but use severity as weak safety support
    is_cataract = (presence_score >= PRESENCE_THRESHOLD) or (severity_score >= 2.0)
    presence_margin = round(abs(presence_score - PRESENCE_THRESHOLD), 3)

    explanation_list = generate_text_explanation(arr, presence_score, is_cataract)
    treatment = generate_treatment_suggestion(arr)

    NO, NC, CO, PSC = arr * 5.0
    prediction = "Cataract Detected" if is_cataract else "No Cataract Detected"

    return {
        "prediction": prediction,
        "is_cataract": bool(is_cataract),

        "presence_score": round(presence_score, 6),
        "presence_margin": presence_margin,
        "presence_threshold": PRESENCE_THRESHOLD,

        "raw_concepts_scaled": {
            "NO": round(float(arr[0]), 4),
            "NC": round(float(arr[1]), 4),
            "CO": round(float(arr[2]), 4),
            "PSC": round(float(arr[3]), 4),
        },

        "NO": round(float(NO), 2),
        "NC": round(float(NC), 2),
        "CO": round(float(CO), 2),
        "PSC": round(float(PSC), 2),
        "concepts": concepts_analysis,

        "overall_score": overall_severity["score"],
        "overall_severity": overall_severity["severity"],
        "severity_method": overall_severity["method"],

        "cataract_type": cataract_type["type"],
        "primary_cataract_type": cataract_type["primary_type"],
        "mixed_subtypes": cataract_type["mixed_subtypes"],
        "cataract_type_margin": cataract_type["type_margin"],
        "cataract_type_all_scores": cataract_type["all_scores"],
        "detected_type_score": cataract_type["detected_type_score"],

        "dominant_concept": dominant_concept["name"],
        "dominant_concept_index": dominant_concept["index"],

        "explanation": explanation_list,
        "explanation_text": " ".join(explanation_list),
        "visual_explanation_note": (
            "The primary visual explanation is the raw Grad-CAM overlay. "
            "Any circle or ring overlay is an auxiliary heuristic aid and not the direct model explanation."
        ),

        "treatment": treatment,
        "treatment_suggestion": treatment["action"],
        "treatment_recommendation": treatment["recommendation"],

        "interpretation_version": "postprocess_v3_weak_concept_scope",
    }