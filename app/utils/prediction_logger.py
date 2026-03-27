import csv
import os
from datetime import datetime
from typing import Optional, Dict, Any

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "prediction_logs.csv")

def initialize_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'timestamp',
                'source',
                'prediction',
                'is_cataract',
                'presence_score',
                'presence_confidence',
                'detected_cataract_type',
                'detected_type_score',
                'concept_NO',
                'concept_NC',
                'concept_CO',
                'concept_PSC',
                'ground_truth_type',
                'ground_truth_severity',
                'notes'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

def log_prediction(
    prediction_result: Dict[str, Any],
    source: str = "unknown",
    ground_truth_type: Optional[str] = None,
    ground_truth_severity: Optional[str] = None,
    notes: Optional[str] = None
):
    initialize_log_file()

    raw_scores = prediction_result.get('raw_scores', {})
    concept_no = raw_scores.get('NO', None)
    concept_nc = raw_scores.get('NC', None)
    concept_co = raw_scores.get('CO', None)
    concept_psc = raw_scores.get('PSC', None)

    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'source': source,
        'prediction': prediction_result.get('prediction'),
        'is_cataract': prediction_result.get('is_cataract'),
        'presence_score': prediction_result.get('presence_score'),
        'presence_confidence': prediction_result.get('presence_confidence'),
        'detected_cataract_type': prediction_result.get('detected_cataract_type'),
        'detected_type_score': prediction_result.get('detected_type_score'),
        'concept_NO': concept_no,
        'concept_NC': concept_nc,
        'concept_CO': concept_co,
        'concept_PSC': concept_psc,
        'ground_truth_type': ground_truth_type,
        'ground_truth_severity': ground_truth_severity,
        'notes': notes
    }

    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'timestamp', 'source', 'prediction', 'is_cataract', 'presence_score',
            'presence_confidence', 'detected_cataract_type', 'detected_type_score',
            'concept_NO', 'concept_NC', 'concept_CO', 'concept_PSC',
            'ground_truth_type', 'ground_truth_severity', 'notes'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(log_entry)

    print(f"Prediction logged to {LOG_FILE}")

def get_prediction_logs(limit: int = 100) -> list:
    if not os.path.exists(LOG_FILE):
        return []

    logs = []
    with open(LOG_FILE, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            logs.append(row)
            if len(logs) >= limit:
                break

    return logs[::-1]