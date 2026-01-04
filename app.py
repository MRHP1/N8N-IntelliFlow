#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Define tokenizer for model deserialization
def comma_tokenizer(text):
    return text.split(',')

app = Flask(__name__)

# ============================================================
# 1️⃣ Load models
# ============================================================
# Using v8 Robust Models (Assignee) & V9 (Deadline)
try:
    model_assignee = joblib.load("models/assignee_model_v8_robust.joblib")
    label_encoder = joblib.load("models/assignee_label_encoder_v8.joblib")
    
    # Load V9 Deadline Models
    try:
        deadline_clf = joblib.load("models/deadline_v9_classifier.joblib")
        deadline_le = joblib.load("models/deadline_v9_label_encoder.joblib")
        deadline_qr = joblib.load("models/deadline_v9_quantiles.joblib")
        print("✅ Models loaded successfully (Assignee v8 + Deadline V9).")
    except Exception as e:
        print(f"⚠️ V9 Deadline Models not found: {e}")
        deadline_clf, deadline_le, deadline_qr = None, None, None

except Exception as e:
    print(f"❌ Error loading models: {e}")
    model_assignee, label_encoder = None, None
    deadline_clf, deadline_le, deadline_qr = None, None, None


# ============================================================
# Helpers
# ============================================================

ASSIGNEE_MAP = {
    'maximk': 'Maxim Khutornenko', 'wfarner': 'Bill Farner',
    'kevints': 'Kevin Sweeney', 'mchucarroll': 'Mark Chu-Carroll',
    'zmanji': 'Zameer Manji', 'joshua.cohen': 'Joshua Cohen',
    'wickman': 'Brian Wickman', 'davmclau': 'David McLaughlin',
    'jsmith': 'Joe Smith', 'skarumuri': 'Suman Karumuri',
    'sshanmugham': 'Santhosh Kumar Shanmugham', 'mnurolahzade': 'Mehrdad Nurolahzade',
    'khuang': 'Kai Huang', 'dhamon': 'Dominic Hamon',
    'serb': 'Stephan Erb', 'bmahler': 'Benjamin Mahler'
}

# Priority Map for V5 Model
PRIORITY_MAP_V5 = {
    1: 'Blocker',
    2: 'Critical',
    3: 'Major',
    4: 'Minor',
    5: 'Trivial'
}

REQUIRED_FIELDS = {"key", "issuetype", "priorityid", "storypoint", "project", "summary"}

def ensure_models_loaded():
    if model_assignee is None or label_encoder is None:
        raise RuntimeError("Assignee models not loaded properly.")
    if deadline_clf is None:
        raise RuntimeError("V9 Deadline models not loaded properly. Run training script first.")


def unwrap_n8n(data):
    """
    Accepts ALL possible n8n formats:
    - { Json: {...} }
    - [ { Json: {...} } ]
    - raw {...}
    - [ {...} ]
    """
    if isinstance(data, list):
        cleaned = []
        for entry in data:
            if isinstance(entry, dict) and "Json" in entry:
                cleaned.append(entry["Json"])
            else:
                cleaned.append(entry)
        return cleaned

    if isinstance(data, dict) and "Json" in data:
        return [data["Json"]]

    if isinstance(data, dict):
        return [data]

    raise ValueError("Unrecognized input format from n8n")


def normalize_fields(item):
    """Normalize fields and handle defaults."""
    normalized = {}

    # normalize storyPoint → storypoint
    sp = item.get("storypoint") or item.get("storyPoint") or 0
    normalized["storypoint"] = float(sp)

    # Defaults
    normalized["key"] = item.get("key", "UNKNOWN-1")
    normalized["issuetype"] = item.get("issuetype", "Story")
    normalized["priorityid"] = float(item.get("priorityid", 3.0)) # Default Priority 3
    normalized["project"] = item.get("project", "Aurora") # Default Project
    normalized["summary"] = item.get("summary", "")
    normalized["description"] = item.get("description", "")
    normalized["components"] = item.get("components", "") # New field for V9
    normalized["status"] = item.get("status", "To Do")

    return normalized


# ============================================================
# 3️⃣ Predict Endpoint (v8 Compatible)
# ============================================================
@app.route("/predict/all", methods=["POST"])
def predict_all_api():
    try:
        ensure_models_loaded()
        raw = request.get_json(force=True)
        items = unwrap_n8n(raw)
        results = []

        for incoming in items:
            item = normalize_fields(incoming)
            
            story_points = item["storypoint"]
            priority = item["priorityid"]

            # ----------------------------------------
            # A. DEADLINE PREDICTION (V9 Robust Model)
            # ----------------------------------------
            deadline_info = {
                "class": None,
                "interval_min": None,
                "interval_avg": None,
                "interval_max": None
            }
            
            if deadline_clf:
                try:
                    # Prepare Input for V9
                    # V9 expects: ['story_points', 'issuetype', 'priority', 'project', 'text_feature', 'components']
                    text_feature = f"{item['summary']} {item['description']}".strip()
                    
                    df_v9 = pd.DataFrame([{
                        'story_points': story_points,
                        'issuetype': item['issuetype'],
                        'priority': item.get('priority', 'Major'), # Raw string priority if available? App default is float ID
                        'project': item['project'],
                        'text_feature': text_feature,
                        'components': item['components']
                    }])
                    
                    # NOTE: Notebook used raw strings for Priority (Blocker, etc.), but app defaults to ID 3.0.
                    # We need to map priority ID to string if it's a number, or use as is.
                    # Re-using PRIORITY_MAP_V5 logic from helper if needed, but updated for V9 if it uses same strings.
                    # Assuming V9 uses strings like 'Major', 'Critical'.
                    priority_val = item.get('priority')
                    if isinstance(priority_val, (int, float)) or (isinstance(priority_val, str) and priority_val.isdigit()):
                         df_v9['priority'] = PRIORITY_MAP_V5.get(int(float(priority_val)), 'Major')
                    elif not priority_val:
                         df_v9['priority'] = 'Major'
                    
                    # 1. Classification
                    pred_class_idx = deadline_clf.predict(df_v9)[0]
                    pred_class = deadline_le.inverse_transform([pred_class_idx])[0]
                    
                    # 2. Quantile Regression
                    lower = deadline_qr[0.05].predict(df_v9)[0]
                    median = deadline_qr[0.50].predict(df_v9)[0]
                    upper = deadline_qr[0.95].predict(df_v9)[0]
                    
                    deadline_info = {
                        "class": pred_class,
                        "interval_min": round(max(0.1, lower), 1),
                        "interval_avg": round(max(0.1, median), 1),
                        "interval_max": round(max(0.1, upper), 1),
                        "confidence": "90%"
                    }

                except Exception as e:
                    print(f"Error in V9 prediction: {e}")


            # ----------------------------------------
            # B. ASSIGNEE PREDICTION (v8 XGBoost NLP)
            # ----------------------------------------
            # Model expects: ['storypoint', 'priorityid', 'issuetype', 'project', 'text_content', 'status']
            
            # Feature Engineering
            text_content = f"{item['summary']} {item['description']}".strip()
            
            df_assignee = pd.DataFrame([{
                'storypoint': story_points,
                'priorityid': priority,
                'issuetype': item["issuetype"],
                'project': item["project"],
                'text_content': text_content,
                'status': item["status"]
            }])

            # Predict Probabilities for Top-K
            y_probs = model_assignee.predict_proba(df_assignee)[0]
            
            # Get Top 3 Indices
            top3_indices = np.argsort(y_probs)[-3:][::-1]
            top3_classes = label_encoder.inverse_transform(top3_indices)
            top3_probs = y_probs[top3_indices]

            # Construct Top-K List
            top_k_recs = []
            for name_code, prob in zip(top3_classes, top3_probs):
                # Map username -> Full Name
                full_name = ASSIGNEE_MAP.get(name_code, name_code)
                top_k_recs.append({
                    "assignee": full_name,
                    "username": name_code, # Keep orig for debugging
                    "probability": float(prob)
                })

            # Best Assignee
            recommended_assignee = top_k_recs[0]["assignee"]

            results.append({
                "key": item["key"],
                "project": item["project"],
                "issuetype": item["issuetype"],
                
                # N8N Expected Fields
                "predicted_deadline": deadline_info, # V9 Struct
                "predicted_deadline_days": deadline_info["interval_avg"], # Backward compat scalar

                "recommended_assignee": recommended_assignee,
                "top_k_recommendations": top_k_recs,
                
                "model_version": "v8_Assignee + V9_Deadline_Robust"
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
