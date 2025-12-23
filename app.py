#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# ============================================================
# 1️⃣ Load models
# ============================================================
# Using V7 Robust Models
try:
    model_assignee = joblib.load("models/assignee_model_v7_robust.joblib")
    label_encoder = joblib.load("models/assignee_label_encoder_v7.joblib")
    model_deadline = joblib.load("models/deadline_model_v7_simple.joblib")
    print("✅ Models loaded successfully (V7).")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model_assignee, model_deadline, label_encoder = None, None, None


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

REQUIRED_FIELDS = {"key", "issuetype", "priorityid", "storypoint", "project", "summary"}

def ensure_models_loaded():
    if model_deadline is None or model_assignee is None or label_encoder is None:
        raise RuntimeError("Models not loaded properly. Check models/ directory.")


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
    normalized["status"] = item.get("status", "To Do")

    return normalized


# ============================================================
# 3️⃣ Predict Endpoint (V7 Compatible)
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

            # ----------------------------------------
            # A. DEADLINE PREDICTION (V7 Random Forest)
            # ----------------------------------------
            # Model expects: ['storypoint', 'key', 'points_per_issue', 'complexity']
            # We treat a single issue as a "Sprint of Size 1"
            
            story_points = item["storypoint"]
            priority = item["priorityid"]
            
            # Feature Engineering for Single Issue
            complexity = story_points * (4 - priority) # Higher priority (1) -> Higher complexity multiplier? 
            # In V7 training: complexity = storypoint * (4 - priorityid)
            # Make sure priorityid is 1-3. If 1 is high, 4-1=3 (High weight). Correct.
            
            df_deadline = pd.DataFrame([{
                'storypoint': story_points,
                'key': 1, # Single issue count
                'points_per_issue': story_points, # total / count
                'complexity': complexity
            }])

            # V7 Model predicts raw days (No log transform)
            pred_days = model_deadline.predict(df_deadline)[0]
            
            # Post-processing constraints
            deadline_days = max(1, round(pred_days))

            # ----------------------------------------
            # B. ASSIGNEE PREDICTION (V7 XGBoost NLP)
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
                "description": item["description"],
                
                # N8N Expected Fields
                "predicted_deadline_days": deadline_days, # Raw Integer
                "recommended_assignee": recommended_assignee,
                "top_k_recommendations": top_k_recs,
                
                "model_version": "V7_Robust"
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
