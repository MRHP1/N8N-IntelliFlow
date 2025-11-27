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
try:
    model_assignee = joblib.load("models/assignee_model_v3_aurora.joblib")
    label_encoder = joblib.load("models/assignee_label_encoder_aurora.joblib")
    model_deadline = joblib.load("models/deadline_model_4features.joblib")
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model_assignee, model_deadline, label_encoder = None, None, None


# ============================================================
# Helpers
# ============================================================

REQUIRED_FIELDS = {"key", "issuetype", "priorityid", "storypoint", "project", "summary"}

def ensure_models_loaded():
    if model_deadline is None or model_assignee is None or label_encoder is None:
        raise RuntimeError("Models not loaded properly. Check your model paths.")


def unwrap_n8n(data):
    """
    Accepts ALL possible n8n formats:
    - { Json: {...} }
    - [ { Json: {...} } ]
    - raw {...}
    - [ {...} ]
    """

    # If array with wrapper
    if isinstance(data, list):
        cleaned = []
        for entry in data:
            if isinstance(entry, dict) and "Json" in entry:
                cleaned.append(entry["Json"])
            else:
                cleaned.append(entry)
        return cleaned

    # single dict wrapper
    if isinstance(data, dict) and "Json" in data:
        return [data["Json"]]

    # single dict without wrapper
    if isinstance(data, dict):
        return [data]

    raise ValueError("Unrecognized input format from n8n")


def normalize_fields(item):
    """Normalize storyPoint/storypoint, ignore extra fields."""
    normalized = {}

    # normalize storyPoint → storypoint
    if "storypoint" in item:
        normalized["storypoint"] = item["storypoint"]
    elif "storyPoint" in item:
        normalized["storypoint"] = item["storyPoint"]
    else:
        normalized["storypoint"] = 0

    for f in ["key", "issuetype", "priorityid", "project", "summary"]:
        normalized[f] = item.get(f)

    for field in REQUIRED_FIELDS:
        if normalized.get(field) is None:
            raise ValueError(f"Missing required field: {field}")

    return normalized


def priority_multiplier(pid):
    mapping = {1: 0.7, 2: 0.85, 3: 1.0, 4: 1.2, 5: 1.4}
    return mapping.get(int(pid), 1.0)


def extract_label(summary, desc=""):
    text = f"{summary} {desc}".lower()
    if any(k in text for k in ["ui","react","frontend","css","html","layout","component"]):
        return "frontend"
    if any(k in text for k in ["api","backend","server","database","controller","prisma","nest"]):
        return "backend"
    if any(k in text for k in ["deploy","docker","pipeline","ci","cd","aws","devops"]):
        return "devops"
    if any(k in text for k in ["test","qa","bug","error","issue"]):
        return "testing"
    return "general"


# ============================================================
# 3️⃣ Predict Endpoint
# ============================================================
@app.route("/predict/all", methods=["POST"])
def predict_all_api():
    try:
        ensure_models_loaded()
        raw = request.get_json(force=True)

        # NEW: robust n8n unwrapping
        items = unwrap_n8n(raw)

        results = []  # Must exist before try blocks

        for incoming in items:
            item = normalize_fields(incoming)

            df_deadline = pd.DataFrame([{
                "issuetype": item["issuetype"],
                "priorityid": item["priorityid"],
                "storypoint": item["storypoint"],
                "project": item["project"],
                "summary": item["summary"]
            }])

            # deadline prediction
            yp = model_deadline.predict(df_deadline)[0]
            deadline_days = int(np.expm1(yp))
            deadline_days = max(1, round(min(deadline_days * priority_multiplier(item["priorityid"]), 180)))

            # assignee prediction
            df2 = df_deadline.copy()
            df2["labels"] = extract_label(item["summary"])

            expected_cols = [
                "storypoint","priorityid","velocity_sp_per_day","sprintlength",
                "completedissuesestimatesum","noofdevelopers","status",
                "issuetype","project","labels"
            ]

            for col in expected_cols:
                if col not in df2.columns:
                    df2[col] = 0 if col in ["storypoint","priorityid","velocity_sp_per_day",
                                            "sprintlength","completedissuesestimatesum","noofdevelopers"] else "Unknown"

            df2 = df2[expected_cols]
            prob = model_assignee.predict_proba(df2)[0]
            classes = label_encoder.inverse_transform(np.arange(len(prob)))

            best = np.argmax(prob)
            top5 = np.argsort(prob)[::-1][:5]

            recs = [{"assignee": classes[i], "probability": float(prob[i])} for i in top5]

            results.append({
                "key": item["key"],
                "issuetype": item["issuetype"],
                "project": item["project"],
                "predicted_deadline_days": deadline_days,
                "recommended_assignee": classes[best],
                "top_k_recommendations": recs
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ============================================================
# 4️⃣ Run Flask
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
