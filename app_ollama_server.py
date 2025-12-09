from flask import Flask, request, jsonify
from typing import Dict, Any
import ollama

app = Flask(__name__)

OLLAMA_MODEL = "phi4-mini:latest"
MAX_WORDS = 100


def ollama_generate(prompt: str, model: str = OLLAMA_MODEL) -> str:
    try:
        result = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return result["message"]["content"].strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        return ""


def normalize_labels(labels):
    """
    Fix labels format because n8n sometimes sends:
    - a list → ["bug", "backend"]
    - a string → "['bug','backend']"
    - empty string
    """

    if labels is None:
        return []

    # Already list
    if isinstance(labels, list):
        return [str(x).strip() for x in labels]

    # Try convert string like "['bug','backend']"
    if isinstance(labels, str):
        cleaned = labels.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                import ast
                lst = ast.literal_eval(cleaned)
                if isinstance(lst, list):
                    return [str(x).strip() for x in lst]
            except:
                pass
        # fallback → treat as single label
        return [cleaned]

    # fallback
    return [str(labels)]


def generate_description(data: Dict[str, Any]) -> str:
    summary = (data.get("summary") or "").strip()
    issuetype = data.get("issuetype", "Task")
    project = data.get("project", "UNKNOWN")

    # NEW: reviewer feedback from PR review
    review_feedback = (data.get("review_feedback") or "").strip()

    labels = normalize_labels(data.get("labels", []))
    label = labels[0] if labels else "general"

    # -------- PROMPT --------
    prompt = f"""
Generate a concise (under {MAX_WORDS} words) Jira issue description.

Project: {project}
Issue Type: {issuetype}
Primary Label: {label}

Summary: {summary}
Reviewer Feedback: {review_feedback}

Guidelines:
- Keep it technical and factual.
- Do not exceed {MAX_WORDS} words.
- Avoid generic or redundant text.
- Output plain English, no markdown, no newlines.
- Make it in descriptive form suitable for developers.
- No bullet points or lists.
- Do not mention the summary in the description.
- No instructions or meta commentary.
- Keep it relevant to the issue type and label.
- Keep it in a single paragraph format.
- Integrate a "Feedback" section for reviewer feedback if provided.
"""

    desc = ollama_generate(prompt)
    return desc.replace("\n", " ").strip()


# ---------- API ----------
@app.route("/generate-description", methods=["POST"])
def api_generate():
    try:
        data = request.get_json(force=True)
        desc = generate_description(data)

        return jsonify({
            "key": data.get("key"),
            "description": desc
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
