from flask import Flask, request, jsonify
from typing import Dict, Any
import ollama
import random

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


def generate_description(data: Dict[str, Any]) -> str:
    summary = data.get("summary", "").strip()
    issuetype = data.get("issuetype", "Task")
    labels = data.get("labels", [])
    project = data.get("project", "Unknown")

    # extract first label
    if isinstance(labels, list):
        label = labels[0].lower() if labels else "general"
    else:
        label = str(labels).lower()

    prompt = f"""
Generate a concise (under {MAX_WORDS} words) Jira issue description.

Project: {project}
Issue Type: {issuetype}
Label: {label}
Summary: {summary}

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
"""

    desc = ollama_generate(prompt)
    return desc.strip()


# ---------- API ROUTE ----------
@app.route("/generate-description", methods=["POST"])
def api_generate():
    try:
        data = request.get_json(force=True)
        desc = generate_description(data)

        # return description + key (NEW)
        return jsonify({
            "key": data.get("key", None),
            "description": desc
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
