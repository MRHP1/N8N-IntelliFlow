#!/usr/bin/env python
# coding: utf-8

# In[28]:


import random, warnings
from typing import Dict, Any

try:
    import ollama
except ImportError:
    raise ImportError("Please install Ollama Python package: pip install ollama")


# In[29]:


# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_MODEL = "phi4-mini:latest"   # Change this if you use another model (e.g. phi4, phi3:mini)
MAX_WORDS = 100


# In[30]:


# # -----------------------------
# # FALLBACK RULE-BASED TEMPLATES
# # -----------------------------
# TEMPLATES = {
#     "frontend": [
#         "Develop responsive UI components for {summary}, ensuring consistent styling and accessibility.",
#         "Implement the user interface for {summary} following design system and usability standards."
#     ],
#     "backend": [
#         "Develop backend logic for {summary}, ensuring efficient data flow and robust API integration.",
#         "Implement server-side features for {summary}, focusing on performance and maintainability."
#     ],
#     "devops": [
#         "Set up CI/CD pipelines and deployment scripts for {summary}, ensuring reliable automation.",
#         "Configure infrastructure and environment management for {summary} using modern DevOps tools."
#     ],
#     "testing": [
#         "Design and execute comprehensive test cases for {summary} to validate system behavior.",
#         "Perform QA validation and regression testing for {summary} to ensure software stability."
#     ],
#     "general": [
#         "Handle the task '{summary}' according to sprint goals and project requirements.",
#         "Implement '{summary}' ensuring correct functionality and documentation."
#     ],
# }


# In[31]:


# -----------------------------
# OLLAMA GENERATION FUNCTION
# -----------------------------
def ollama_generate(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Generate text using a local Ollama model.
    Returns empty string if generation fails.
    """
    try:
        result = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return result["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ Ollama generation failed: {e}")
        return ""


# In[32]:


# -----------------------------
# MAIN DESCRIPTION FUNCTION
# -----------------------------
def generate_description(data: Dict[str, Any]) -> str:
    """
    Generate a concise, technical description for a Jira issue.
    Falls back to rule-based template if Ollama fails.
    """
    summary = data.get("summary", "").strip()
    issuetype = data.get("issuetype", "Task")
    labels = data.get("labels", [])
    project = data.get("project", "Unknown")

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
- Output plain English, no markdown.
- Make it in descriptive form suitable for developers.
- No bullet points or lists.
- Do not mention the summary in the description.
- Provide a single paragraph only.
- No instructions or meta commentary.
"""

    # Try Ollama generation
    desc = ollama_generate(prompt, OLLAMA_MODEL)

    # Fallback if model fails or returns empty output
    # if not desc:
    #     desc = random.choice(TEMPLATES.get(label, TEMPLATES["general"])).format(summary=summary)

    return desc.strip()


# In[33]:


# -----------------------------
# TEST EXAMPLES
# -----------------------------
if __name__ == "__main__":
    tests = [
        {"summary": "Implement login and registration UI", "issuetype": "Story", "labels": ["frontend"], "project": "Aurora"},
        {"summary": "Fix API timeout during user creation", "issuetype": "Bug", "labels": ["backend"], "project": "Meso"},
        {"summary": "Set up Docker and CI/CD pipeline", "issuetype": "Task", "labels": ["devops"], "project": "SpringXD"},
        {"summary": "Perform regression testing for checkout flow", "issuetype": "Improvement", "labels": ["testing"], "project": "UserGrid"},
    ]

    for t in tests:
        print(f"\n[{t['project']}] {t['summary']}")
        print("→", generate_description(t))

