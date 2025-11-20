# N8N-IntelliFlow
## Intelligent Workflow Automation for Agile Project Management, using n8n and Machine Learning.

N8N-IntelliFlow is an end-to-end intelligent automation system that integrates **n8n workflow orchestration** with **Machine Learning models** to automate project management tasks such as **task assignment**, **deadline prediction**, and **resource optimization**. It brings real-time predictions, automated decision support, and Explainable AI (XAI) into Agile/Scrum workflows.

## Key Features
### Automated Task Assignment
- ML models: Random Forest / XGBoost
- Predicts best assignee based on issue metadata, workload, and history

### Deadline Prediction
- LSTM/DNN regression models
- Predicts estimated completion time

### Resource Allocation Optimization
- Linear Programming and Reinforcement Learning

### n8n Integration
- Real-time automation via triggers, webhooks, HTTP nodes

### Explainable AI (SHAP & LIME)
- Feature importance
- Rationale for recommendations

## Project Structure
```
N8N-IntelliFlow/
├── api/
│   ├── app.py
│   ├── models_assignment.joblib
│   ├── models_deadline.joblib
│   └── optimizer/
├── notebooks/
│   ├── training_assignment.ipynb
│   ├── training_deadline.ipynb
│   └── preprocessing.ipynb
├── n8n-workflows/
│   ├── predict_assignment.json
│   ├── predict_deadline.json
│   └── full_pipeline.json
├── data/
│   ├── raw/
│   └── processed/
└── docs/
    └── TKTI Progress 3.pdf
```

## System Architecture
```
Jira → n8n Trigger → Flask ML API → n8n Actions → Notifications / Updates
```

## Tech Stack
- n8n
- Python + Flask
- XGBoost, RandomForest, LSTM, DNN
- SHAP, LIME
- Linear Programming & RL

## Dataset
Agile Scrum Sprint Velocity Dataset  
https://github.com/RandulaKoralage/AgileScrumSprintVelocityDataSet

## License
MIT License
