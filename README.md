# End-to-End Customer Churn Prediction Pipeline

An end-to-end machine learning pipeline for customer churn prediction built with FastAPI, TensorFlow, Scikit-learn, and AWS S3.

---

## Overview

This project demonstrates a production-oriented machine learning workflow including:

- Data preprocessing
- Feature engineering
- Model training
- Hyperparameter tuning
- Threshold optimization
- Model persistence
- API deployment
- AWS integration

The goal of the project is to predict customer churn based on customer attributes and account activity.

---

## Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- FastAPI
- Pandas
- NumPy
- AWS S3
- Joblib / Pickle

---

## Project Structure

```bash
.
├── main/
│   └── api.py              # FastAPI application
│
├── src/
│   ├── preprocess.py       # Data preprocessing
│   ├── build.py            # Model architecture
│   ├── train.py            # Model training
│   ├── evaluate.py         # Evaluation + threshold optimization
│   ├── predict.py          # Inference pipeline
│   └── retrieve.py         # Retrieve saved artifacts
│
├── inception/
│   └── aws.py              # AWS S3 configuration
│
├── pipeline.py             # End-to-end training pipeline
│
├── req.py                  # Integrates API calls requests and JSON responses
└── README.md
```

---

## Features

### Data Preprocessing

- Missing value handling
- Feature encoding
- Scaling pipeline
- Train / validation / test split

### Model Training

- TensorFlow neural network
- Hyperparameter tuning
- Modular architecture

### Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- Threshold optimization

### Deployment

- FastAPI inference endpoint
- Flexible input handling
- Supports both lists and dictionaries

---

## API Usage

### Run the API

```bash
uvicorn main:app --reload
```

---

### Prediction Endpoint

#### POST `/predict`

Example request:

```json
[2,0,18,0,1695,26,28,14,2,1,1,25,203]
```

Example using curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d "[2,0,18,0,1695,26,28,14,2,1,1,25,203]"
```

---

## Model Pipeline

```text
Raw Data
   ↓
Preprocessing
   ↓
Training
   ↓
Evaluation
   ↓
Threshold Optimization
   ↓
Saved Model + Preprocessor
   ↓
FastAPI Inference
```

---

## Future Improvements

- DVC pipeline orchestration
- CI/CD integration
- Docker containerization
- Model registry
- Automated retraining
- Experiment tracking with MLflow

---

## Author

'Labi-Oluwafemi Folusho

Built as a personal machine learning engineering project focused on production-oriented ML workflows.