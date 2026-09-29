# Week 8 Report — Model Packaging, GitHub & Deployment Preparation

**Program:** EDP AI/ML Internship  
**Week:** 8  
**Topic:** Model Packaging, REST API Development, GitHub CI, and Deployment Preparation  
**Author:** G . Rahul Reddy (B.Tech IT, Institute of Aeronautical Engineering)  

---

## 1. Executive Summary

In Week 8, the best-performing model developed during Weeks 5–7 (**Tuned XGBoost**, achieving an F1-score of 0.9945 and accuracy of 99.50%) was packaged into a production-ready, containerized inference system.

All requirements and strict constraints have been honored:
* **Zero Model Retraining**: The pre-existing trained model [`models/tuned_xgboost.pkl`](../models/tuned_xgboost.pkl) remains completely untouched and preserved.
* **Zero Data Alteration**: All Week 5–7 processed datasets, metric files, confusion matrices, and reports remain intact.
* **Exact Feature Sequence**: Enforced canonical 30-feature ordering (`Time`, `V1` through `V28`, `Amount`).
* **Precise Preprocessing**: Applied [`models/scaler.pkl`](../models/scaler.pkl) exclusively to `Time` and `Amount`, leaving PCA features `V1`–`V28` unscaled.

---

## 2. Architecture & Packaging Strategy

```
                          Incoming Raw Transaction
                       (JSON Payload / Dict / CSV)
                                    │
                                    ▼
                     [ Pydantic Schema Validation ]
                     • Checks 30 required attributes
                     • Type enforcement (float)
                                    │
                                    ▼
                       [ Inference Preprocessing ]
                     • Extracts ['Time', 'Amount']
                     • Applies models/scaler.pkl
                     • Keeps V1–V28 unscaled
                     • Enforces exact 30-column order
                                    │
                                    ▼
                       [ XGBoost Inference Engine ]
                     • models/tuned_xgboost.pkl
                     • Computes predict() & predict_proba()
                                    │
                                    ▼
                       [ Structured API Response ]
                     • Prediction (0 or 1)
                     • Label ("Legitimate" or "Fraudulent")
                     • Fraud Probability ([0.0 - 1.0])
                     • Risk Tier (LOW / MEDIUM / HIGH)
```

---

## 3. Implemented Components

### 3.1 Inference Engine (`src/predict.py`)
Encapsulated in the `FraudPredictor` class:
* Loads model and scaler artifacts upon initialization.
* Re-indexes incoming columns to guarantee column order independence (prevents silent permutation errors).
* Categorizes transactions into actionable risk tiers (`LOW`, `MEDIUM`, `HIGH`) based on predicted probabilities.
* Includes a command-line interface:
  ```bash
  python src/predict.py --sample legit
  python src/predict.py --sample fraud
  ```

### 3.2 Production REST API (`app.py`)
Built using **FastAPI** with automatic OpenAPI/Swagger documentation at `/docs`:
* `GET /health`: Returns service health status, model type, and artifact readiness.
* `GET /sample`: Provides pre-validated legitimate and fraudulent sample transaction payloads for instant testing.
* `POST /predict`: Evaluates a single transaction, accepting custom threshold queries (`?threshold=0.5`).
* `POST /predict/batch`: High-throughput batch prediction returning individual results and aggregated counts (`fraud_count`, `legitimate_count`).

### 3.3 Automated Test Suite (`tests/test_inference.py`)
11 comprehensive unit and integration tests covering:
1. Artifact existence and loading.
2. Canonical 30-feature sequence validation.
3. Scaler scope (strictly `Time` and `Amount`).
4. Single legitimate prediction verification.
5. Single fraudulent prediction verification.
6. Key order independence (shuffled keys test).
7. Missing feature rejection (`ValueError`).
8. Batch prediction handling.
9. FastAPI `/health` endpoint.
10. FastAPI `/predict` endpoint.
11. FastAPI `/predict/batch` endpoint.

**Test Run Result:**
```text
Ran 11 tests in 0.158s
OK (All 11 tests passed)
```

### 3.4 Containerization (`Dockerfile` & `.dockerignore`)
* Minimal `python:3.11-slim` container base.
* Packages application code, `requirements.txt`, and production model binaries (`tuned_xgboost.pkl`, `scaler.pkl`).
* Exposes port `8000` with container healthchecks configured for Kubernetes and Docker orchestrators.

### 3.5 GitHub Actions CI (`.github/workflows/ci.yml`)
* Automates validation on every `push` and `pull_request` to `main`.
* Tests cross-compatibility on Python 3.10 and 3.11.
* Executes unit test discovery and inference smoke tests automatically.

---

## 4. Verification & Prediction Samples

| Sample Type | True Class | Predicted Class | Label | Fraud Probability | Risk Level |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Legitimate Sample** | 0 | 0 | Legitimate | `0.01%` (0.0001) | `LOW` |
| **Fraudulent Sample** | 1 | 1 | Fraudulent | `99.98%` (0.9998) | `HIGH` |

---

## 5. Deployment Instructions

### Local Execution
```bash
# Start FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Open interactive documentation: `http://localhost:8000/docs`

### Docker Deployment
```bash
# Build the Docker image
docker build -t fraud-detection-api:latest .

# Run the container
docker run -d -p 8000:8000 --name fraud-api fraud-detection-api:latest
```

---

## 6. Conclusion
Week 8 successfully bridges the gap between machine learning experimentation and production engineering. The tuned XGBoost model from Weeks 5–7 is now fully packaged, tested, documented, and prepared for seamless deployment across local, Docker, and cloud environments.
