# Credit Card Fraud Detection — Week 8

## EDP AI/ML Internship — Model Packaging & Deployment Preparation

**Project:** Credit Card Fraud Detection
**Internship Phase:** Week 8
**Focus:** Model Packaging, Inference, REST API, Testing, Docker, and CI

---

## 1. Overview

This repository contains the **Week 8 implementation** of the Credit Card Fraud Detection project.

The machine learning model used in this phase was developed and evaluated during the previous Weeks 5–7. Week 8 does not focus on developing a new machine learning model. Instead, it focuses on preparing the existing trained model for practical use.

The main goal of Week 8 is to convert the trained XGBoost model into a reusable prediction system with:

* Packaged model artifacts
* Reusable preprocessing
* Standalone inference
* REST API
* Automated testing
* Docker containerization
* Continuous integration

The previous Weeks 5–7 project remains the model-development stage, while this repository represents the **packaging and deployment-preparation stage**.

---

## 2. Relationship with Weeks 5–7

The project is divided into two stages.

### Weeks 5–7 — Model Development

The earlier project focused on:

```text
Data Preparation
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Hyperparameter Tuning
      ↓
Model Selection
```

The trained and tuned XGBoost model from that stage is reused in Week 8.

### Week 8 — Model Packaging and Deployment Preparation

This repository extends the previous work:

```text
Trained Model
      ↓
Model Packaging
      ↓
Preprocessing Packaging
      ↓
Inference
      ↓
REST API
      ↓
Testing
      ↓
Docker
      ↓
Continuous Integration
```

This approach avoids retraining the model and concentrates on making the existing model usable as an application.

---

## 3. Week 8 Objectives

The objectives of this phase are:

* Package the trained XGBoost model.
* Package the required preprocessing scaler.
* Maintain the correct input feature order.
* Create a reusable prediction module.
* Generate fraud probabilities.
* Assign a simple risk level to predictions.
* Build a REST API using FastAPI.
* Provide interactive API documentation.
* Add automated tests.
* Containerize the application using Docker.
* Configure GitHub Actions for continuous integration.
* Prepare the application for future cloud deployment.

---

## 4. Model Used

Week 8 reuses the tuned XGBoost classifier developed during the previous machine learning stage.

The packaged model is stored at:

```text
models/tuned_xgboost.pkl
```

The preprocessing scaler is stored separately:

```text
models/scaler.pkl
```

The model and scaler can be loaded directly by the inference application without retraining.

---

## 5. Input Features

The prediction system expects exactly **30 features** in the following order:

```text
Time
V1
V2
V3
V4
V5
V6
V7
V8
V9
V10
V11
V12
V13
V14
V15
V16
V17
V18
V19
V20
V21
V22
V23
V24
V25
V26
V27
V28
Amount
```

The feature order is kept fixed to ensure that input data is passed to the trained model correctly.

---

## 6. Preprocessing

The Week 8 inference pipeline uses the packaged scaler:

```text
models/scaler.pkl
```

Only the following features are scaled:

```text
Time
Amount
```

The features:

```text
V1 – V28
```

are passed without additional scaling.

This preprocessing behavior is preserved so that inference remains consistent with the model's expected input format.

---

## 7. Week 8 Project Structure

```text
EDP-AI-ML-Week8/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── models/
│   ├── tuned_xgboost.pkl
│   └── scaler.pkl
│
├── src/
│   ├── package_scaler.py
│   └── predict.py
│
├── tests/
│   ├── __init__.py
│   └── test_inference.py
│
├── reports/
│   └── week8_report.md
│
├── app.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── .gitignore
└── README.md
```

The repository may also contain supporting files carried forward from the previous project because the Week 8 implementation depends on the existing trained model.

---

## 8. Model Packaging

The trained model is stored as a reusable Joblib file:

```text
models/tuned_xgboost.pkl
```

The fitted scaler is stored as:

```text
models/scaler.pkl
```

These artifacts allow the application to load the model and preprocessing configuration directly during inference.

No model retraining is required when starting the prediction service.

---

## 9. Standalone Prediction

The prediction module is located at:

```text
src/predict.py
```

It provides reusable inference functionality and can also be used from the command line.

### Test a legitimate transaction

```powershell
python src/predict.py --sample legit
```

### Test a fraud transaction

```powershell
python src/predict.py --sample fraud
```

The output provides:

* Prediction
* Transaction label
* Fraud probability
* Risk level

Example:

```text
Prediction: 0
Label: Legitimate
Fraud Probability: 0.0100%
Risk Level: LOW
```

Example fraud prediction:

```text
Prediction: 1
Label: Fraudulent
Fraud Probability: 99.9800%
Risk Level: HIGH
```

The displayed probability depends on the model and input transaction.

---

## 10. FastAPI REST API

Week 8 exposes the trained model through a REST API using FastAPI.

The API application is:

```text
app.py
```

Start the application with:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will run at:

```text
http://localhost:8000
```

---

## 11. API Endpoints

| Method | Endpoint         | Description                     |
| ------ | ---------------- | ------------------------------- |
| GET    | `/`              | Displays basic API information  |
| GET    | `/health`        | Checks API and model status     |
| GET    | `/sample`        | Returns sample transaction data |
| POST   | `/predict`       | Predicts one transaction        |
| POST   | `/predict/batch` | Predicts multiple transactions  |

---

## 12. Swagger Documentation

FastAPI automatically provides interactive API documentation.

After starting the server, open:

```text
http://localhost:8000/docs
```

Swagger can be used to:

* View API endpoints.
* View request formats.
* Submit transaction data.
* Test predictions.
* Inspect API responses.

This makes it possible to test the model through a browser without creating a separate frontend application.

---

## 13. Single Prediction

The endpoint:

```text
POST /predict
```

accepts one transaction containing the required 30 features.

A prediction response contains information such as:

```json
{
  "prediction": 0,
  "label": "Legitimate",
  "fraud_probability": 0.0001,
  "threshold_used": 0.5,
  "risk_level": "LOW"
}
```

The prediction value represents the classification produced by the trained model.

---

## 14. Batch Prediction

The endpoint:

```text
POST /predict/batch
```

allows multiple transactions to be processed in a single request.

Batch prediction is useful when several transaction records need to be evaluated together instead of sending separate requests for every transaction.

---

## 15. Automated Testing

Week 8 includes automated tests under:

```text
tests/test_inference.py
```

Run the tests using:

```powershell
python -m unittest discover -s tests
```

The tests check important parts of the inference system, including:

* Model loading
* Scaler loading
* Input feature handling
* Prediction behavior
* API functionality
* Response structure
* Sample inference

### Validation Result

The completed test run produced:

```text
11 tests passed
OK
```

---

## 16. Docker Containerization

The application can be packaged into a Docker image.

The Docker configuration is provided in:

```text
Dockerfile
```

Build the Docker image:

```powershell
docker build -t fraud-detection-api:latest .
```

Run the container:

```powershell
docker run -d -p 8000:8000 --name fraud-api fraud-detection-api:latest
```

After starting the container, the API is available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

The `.dockerignore` file prevents unnecessary files from being included in the Docker build context.

---

## 17. Continuous Integration

GitHub Actions is configured using:

```text
.github/workflows/ci.yml
```

The workflow automatically runs project checks when changes are pushed to GitHub.

The CI process installs the required dependencies and runs the automated tests.

This helps identify problems before the application is deployed.

---

## 18. Installing Dependencies

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

The required libraries include the machine learning and API dependencies needed for model inference and testing.

---

## 19. Week 8 Validation

The Week 8 implementation was validated through the following checks.

### Model artifacts

```text
tuned_xgboost.pkl
scaler.pkl
```

were available for inference.

### Feature handling

```text
30 input features
Time and Amount scaled
V1–V28 kept without additional scaling
```

### Sample inference

```text
Legitimate sample → Prediction 0
Fraud sample      → Prediction 1
```

### Automated tests

```text
11 tests passed
OK
```

### API

The following endpoints were implemented:

```text
/
 /health
 /sample
 /predict
 /predict/batch
```

### Containerization

The project includes:

```text
Dockerfile
.dockerignore
```

### Continuous Integration

The project includes:

```text
.github/workflows/ci.yml
```

---

## 20. Week 8 Deliverables

The completed Week 8 implementation contains:

* Packaged XGBoost model
* Packaged preprocessing scaler
* Fixed 30-feature input structure
* Standalone inference module
* Fraud probability calculation
* Risk-level classification
* FastAPI REST API
* Swagger documentation
* Single prediction endpoint
* Batch prediction endpoint
* Automated tests
* Docker configuration
* GitHub Actions workflow
* Week 8 project report

---

## 21. Week 8 Outcome

The main outcome of Week 8 is a transition from a trained machine learning model to a reusable inference application.

The workflow is:

```text
Existing Trained Model
        ↓
Packaged Model + Scaler
        ↓
Prediction Module
        ↓
FastAPI REST API
        ↓
Automated Testing
        ↓
Docker Container
        ↓
CI Validation
```

This structure makes the trained fraud detection model easier to test, reuse, and prepare for deployment.

---

## 22. Future Improvements

The current implementation can be extended with additional production features such as:

* Cloud deployment
* API authentication
* HTTPS
* Request logging
* Monitoring
* Model version management
* Frontend integration
* Database integration
* Model performance monitoring

These features are possible future extensions and are not required for the current Week 8 implementation.

---

## 23. Conclusion

Week 8 focuses on preparing the Credit Card Fraud Detection model for practical use.

Instead of training another model, the existing tuned XGBoost model is packaged with its required preprocessing configuration. A standalone inference module and FastAPI service provide prediction capabilities, while automated tests, Docker, and GitHub Actions improve the project's reliability and deployment readiness.

Therefore, Week 8 serves as the **model packaging and deployment-preparation stage** following the machine learning development completed during Weeks 5–7.

---

## Author

**G . Rahul Reddy**

B.Tech — Information Technology
Institute of Aeronautical Engineering, Hyderabad

---

## License

The project is intended for educational, learning, and internship purposes.
