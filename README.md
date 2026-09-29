ML Classification Project — Credit Card Fraud Detection

Program: EDP AI/ML Internship
Weeks: 5–8
Project Type: Machine Learning Classification

1. Project Overview

This project focuses on detecting fraudulent credit card transactions using supervised machine learning classification algorithms.

The project continues the work from Week 4, where class imbalance was handled using SMOTE. In Weeks 5–7, the processed dataset is used to train, evaluate, compare, and tune two classification models:

Random Forest

XGBoost

The complete workflow covers data preprocessing, model training, evaluation, hyperparameter tuning, and performance comparison.

2. Objectives

The main objectives of this project are:

Prepare the credit card transaction dataset for machine learning.

Handle class imbalance using the preprocessing pipeline developed earlier.

Train a Random Forest classification model.

Train an XGBoost classification model.

Evaluate models using multiple classification metrics.

Tune model hyperparameters using GridSearchCV.

Compare tuned and untuned models.

Save trained models and evaluation results for further analysis.

3. Dataset

The project uses a credit card transaction dataset containing transaction-related features and a target column indicating whether a transaction is fraudulent.

The standard Kaggle Credit Card Fraud Detection dataset contains:

284,807 transactions

492 fraudulent transactions

Highly imbalanced target classes

Features such as Time, Amount, and anonymized PCA-transformed variables (V1–V28)

Target column: Class

0 = Legitimate transaction

1 = Fraudulent transaction

Important Note

The dataset currently used in this project contains 5,040 rows after the supplied dataset was prepared/reduced for the internship work. Therefore, the reported model results in this repository should be interpreted as results on this current sample dataset and not as results obtained from the full 284,807-row Kaggle dataset.

4. Class Imbalance

Fraud detection is an imbalanced classification problem because fraudulent transactions are much fewer than legitimate transactions.

Class imbalance can cause a model to favor the majority class. Therefore, the preprocessing pipeline uses techniques from the previous internship work to prepare the data appropriately.

The train/test split is performed using stratification so that the class distribution is maintained between training and testing data.

5. Project Workflow

Credit Card Dataset
        |
        v
Data Preprocessing
        |
        +--> Feature/Target Separation
        |
        +--> Feature Scaling
        |
        +--> Stratified Train/Test Split
        |
        v
Processed Dataset
        |
        +----------------------+
        |                      |
        v                      v
 Random Forest             XGBoost
        |                      |
        v                      v
 Untuned Models
        |                      |
        +----------+-----------+
                   |
                   v
          Model Evaluation
                   |
                   +--> Accuracy
                   +--> Precision
                   +--> Recall
                   +--> F1 Score
                   +--> ROC-AUC
                   +--> Confusion Matrix
                   |
                   v
          Hyperparameter Tuning
              GridSearchCV
                   |
                   v
          Tuned Random Forest
          Tuned XGBoost
                   |
                   v
          Final Model Comparison

6. Methodology

6.1 Data Preprocessing

The preprocessing script performs the following operations:

Loads the raw credit card dataset.

Separates input features and target labels.

Scales numerical features such as Time and Amount.

Performs a stratified train/test split.

Saves processed training and testing datasets.

Generated files:

data/processed/
├── X_train.csv
├── X_test.csv
├── y_train.csv
└── y_test.csv

The current execution produced:

Train shape: (4032, 30)
Test shape:  (1008, 30)

Train fraud rate: 0.7937%
Test fraud rate:  0.7937%

7. Model Training

7.1 Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to produce a robust classification model.

The trained model is saved as:

models/random_forest.pkl

Untuned test accuracy:

99.21%

7.2 XGBoost

XGBoost is a gradient boosting algorithm that builds decision trees sequentially and improves the model by focusing on previous prediction errors.

The trained model is saved as:

models/xgboost.pkl

Untuned test accuracy:

99.40%

8. Model Evaluation

The models are evaluated using:

Accuracy

Measures the overall percentage of correctly classified transactions.

Precision

Measures how many transactions predicted as fraud are actually fraudulent.

Recall

Measures how many actual fraudulent transactions are correctly detected.

F1 Score

Provides a balance between precision and recall.

ROC-AUC

Measures the model's ability to distinguish between legitimate and fraudulent transactions across classification thresholds.

Confusion Matrix

Shows:

True Positives

True Negatives

False Positives

False Negatives

9. Hyperparameter Tuning

Hyperparameter tuning is performed using GridSearchCV with 5-fold cross-validation.

The tuning objective uses:

scoring = f1_weighted

Random Forest Search

The tuned Random Forest selected:

n_estimators = 100
max_depth = 10
min_samples_split = 2

Best cross-validation F1:

0.9903

XGBoost Search

The tuned XGBoost selected:

n_estimators = 200
max_depth = 3
learning_rate = 0.1

Best cross-validation F1:

0.9930

10. Final Model Comparison

The current evaluation produced the following results:

Model

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Random Forest

99.21%

98.42%

99.21%

98.81%

98.73%

XGBoost

99.40%

99.31%

99.40%

99.31%

99.46%

Tuned Random Forest

99.21%

98.42%

99.21%

98.81%

98.29%

Tuned XGBoost

99.50%

99.45%

99.50%

99.45%

99.10%

Best Overall Model

Based on the current test results, Tuned XGBoost provides the best overall performance for accuracy, precision, recall, and F1 score.

However, the untuned XGBoost achieved a slightly higher ROC-AUC:

Untuned XGBoost ROC-AUC = 99.46%
Tuned XGBoost ROC-AUC   = 99.10%

Therefore, model selection should depend on the metric that is most important for the intended fraud-detection application.

11. Project Structure

ML-Classification-Project/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   │
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_random_forest.ipynb
│   ├── 03_xgboost.ipynb
│   └── 04_evaluation_and_tuning.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── evaluate.py
│   ├── tune.py
│   ├── package_scaler.py
│   └── predict.py
│
├── models/
│   ├── scaler.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── tuned_random_forest.pkl
│   └── tuned_xgboost.pkl
│
├── tests/
│   ├── __init__.py
│   └── test_inference.py
│
├── results/
│   ├── metrics/
│   ├── confusion_matrices/
│   └── comparison.csv
│
├── reports/
│   ├── week5_report.md
│   ├── week6_7_report.md
│   └── week8_report.md
│
├── app.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── .gitignore
└── README.md

12. Installation

Clone or download the repository and open a terminal in the project directory.

Create a virtual environment:

python -m venv .venv

Activate it on Windows PowerShell:

.venv\Scripts\Activate.ps1

Install dependencies:

python -m pip install -r requirements.txt

Test the important libraries:

python -c "import pandas, sklearn, xgboost, joblib; print('All libraries installed successfully')"

13. Running the Project

Run the scripts from the project root:

Step 1 — Preprocessing

python src/preprocessing.py

Step 2 — Train Random Forest

python src/train_random_forest.py

Step 3 — Train XGBoost

python src/train_xgboost.py

Step 4 — Hyperparameter Tuning

python src/tune.py

Step 5 — Evaluation and Comparison

python src/evaluate.py

14. Notebook Execution Order

If using Jupyter Notebook, run the notebooks in this order:

01_data_preprocessing.ipynb
        ↓
02_random_forest.ipynb
        ↓
03_xgboost.ipynb
        ↓
04_evaluation_and_tuning.ipynb

15. Data Leakage Prevention

To avoid data leakage:

The train/test split is performed before model evaluation.

Scaling parameters are derived from training data.

Test data is kept separate during model training.

Cross-validation is used during hyperparameter tuning.

Final test-set evaluation is performed after model selection.

This helps provide a more reliable estimate of model performance.

16. Generated Results

The project generates:

results/
├── metrics/
│   ├── random_forest_metrics.txt
│   ├── xgboost_metrics.txt
│   ├── tuned_random_forest_metrics.txt
│   └── tuned_xgboost_metrics.txt
│
├── confusion_matrices/
│   └── generated confusion matrix files
│
└── comparison.csv

The trained models are stored in:

models/

17. Internship Week Mapping

Week 5 — Classification Models

Activities:

Prepared processed dataset.

Trained Random Forest.

Trained XGBoost.

Saved trained models.

Checked initial model performance.

Week 6 — Model Evaluation

Activities:

Calculated accuracy.

Calculated precision.

Calculated recall.

Calculated F1 score.

Calculated ROC-AUC.

Generated confusion matrices.

Compared model performance.

Week 7 — Hyperparameter Tuning

Activities:

Used GridSearchCV.

Applied 5-fold cross-validation.

Tuned Random Forest.

Tuned XGBoost.

Compared tuned and untuned models.

Selected the strongest model based on evaluation metrics.

Week 8 — Model Packaging & Deployment Preparation

Activities:

Packaged Tuned XGBoost model (models/tuned_xgboost.pkl) and Scaler (models/scaler.pkl).

Enforced canonical 30-feature sequence: Time, V1..V28, Amount.

Restricted scaling exclusively to Time and Amount while preserving V1..V28 unscaled.

Implemented standalone inference engine (src/predict.py) with risk scoring.

Built production REST API using FastAPI (app.py) with OpenAPI / Swagger documentation.

Configured containerized deployment with Dockerfile and .dockerignore.

Created automated GitHub Actions CI workflow (.github/workflows/ci.yml).

Implemented automated test suite (tests/test_inference.py) verifying pipeline integrity.

18. Learning Outcomes

After completing this project, the following concepts were practiced:

Binary classification

Credit card fraud detection

Class imbalance

Data preprocessing

Feature scaling

Stratified train/test splitting

Random Forest

XGBoost

Model evaluation

Confusion matrices

Precision and recall

F1 score

ROC-AUC

Cross-validation

GridSearchCV

Hyperparameter tuning

Model comparison

Saving machine learning models using Joblib

19. Future Improvements

Possible future improvements include:

Use the complete Kaggle dataset for final benchmarking.

Evaluate minority-class precision and recall separately.

Experiment with PR-AUC because fraud detection is highly imbalanced.

Test additional algorithms such as LightGBM or CatBoost.

Perform threshold optimization based on business requirements.

Add explainability using SHAP.

Build a Streamlit dashboard for real-time fraud prediction.

Deploy the model as a REST API.

Add automated ML model monitoring.

20. Conclusion

This project demonstrates an end-to-end machine learning classification workflow for credit card fraud detection.

Random Forest and XGBoost were trained and evaluated using multiple performance metrics. Hyperparameter tuning with GridSearchCV was then applied to improve model performance.

On the current sample dataset, Tuned XGBoost achieved the highest accuracy and F1 score, making it the strongest model according to those metrics.

The project also demonstrates important real-world machine learning practices such as preprocessing, stratified data splitting, model comparison, cross-validation, hyperparameter tuning, and result tracking.

21. Week 8 — Model Packaging, REST API & Deployment Preparation

In Week 8, the project transition from model experimentation to a production-grade inference service:

21.1 Packaged Model & Preprocessing Integrity
- Best Model: models/tuned_xgboost.pkl
- Scaler: models/scaler.pkl
- Exact 30-feature sequence enforced:
  ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
   'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
   'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
- Selective Scaling: Only Time and Amount are scaled; V1 through V28 remain unscaled.

21.2 Command-Line Inference
Run standalone inference on pre-validated transactions:

python src/predict.py --sample legit
python src/predict.py --sample fraud

21.3 Production REST API (FastAPI)
Start the Uvicorn server locally:

uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Available Endpoints:
- GET /: API root and documentation links
- GET /health: Service health and model status
- GET /sample: Pre-populated legitimate and fraud payloads
- POST /predict: Single transaction prediction
- POST /predict/batch: Batch transaction predictions

Interactive Swagger UI documentation is available at:
http://localhost:8000/docs

21.4 Example API Requests (cURL)

Single Transaction Prediction:
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Time": 31604.21, "V1": -0.42015, "V2": -0.43649, "V3": -1.51860, "V4": 2.98559,
       "V5": -1.63805, "V6": -2.23789, "V7": -2.27306, "V8": -1.00042, "V9": -0.78319,
       "V10": 0.57358, "V11": -0.57348, "V12": 1.55848, "V13": 2.59228, "V14": 1.45910,
       "V15": 0.82946, "V16": -1.08856, "V17": 1.44921, "V18": -0.39534, "V19": -1.57161,
       "V20": 1.14596, "V21": 0.23719, "V22": 0.67931, "V23": -0.45521, "V24": 0.28307,
       "V25": 1.18469, "V26": -0.61417, "V27": -1.60533, "V28": 0.66319, "Amount": 149.97
     }'

Response:
{
  "prediction": 0,
  "label": "Legitimate",
  "fraud_probability": 0.0001,
  "threshold_used": 0.5,
  "risk_level": "LOW"
}

21.5 Containerization (Docker)
Build the image:
docker build -t fraud-detection-api:latest .

Run the container:
docker run -d -p 8000:8000 --name fraud-api fraud-detection-api:latest

21.6 CI/CD & Automated Testing
Run the complete test suite locally:
python -m unittest discover -s tests

Continuous integration is automated via GitHub Actions (.github/workflows/ci.yml) testing Python 3.10 and 3.11 environments.

22. Author

G . Rahul Reddy

B.Tech — Information Technology
Institute of Aeronautical Engineering, Hyderabad

23. License

This project is created for educational and internship purposes.