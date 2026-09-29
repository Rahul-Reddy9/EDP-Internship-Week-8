"""
test_inference.py
-----------------
Week 8: Unit and integration tests for model packaging, feature order integrity,
and FastAPI endpoint execution.
"""

import os
import unittest
import numpy as np
import pandas as pd
import joblib

from src.predict import (
    FraudPredictor,
    FEATURE_ORDER,
    SCALE_COLS,
    get_sample_transaction,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
)
from app import app
from fastapi.testclient import TestClient


class TestFraudModelPackaging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = FraudPredictor()
        cls.client = TestClient(app)

    def test_01_artifacts_exist_and_load(self):
        """Verify model and scaler files exist and load properly."""
        self.assertTrue(os.path.exists(DEFAULT_MODEL_PATH), f"Missing {DEFAULT_MODEL_PATH}")
        self.assertTrue(os.path.exists(DEFAULT_SCALER_PATH), f"Missing {DEFAULT_SCALER_PATH}")
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.scaler)

    def test_02_canonical_feature_order_integrity(self):
        """Verify exact 30-feature sequence: Time, V1..V28, Amount."""
        self.assertEqual(len(FEATURE_ORDER), 30)
        self.assertEqual(FEATURE_ORDER[0], "Time")
        self.assertEqual(FEATURE_ORDER[-1], "Amount")
        for i in range(1, 29):
            self.assertEqual(FEATURE_ORDER[i], f"V{i}")

        # Check that the model artifact's feature_names_in_ matches exactly
        if hasattr(self.predictor.model, "feature_names_in_"):
            np.testing.assert_array_equal(self.predictor.model.feature_names_in_, FEATURE_ORDER)

    def test_03_scaler_attributes(self):
        """Verify scaler only transforms Time and Amount (2 features)."""
        self.assertEqual(self.predictor.scaler.n_features_in_, 2)
        if hasattr(self.predictor.scaler, "feature_names_in_"):
            np.testing.assert_array_equal(self.predictor.scaler.feature_names_in_, SCALE_COLS)

    def test_04_single_legitimate_prediction(self):
        """Test inference output format on legitimate transaction."""
        sample = get_sample_transaction(fraud=False)
        results = self.predictor.predict(sample)

        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertIn("prediction", res)
        self.assertIn("label", res)
        self.assertIn("fraud_probability", res)
        self.assertIn("risk_level", res)
        self.assertIn("threshold_used", res)

        self.assertIn(res["prediction"], [0, 1])
        self.assertGreaterEqual(res["fraud_probability"], 0.0)
        self.assertLessEqual(res["fraud_probability"], 1.0)
        self.assertEqual(res["prediction"], 0)
        self.assertEqual(res["label"], "Legitimate")
        self.assertEqual(res["risk_level"], "LOW")

    def test_05_single_fraudulent_prediction(self):
        """Test inference output format on fraudulent transaction."""
        sample = get_sample_transaction(fraud=True)
        results = self.predictor.predict(sample)

        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res["prediction"], 1)
        self.assertEqual(res["label"], "Fraudulent")
        self.assertGreaterEqual(res["fraud_probability"], 0.5)
        self.assertEqual(res["risk_level"], "HIGH")

    def test_06_column_order_independence(self):
        """Verify input with reversed key order yields identical prediction (schema alignment test)."""
        sample = get_sample_transaction(fraud=False)
        reversed_sample = {k: sample[k] for k in reversed(list(sample.keys()))}

        res_normal = self.predictor.predict(sample)[0]
        res_reversed = self.predictor.predict(reversed_sample)[0]

        self.assertEqual(res_normal["prediction"], res_reversed["prediction"])
        self.assertEqual(res_normal["fraud_probability"], res_reversed["fraud_probability"])

    def test_07_missing_feature_validation(self):
        """Verify omitting any feature raises ValueError."""
        sample = get_sample_transaction(fraud=False)
        del sample["V12"]
        with self.assertRaises(ValueError):
            self.predictor.predict(sample)

    def test_08_batch_prediction(self):
        """Test batch processing with list of transactions."""
        batch = [get_sample_transaction(fraud=False), get_sample_transaction(fraud=True)]
        results = self.predictor.predict(batch)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["prediction"], 0)
        self.assertEqual(results[1]["prediction"], 1)

    def test_09_fastapi_health_endpoint(self):
        """Test GET /health status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["scaler_loaded"])
        self.assertEqual(data["expected_features_count"], 30)

    def test_10_fastapi_predict_endpoint(self):
        """Test POST /predict with sample payload."""
        sample = get_sample_transaction(fraud=False)
        response = self.client.post("/predict", json=sample)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["prediction"], 0)
        self.assertEqual(data["label"], "Legitimate")
        self.assertIn("fraud_probability", data)

    def test_11_fastapi_predict_batch_endpoint(self):
        """Test POST /predict/batch."""
        sample_legit = get_sample_transaction(fraud=False)
        sample_fraud = get_sample_transaction(fraud=True)
        payload = {"transactions": [sample_legit, sample_fraud]}
        response = self.client.post("/predict/batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_processed"], 2)
        self.assertEqual(data["fraud_count"], 1)
        self.assertEqual(data["legitimate_count"], 1)


if __name__ == "__main__":
    unittest.main()
