import csv
from pathlib import Path
import unittest


class TestMetricsIntegrity(unittest.TestCase):
    def test_metrics_file_exists_and_has_models(self):
        path = Path("docs/assets/model_metrics.csv")
        self.assertTrue(path.exists(), "metrics csv should exist")
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 5)

        required_models = {"Logistic Regression", "Naive Bayes", "Decision Tree", "SVM", "XGBoost"}
        self.assertEqual({r["Model"] for r in rows}, required_models)

        for row in rows:
            for key in ["Accuracy", "F1 Score", "Recall", "Precision"]:
                value = float(row[key])
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
