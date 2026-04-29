"""Small utility to print the best performing model from exported benchmark CSV."""

from __future__ import annotations

import csv
from pathlib import Path


def load_metrics(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed: list[dict[str, float | str]] = []
    for row in rows:
        parsed.append(
            {
                "Model": row["Model"],
                "Accuracy": float(row["Accuracy"]),
                "F1 Score": float(row["F1 Score"]),
                "Recall": float(row["Recall"]),
                "Precision": float(row["Precision"]),
            }
        )
    return parsed


def best_model_by_accuracy(metrics: list[dict[str, float | str]]) -> dict[str, float | str]:
    return max(metrics, key=lambda row: float(row["Accuracy"]))


if __name__ == "__main__":
    file_path = Path("docs/assets/model_metrics.csv")
    data = load_metrics(file_path)
    best = best_model_by_accuracy(data)
    print(f"Best model by accuracy: {best['Model']} ({best['Accuracy']:.4f})")
