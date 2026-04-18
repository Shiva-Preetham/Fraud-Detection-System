from __future__ import annotations

import argparse
import json

from fraud_detection.config import get_settings
from fraud_detection.features import build_feature_rows
from fraud_detection.model import FraudScoringModel
from fraud_detection.synthetic_data import generate_transactions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fraud scoring model.")
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--fraud-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    settings = get_settings()
    transactions = generate_transactions(args.rows, fraud_rate=args.fraud_rate, seed=args.seed)
    feature_rows = build_feature_rows(transactions)
    model = FraudScoringModel.train(feature_rows)
    model.save(settings.model_path)

    metrics_path = settings.model_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(model.metrics, indent=2), encoding="utf-8")
    print(json.dumps({"model_path": str(settings.model_path), "metrics": model.metrics}, indent=2))


if __name__ == "__main__":
    main()
