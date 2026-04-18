from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: str) -> Path:
    value = os.getenv(name, default)
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    kafka_topic: str = os.getenv("KAFKA_TOPIC", "transactions")
    risk_threshold: float = float(os.getenv("RISK_THRESHOLD", "0.65"))
    lake_root: Path = _path_from_env("FRAUD_LAKE_ROOT", "data")
    model_path: Path = _path_from_env("FRAUD_MODEL_PATH", "models/fraud_model.joblib")
    alert_log_path: Path = _path_from_env("ALERT_LOG_PATH", "logs/alerts.jsonl")
    aws_region: str = os.getenv("AWS_REGION", "ap-south-1")
    sns_topic_arn: str = os.getenv("SNS_TOPIC_ARN", "")

    @property
    def bronze_path(self) -> Path:
        return self.lake_root / "bronze" / "transactions"

    @property
    def silver_path(self) -> Path:
        return self.lake_root / "silver" / "features"

    @property
    def gold_path(self) -> Path:
        return self.lake_root / "gold" / "scored_transactions"

    @property
    def checkpoint_path(self) -> Path:
        return self.lake_root / "checkpoints" / "spark_stream"


def get_settings() -> Settings:
    settings = Settings()
    for path in [
        settings.bronze_path,
        settings.silver_path,
        settings.gold_path,
        settings.checkpoint_path,
        settings.model_path.parent,
        settings.alert_log_path.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return settings
