from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def classify_alert(row: dict) -> str:
    if row.get("txn_count_last_10m", 0) >= 8:
        return "velocity_spike"
    if row.get("country_mismatch") and row.get("geo_distance_from_last_km", 0) > 800:
        return "geo_anomaly"
    if row.get("is_card_not_present") and float(row.get("amount", 0)) >= 500:
        return "card_not_present"
    if row.get("is_new_device") and float(row.get("amount", 0)) >= 1000:
        return "new_device_high_value"
    return "high_risk_score"


def alert_message(row: dict) -> str:
    pattern = classify_alert(row)
    if pattern == "velocity_spike":
        return f"Velocity spike: {row['card_id']} has {row['txn_count_last_10m']} txns in 10 min"
    if pattern == "geo_anomaly":
        return f"Geo-anomaly: {row['card_id']} moved {row['geo_distance_from_last_km']} km"
    if pattern == "card_not_present":
        return f"Card-not-present spike: {row['txn_id']} at {row['merchant_name']}"
    if pattern == "new_device_high_value":
        return f"New device high-value payment: {row['txn_id']}"
    return f"High risk score: {row['txn_id']} scored {row['risk_score']}"


class AlertRouter:
    def __init__(self, alert_log_path: Path):
        self.alert_log_path = alert_log_path
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, row: dict) -> dict:
        alert = {
            "alert_id": f"ALERT-{row['txn_id']}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "txn_id": row["txn_id"],
            "card_id": row["card_id"],
            "risk_score": row["risk_score"],
            "pattern": classify_alert(row),
            "message": alert_message(row),
            "severity": "critical" if row["risk_score"] >= 0.85 else "high",
        }
        with self.alert_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(alert) + "\n")
        return alert
