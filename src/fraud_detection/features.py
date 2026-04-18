from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Iterable


def parse_time(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class StatefulFeatureEngineer:
    """Keeps card-level memory needed for real-time fraud features."""

    def __init__(self, max_history: int = 500):
        self.max_history = max_history
        self.card_history: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=max_history))
        self.seen_merchants: dict[str, set[str]] = defaultdict(set)
        self.seen_devices: dict[str, set[str]] = defaultdict(set)

    def transform(self, transaction: dict) -> dict:
        event_time = parse_time(transaction["event_time"])
        card_id = transaction["card_id"]
        amount = float(transaction["amount"])
        history = self.card_history[card_id]

        last_hour = [row for row in history if event_time - parse_time(row["event_time"]) <= timedelta(hours=1)]
        last_10m = [row for row in history if event_time - parse_time(row["event_time"]) <= timedelta(minutes=10)]
        last_7d = [row for row in history if event_time - parse_time(row["event_time"]) <= timedelta(days=7)]

        prior_amounts = [float(row["amount"]) for row in history]
        amount_mean = mean(prior_amounts) if prior_amounts else amount
        amount_std = pstdev(prior_amounts) if len(prior_amounts) > 1 else 1.0
        amount_z_score = (amount - amount_mean) / max(amount_std, 1.0)

        prior_location = history[-1] if history else None
        geo_distance = 0.0
        minutes_since_last = 0.0
        if prior_location:
            geo_distance = haversine_km(
                float(prior_location["latitude"]),
                float(prior_location["longitude"]),
                float(transaction["latitude"]),
                float(transaction["longitude"]),
            )
            minutes_since_last = max(
                0.0,
                (event_time - parse_time(prior_location["event_time"])).total_seconds() / 60,
            )

        merchant_id = transaction["merchant_id"]
        device_id = transaction["device_id"]
        features = {
            **transaction,
            "event_hour": event_time.hour,
            "event_day_of_week": event_time.weekday(),
            "is_night": int(event_time.hour < 5 or event_time.hour > 22),
            "is_card_not_present": int(not bool(transaction.get("card_present", False))),
            "country_mismatch": int(transaction.get("country") != transaction.get("card_country")),
            "is_new_merchant": int(merchant_id not in self.seen_merchants[card_id]),
            "is_new_device": int(device_id not in self.seen_devices[card_id]),
            "txn_count_last_10m": len(last_10m),
            "txn_count_last_1h": len(last_hour),
            "distinct_merchants_last_1h": len({row["merchant_id"] for row in last_hour}),
            "spend_sum_last_1h": round(sum(float(row["amount"]) for row in last_hour), 2),
            "avg_amount_last_7d": round(mean([float(row["amount"]) for row in last_7d]), 2) if last_7d else amount,
            "amount_z_score": round(amount_z_score, 4),
            "geo_distance_from_last_km": round(geo_distance, 2),
            "minutes_since_last_txn": round(minutes_since_last, 2),
        }

        self.seen_merchants[card_id].add(merchant_id)
        self.seen_devices[card_id].add(device_id)
        history.append(transaction)
        return features


def build_feature_rows(transactions: Iterable[dict]) -> list[dict]:
    engineer = StatefulFeatureEngineer()
    sorted_transactions = sorted(transactions, key=lambda row: parse_time(row["event_time"]))
    return [engineer.transform(row) for row in sorted_transactions]
