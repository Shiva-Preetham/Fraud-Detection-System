from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from typing import Iterable


CITIES = [
    ("Mumbai", "IN", 19.0760, 72.8777),
    ("Bengaluru", "IN", 12.9716, 77.5946),
    ("Delhi", "IN", 28.6139, 77.2090),
    ("Hyderabad", "IN", 17.3850, 78.4867),
    ("London", "GB", 51.5072, -0.1276),
    ("Dubai", "AE", 25.2048, 55.2708),
    ("Singapore", "SG", 1.3521, 103.8198),
    ("Bucharest", "RO", 44.4268, 26.1025),
]

MERCHANTS = [
    ("MRC-101", "BigBasket", "grocery"),
    ("MRC-102", "Indigo", "travel"),
    ("MRC-103", "Flipkart", "ecommerce"),
    ("MRC-104", "Swiggy", "food"),
    ("MRC-105", "ATM", "cash"),
    ("MRC-106", "Steam", "digital_goods"),
    ("MRC-107", "HotelDesk", "travel"),
    ("MRC-108", "PayLink", "wallet"),
]

FRAUD_PATTERNS = [
    "velocity_spike",
    "geo_anomaly",
    "card_not_present",
    "new_device_high_value",
]


def _event_time(index: int, start: datetime | None = None) -> datetime:
    base = start or datetime.now(timezone.utc) - timedelta(minutes=30)
    return base + timedelta(seconds=index * 2)


def _normal_amount(rng: random.Random, merchant_category: str) -> float:
    base = {
        "grocery": 42,
        "travel": 220,
        "ecommerce": 75,
        "food": 18,
        "cash": 120,
        "digital_goods": 35,
        "wallet": 50,
    }.get(merchant_category, 60)
    return round(max(2.0, rng.lognormvariate(0, 0.45) * base), 2)


def generate_transaction(
    index: int,
    rng: random.Random,
    fraud_rate: float = 0.03,
    start: datetime | None = None,
) -> dict:
    card_id = f"CARD-{rng.randint(1000, 1099)}"
    merchant_id, merchant_name, category = rng.choice(MERCHANTS)
    city, country, lat, lon = rng.choice(CITIES[:4])
    amount = _normal_amount(rng, category)
    channel = rng.choice(["pos", "atm", "ecommerce", "wallet"])
    card_present = channel in {"pos", "atm"}
    device_id = f"DEV-{rng.randint(1, 160):03d}"
    label = 0
    fraud_pattern = "legit"

    if rng.random() < fraud_rate:
        label = 1
        fraud_pattern = rng.choice(FRAUD_PATTERNS)
        if fraud_pattern == "velocity_spike":
            amount = round(rng.uniform(500, 3500), 2)
            merchant_id, merchant_name, category = rng.choice(MERCHANTS)
        elif fraud_pattern == "geo_anomaly":
            city, country, lat, lon = rng.choice(CITIES[4:])
            amount = round(rng.uniform(700, 5000), 2)
        elif fraud_pattern == "card_not_present":
            channel = "ecommerce"
            card_present = False
            amount = round(rng.uniform(250, 2500), 2)
        elif fraud_pattern == "new_device_high_value":
            device_id = f"DEV-X-{rng.randint(1000, 9999)}"
            amount = round(rng.uniform(1000, 6000), 2)

    return {
        "txn_id": f"TXN-{index:08d}",
        "event_time": _event_time(index, start).isoformat(),
        "card_id": card_id,
        "amount": amount,
        "currency": "INR",
        "merchant_id": merchant_id,
        "merchant_name": merchant_name,
        "merchant_category": category,
        "channel": channel,
        "card_present": card_present,
        "device_id": device_id,
        "ip_address": f"10.{rng.randint(1, 254)}.{rng.randint(1, 254)}.{rng.randint(1, 254)}",
        "city": city,
        "country": country,
        "card_country": "IN",
        "latitude": lat + rng.uniform(-0.04, 0.04),
        "longitude": lon + rng.uniform(-0.04, 0.04),
        "label": label,
        "fraud_pattern": fraud_pattern,
    }


def generate_transactions(rows: int, fraud_rate: float = 0.03, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    start = datetime.now(timezone.utc) - timedelta(seconds=rows * 2)
    return [generate_transaction(i, rng, fraud_rate=fraud_rate, start=start) for i in range(rows)]


def iter_transactions(rate: int, fraud_rate: float = 0.03, seed: int = 7) -> Iterable[dict]:
    rng = random.Random(seed)
    start = datetime.now(timezone.utc)
    index = 0
    while True:
        for _ in range(max(1, rate)):
            yield generate_transaction(index, rng, fraud_rate=fraud_rate, start=start)
            index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic payment transactions as JSON lines.")
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--fraud-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    for transaction in generate_transactions(args.rows, args.fraud_rate, args.seed):
        print(json.dumps(transaction))


if __name__ == "__main__":
    main()
