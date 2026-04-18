from __future__ import annotations

import argparse
import json
import time

from kafka import KafkaProducer

from fraud_detection.config import get_settings
from fraud_detection.synthetic_data import iter_transactions


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce synthetic card transactions to Kafka.")
    parser.add_argument("--rate", type=int, default=10, help="Transactions per second.")
    parser.add_argument("--fraud-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    settings = get_settings()
    producer = KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        linger_ms=10,
    )
    sleep_seconds = 1 / max(1, args.rate)

    for transaction in iter_transactions(args.rate, fraud_rate=args.fraud_rate, seed=args.seed):
        producer.send(settings.kafka_topic, transaction)
        producer.flush()
        print(json.dumps(transaction))
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
