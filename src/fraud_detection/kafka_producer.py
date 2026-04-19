from __future__ import annotations

import argparse
import json
import random
import time

from fraud_detection.config import get_settings
from fraud_detection.kafka_compat import load_kafka
from fraud_detection.olist_data import load_olist_replay


def _streamable_transaction(record: dict) -> dict:
    payload = dict(record)
    payload.pop("risk_score", None)
    payload.pop("prediction", None)
    payload.pop("scored_at", None)
    payload.pop("latency_ms", None)
    payload.pop("pipeline_stage", None)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce Olist transactions to Kafka one row at a time.")
    parser.add_argument("--topic", default="transactions")
    parser.add_argument("--min-delay", type=float, default=0.5)
    parser.add_argument("--max-delay", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows to send. 0 means stream all rows.")
    args = parser.parse_args()

    settings = get_settings()
    records, metadata = load_olist_replay(str(settings.olist_data_dir))
    _, KafkaProducer, _, kafka_import_error = load_kafka()
    if KafkaProducer is None:
        raise RuntimeError(
            "Kafka producer is unavailable because kafka-python could not be imported. "
            f"Details: {kafka_import_error}"
        )
    producer = KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        linger_ms=5,
    )

    print(
        json.dumps(
            {
                "topic": args.topic,
                "rows_available": metadata.get("rows", len(records)),
                "source_dir": str(settings.olist_data_dir),
            }
        )
    )

    sent = 0
    for record in records:
        producer.send(args.topic, _streamable_transaction(record))
        producer.flush()
        sent += 1
        print(json.dumps({"sent": sent, "txn_id": record["txn_id"], "event_time": record["event_time"]}))
        if args.limit and sent >= args.limit:
            break
        time.sleep(random.uniform(args.min_delay, args.max_delay))


if __name__ == "__main__":
    main()
