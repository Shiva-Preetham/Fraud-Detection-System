from __future__ import annotations

import sys

import six


def bootstrap_kafka_vendor_six() -> None:
    """Patch kafka-python's vendored six import path for Python 3.12 runtimes."""
    sys.modules.setdefault("kafka.vendor.six", six)
    sys.modules.setdefault("kafka.vendor.six.moves", six.moves)


def load_kafka() -> tuple[object | None, object | None, object | None, str | None]:
    bootstrap_kafka_vendor_six()
    try:
        from kafka import KafkaConsumer, KafkaProducer  # type: ignore
        from kafka.errors import KafkaError, NoBrokersAvailable  # type: ignore

        return (
            KafkaConsumer,
            KafkaProducer,
            (KafkaError, NoBrokersAvailable),
            None,
        )
    except Exception as exc:  # pragma: no cover - environment dependent import issue
        return None, None, None, str(exc)
