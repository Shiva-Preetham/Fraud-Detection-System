from __future__ import annotations

import asyncio
import json
import random
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from fraud_detection.alerting import AlertRouter
from fraud_detection.config import PROJECT_ROOT, get_settings
from fraud_detection.features import StatefulFeatureEngineer
from fraud_detection.kafka_compat import load_kafka
from fraud_detection.model import load_model_or_heuristic
from fraud_detection.synthetic_data import generate_transaction


settings = get_settings()
app = FastAPI(title="Fraud Detection Ops Center")
frontend_dir = PROJECT_ROOT / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

rng = random.Random(42)
engineer = StatefulFeatureEngineer()
model = load_model_or_heuristic(settings.model_path)
alerts = AlertRouter(settings.alert_log_path)
verified_transactions: deque[dict] = deque(maxlen=3000)
pending_buffer: deque[dict] = deque(maxlen=10)
recent_alerts: deque[dict] = deque(maxlen=40)
latency_ms: deque[float] = deque(maxlen=500)
event_index = 0
consumer_thread: threading.Thread | None = None
consumer_stop = threading.Event()
state_lock = threading.Lock()
consumer_status = {
    "connected": False,
    "started": False,
    "last_error": "Kafka consumer not started yet.",
}
dataset_info = {
    "dataset_name": "Kafka transactions stream",
    "label_note": "Transactions are ingested from Kafka one by one and scored in the backend consumer.",
}
KafkaConsumer, _, kafka_errors, kafka_import_error = load_kafka()
KafkaError: type[Exception] = kafka_errors[0] if kafka_errors else Exception
NoBrokersAvailable: type[Exception] = kafka_errors[1] if kafka_errors else Exception


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _compute_metrics(rows: list[dict]) -> dict:
    tp = sum(1 for row in rows if row["prediction"] == "FRAUD" and row["label"] == 1)
    fp = sum(1 for row in rows if row["prediction"] == "FRAUD" and row["label"] == 0)
    tn = sum(1 for row in rows if row["prediction"] == "LEGIT" and row["label"] == 0)
    fn = sum(1 for row in rows if row["prediction"] == "LEGIT" and row["label"] == 1)

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)

    auc_roc = None
    labels = [row["label"] for row in rows]
    if rows and len(set(labels)) > 1:
        try:
            from sklearn.metrics import roc_auc_score

            auc_roc = float(roc_auc_score(labels, [row["risk_score"] for row in rows]))
        except Exception:
            auc_roc = None

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc_roc": round(auc_roc, 4) if auc_roc is not None else None,
        "avg_latency_ms": round(sum(latency_ms) / max(1, len(latency_ms)), 2),
        "threshold": settings.risk_threshold,
        "evaluated_rows": len(rows),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def _process_transaction(transaction: dict) -> dict:
    started = time.perf_counter()
    scored = dict(transaction)
    if "risk_score" not in scored:
        scored["risk_score"] = model.predict_scores([scored])[0]
    scored["prediction"] = "FRAUD" if scored["risk_score"] >= settings.risk_threshold else "LEGIT"
    scored["scored_at"] = _utc_now().isoformat()
    elapsed_ms = (time.perf_counter() - started) * 1000
    scored["latency_ms"] = round(elapsed_ms, 3)
    latency_ms.append(elapsed_ms)
    return scored


def _ingest_scored_transaction(scored: dict) -> dict:
    global event_index

    event_index += 1
    scored["pipeline_stage"] = "buffered"
    pending_buffer.appendleft(scored)
    if len(pending_buffer) > 10:
        verified = pending_buffer.pop()
        verified["pipeline_stage"] = "verified"
        verified_transactions.appendleft(verified)
        if verified["prediction"] == "FRAUD":
            recent_alerts.appendleft(_make_alert(verified))
    return scored


def _consume_kafka_messages() -> None:
    consumer_status["started"] = True
    if KafkaConsumer is None:
        consumer_status["connected"] = False
        consumer_status["last_error"] = (
            "Kafka client import failed. "
            f"Install or repair kafka-python before starting the stream. Details: {kafka_import_error}"
        )
        return

    while not consumer_stop.is_set():
        try:
            consumer = KafkaConsumer(
                settings.kafka_topic,
                bootstrap_servers=settings.kafka_bootstrap_servers,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                consumer_timeout_ms=1000,
                value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
            )
            consumer_status["connected"] = True
            consumer_status["last_error"] = ""

            while not consumer_stop.is_set():
                message_batch = consumer.poll(timeout_ms=1000, max_records=1)
                if not message_batch:
                    continue
                for _, records in message_batch.items():
                    for record in records:
                        scored = _process_transaction(record.value)
                        with state_lock:
                            _ingest_scored_transaction(scored)

            consumer.close()
        except NoBrokersAvailable:
            consumer_status["connected"] = False
            consumer_status["last_error"] = (
                f"No Kafka broker available at {settings.kafka_bootstrap_servers}. "
                "Start Kafka/Redpanda and the producer to stream transactions."
            )
            time.sleep(2)
        except KafkaError as exc:
            consumer_status["connected"] = False
            consumer_status["last_error"] = f"Kafka error: {exc}"
            time.sleep(2)
        except Exception as exc:  # keep the consumer alive for local retries
            consumer_status["connected"] = False
            consumer_status["last_error"] = f"Consumer crashed: {exc}"
            time.sleep(2)


def _heatmap(rows: list[dict]) -> list[dict]:
    heatmap = []
    for day in range(7):
        for hour in range(24):
            observed = [
                row
                for row in rows
                if row.get("event_day_of_week") == day and row.get("event_hour") == hour
            ]
            count = len(observed)
            avg_risk = sum(row["risk_score"] for row in observed) / max(1, count)
            fraud_count = sum(1 for row in observed if row["label"] == 1)
            fraud_rate = fraud_count / max(1, count)
            heatmap.append(
                {
                    "day": day,
                    "hour": hour,
                    "risk": round(avg_risk, 3),
                    "count": count,
                    "fraud_count": fraud_count,
                    "fraud_rate": round(fraud_rate, 3),
                }
            )
    return heatmap


def _risk_series(rows: list[dict], size: int = 60) -> list[dict]:
    series = list(reversed(rows[:size]))
    return [
        {
            "index": index,
            "risk_score": row["risk_score"],
            "prediction": row["prediction"],
            "label": row["label"],
            "amount": row["amount"],
        }
        for index, row in enumerate(series)
    ]


def _pattern_breakdown(rows: list[dict]) -> list[dict]:
    counts = Counter(row.get("fraud_pattern", "unknown") for row in rows if row["label"] == 1)
    return [{"pattern": key, "count": value} for key, value in counts.most_common(8)]


def _channel_breakdown(rows: list[dict]) -> list[dict]:
    counts = Counter(row.get("channel", "unknown") for row in rows)
    total = sum(counts.values()) or 1
    return [
        {"channel": key, "count": value, "share": round(value / total, 4)}
        for key, value in counts.most_common()
    ]


def _make_alert(row: dict) -> dict:
    pattern = row.get("fraud_pattern", "risk_case")
    messages = {
        "velocity_spike": f"Velocity anomaly: {row['card_id']} placed {row['customer_orders_24h']} orders in 24h",
        "installment_abuse": f"Installment risk: {row['txn_id']} used {row['payment_installments']} installments",
        "chargeback_risk": f"Chargeback proxy: {row['txn_id']} review score {row['review_score']}",
        "geo_freight_anomaly": f"Geo-freight anomaly: {row['txn_id']} spans {row['seller_distance_km']:.0f} km",
        "order_failure_risk": f"Order failure risk: {row['txn_id']} status {row['order_status']}",
    }
    alert_row = {**row, "risk_score": row["risk_score"]}
    fallback = alerts.emit(alert_row)
    return {
        "alert_id": fallback["alert_id"],
        "created_at": fallback["created_at"],
        "txn_id": row["txn_id"],
        "severity": "critical" if row["risk_score"] >= 0.82 else "high",
        "message": messages.get(pattern, fallback["message"]),
    }


def _next_from_synthetic() -> dict:
    transaction = generate_transaction(event_index, rng, fraud_rate=0.08)
    features = engineer.transform(transaction)
    scored = _process_transaction(
        {
        **features,
        "payment_installments": 1,
        "payment_sequential": 1,
        "order_item_count": 1,
        "seller_count": 1,
        "review_score": 5.0,
        "delivery_delay_days": 0.0,
        "approval_delay_hours": 0.0,
        "freight_ratio": 0.0,
        "customer_orders_24h": 0,
        "customer_orders_30d": 0,
        "customer_spend_30d": 0.0,
        "seller_distance_km": 0.0,
        "order_status": "delivered",
        "proxy_reason": transaction.get("fraud_pattern", "normal"),
        }
    )
    return _ingest_scored_transaction(scored)


def _summary() -> dict:
    rows = list(verified_transactions)
    current_window = rows[:60]
    evaluation_window = rows[:240]
    full_reference = rows[:720]
    feed_rows = list(verified_transactions)[:10] + list(pending_buffer)[:10]
    feed_rows = sorted(feed_rows, key=lambda row: row["scored_at"], reverse=True)[:14]

    flagged = [row for row in current_window if row["prediction"] == "FRAUD"]
    avg_risk = sum(row["risk_score"] for row in current_window) / max(1, len(current_window))
    metrics = _compute_metrics(evaluation_window)

    return {
        "total_streamed": event_index,
        "window": {
            "label": "Current analytics window",
            "transaction_count": len(current_window),
        },
        "fraud": {
            "flagged_count": len(flagged),
            "flagged_rate": round(_safe_ratio(len(flagged), len(current_window)), 4),
            "threshold": settings.risk_threshold,
        },
        "avg_risk_score": round(avg_risk, 3),
        "metrics": metrics,
        "recent_transactions": feed_rows,
        "recent_alerts": list(recent_alerts)[:8],
        "heatmap": _heatmap(full_reference),
        "risk_series": _risk_series(rows),
        "pattern_breakdown": _pattern_breakdown(current_window),
        "channel_breakdown": _channel_breakdown(current_window),
        "dataset_info": dataset_info,
        "pipeline": {
            "replay_mode": "Kafka -> consumer -> ML score one by one -> buffer 10 -> verified dashboard feed",
            "buffer_size": 10,
            "pending_count": len(pending_buffer),
            "verified_count": len(verified_transactions),
            "kafka_enabled": consumer_status["connected"],
            "spark_enabled": False,
            "aws_enabled": False,
            "aws_note": "AWS modules exist in the repo as architecture scaffolding, but the current Kafka pipeline runs locally without AWS services.",
            "consumer_error": consumer_status["last_error"],
        },
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (frontend_dir / "index.html").read_text(encoding="utf-8")


@app.on_event("startup")
async def startup_event() -> None:
    global consumer_thread
    consumer_stop.clear()
    if consumer_thread is None or not consumer_thread.is_alive():
        consumer_thread = threading.Thread(target=_consume_kafka_messages, daemon=True)
        consumer_thread.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    consumer_stop.set()


@app.get("/api/summary")
async def summary() -> dict:
    with state_lock:
        return _summary()


@app.get("/api/stream")
async def stream() -> StreamingResponse:
    async def events():
        while True:
            with state_lock:
                latest = pending_buffer[0] if pending_buffer else (verified_transactions[0] if verified_transactions else None)
                payload = {"transaction": latest, "summary": _summary()}
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(events(), media_type="text/event-stream")
