from __future__ import annotations

import asyncio
import json
import random
import time
from collections import Counter, deque
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from fraud_detection.alerting import AlertRouter
from fraud_detection.config import PROJECT_ROOT, get_settings
from fraud_detection.features import StatefulFeatureEngineer
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
recent_transactions: deque[dict] = deque(maxlen=1500)
recent_alerts: deque[dict] = deque(maxlen=30)
latency_ms: deque[float] = deque(maxlen=300)
event_index = 0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _pct_change(current: int, previous: int) -> float | None:
    if previous <= 0:
        return None
    return ((current - previous) / previous) * 100


def _window_split(rows: list[dict], size: int) -> tuple[list[dict], list[dict]]:
    current = rows[:size]
    previous = rows[size : size * 2]
    return current, previous


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


def _heatmap(rows: list[dict]) -> list[dict]:
    heatmap = []
    for day in range(7):
        for hour in range(24):
            observed = [
                row
                for row in rows
                if row.get("event_day_of_week") == day and row.get("event_hour") == hour
            ]
            avg_risk = sum(row["risk_score"] for row in observed) / max(1, len(observed))
            fraud_count = sum(1 for row in observed if row["prediction"] == "FRAUD")
            heatmap.append(
                {
                    "day": day,
                    "hour": hour,
                    "risk": round(avg_risk, 3),
                    "count": len(observed),
                    "fraud_count": fraud_count,
                }
            )
    return heatmap


def _risk_series(rows: list[dict], size: int = 80) -> list[dict]:
    series = list(reversed(rows[:size]))
    return [
        {
            "index": index,
            "risk_score": row["risk_score"],
            "prediction": row["prediction"],
            "label": row["label"],
        }
        for index, row in enumerate(series)
    ]


def _pattern_breakdown(rows: list[dict]) -> list[dict]:
    counts = Counter(row.get("fraud_pattern", "unknown") for row in rows if row["prediction"] == "FRAUD")
    return [{"pattern": key, "count": value} for key, value in counts.most_common(6)]


def _channel_breakdown(rows: list[dict]) -> list[dict]:
    counts = Counter(row.get("channel", "unknown") for row in rows)
    total = sum(counts.values()) or 1
    return [
        {"channel": key, "count": value, "share": round(value / total, 4)}
        for key, value in counts.most_common()
    ]


def _next_scored_transaction() -> dict:
    global event_index

    transaction = generate_transaction(event_index, rng, fraud_rate=0.08)
    event_index += 1

    started = time.perf_counter()
    features = engineer.transform(transaction)
    risk_score = model.predict_scores([features])[0]
    elapsed_ms = (time.perf_counter() - started) * 1000
    latency_ms.append(elapsed_ms)

    scored = {
        **features,
        "risk_score": risk_score,
        "prediction": "FRAUD" if risk_score >= settings.risk_threshold else "LEGIT",
        "scored_at": _utc_now().isoformat(),
        "latency_ms": round(elapsed_ms, 3),
    }
    recent_transactions.appendleft(scored)
    if risk_score >= settings.risk_threshold:
        recent_alerts.appendleft(alerts.emit(scored))
    return scored


def _summary() -> dict:
    rows = list(recent_transactions)
    current_window, previous_window = _window_split(rows, 60)
    evaluation_window = rows[:240]

    current_count = len(current_window)
    previous_count = len(previous_window)
    flagged = [row for row in current_window if row["prediction"] == "FRAUD"]
    avg_risk = sum(row["risk_score"] for row in current_window) / max(1, current_count)
    metrics = _compute_metrics(evaluation_window)

    return {
        "total_streamed": event_index,
        "window": {
            "label": "Last 60 scored events",
            "transaction_count": current_count,
            "previous_count": previous_count,
            "delta_pct": round(_pct_change(current_count, previous_count), 2)
            if _pct_change(current_count, previous_count) is not None
            else None,
        },
        "fraud": {
            "flagged_count": len(flagged),
            "flagged_rate": round(_safe_ratio(len(flagged), current_count), 4),
            "threshold": settings.risk_threshold,
        },
        "avg_risk_score": round(avg_risk, 3),
        "metrics": metrics,
        "recent_transactions": rows[:12],
        "recent_alerts": list(recent_alerts)[:8],
        "heatmap": _heatmap(rows[:720]),
        "risk_series": _risk_series(rows),
        "pattern_breakdown": _pattern_breakdown(current_window),
        "channel_breakdown": _channel_breakdown(current_window),
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (frontend_dir / "index.html").read_text(encoding="utf-8")


@app.get("/api/summary")
async def summary() -> dict:
    while len(recent_transactions) < 120:
        _next_scored_transaction()
    return _summary()


@app.get("/api/stream")
async def stream() -> StreamingResponse:
    async def events():
        while True:
            scored = _next_scored_transaction()
            payload = {"transaction": scored, "summary": _summary()}
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(1.5)

    return StreamingResponse(events(), media_type="text/event-stream")
