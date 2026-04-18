from __future__ import annotations

import asyncio
import json
import random
from collections import deque
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
recent_transactions: deque[dict] = deque(maxlen=40)
recent_alerts: deque[dict] = deque(maxlen=20)
event_index = 0


def _next_scored_transaction() -> dict:
    global event_index
    transaction = generate_transaction(event_index, rng, fraud_rate=0.08)
    event_index += 1
    features = engineer.transform(transaction)
    risk_score = model.predict_scores([features])[0]
    scored = {
        **features,
        "risk_score": risk_score,
        "prediction": "FRAUD" if risk_score >= settings.risk_threshold else "LEGIT",
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    recent_transactions.appendleft(scored)
    if risk_score >= settings.risk_threshold:
        recent_alerts.appendleft(alerts.emit(scored))
    return scored


def _summary() -> dict:
    rows = list(recent_transactions)
    flagged = [row for row in rows if row["risk_score"] >= settings.risk_threshold]
    avg_risk = sum(row["risk_score"] for row in rows) / max(1, len(rows))
    return {
        "transactions_today": 184703 + event_index,
        "fraud_flagged": 231 + len(flagged),
        "avg_risk_score": round(avg_risk, 3),
        "threshold": settings.risk_threshold,
        "metrics": model.metrics,
        "recent_transactions": rows[:12],
        "recent_alerts": list(recent_alerts)[:8],
        "heatmap": _heatmap(rows),
    }


def _heatmap(rows: list[dict]) -> list[dict]:
    heatmap = []
    for day in range(7):
        for hour in range(24):
            baseline = 0.05 + ((day + hour) % 5) * 0.015
            observed = [
                row
                for row in rows
                if row.get("event_day_of_week") == day and row.get("event_hour") == hour
            ]
            if observed:
                baseline = sum(row["risk_score"] for row in observed) / len(observed)
            heatmap.append({"day": day, "hour": hour, "risk": round(min(baseline, 0.99), 3)})
    return heatmap


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (frontend_dir / "index.html").read_text(encoding="utf-8")


@app.get("/api/summary")
async def summary() -> dict:
    while len(recent_transactions) < 12:
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
