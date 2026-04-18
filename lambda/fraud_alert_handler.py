from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import boto3


def _format_message(record: dict) -> str:
    return (
        f"{record['prediction']} alert for {record['txn_id']} | "
        f"card={record['card_id']} | score={record['risk_score']} | "
        f"amount={record['amount']} | merchant={record['merchant_name']}"
    )


def lambda_handler(event, context):
    sns_topic_arn = os.getenv("SNS_TOPIC_ARN", "")
    sns = boto3.client("sns") if sns_topic_arn else None
    published = []

    for record in event.get("Records", []):
        body = record.get("body") or record.get("Sns", {}).get("Message", "{}")
        payload = json.loads(body)
        if float(payload.get("risk_score", 0)) < float(os.getenv("RISK_THRESHOLD", "0.65")):
            continue

        message = _format_message(payload)
        alert = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "payload": payload,
        }
        if sns:
            sns.publish(TopicArn=sns_topic_arn, Subject="High-risk fraud alert", Message=json.dumps(alert))
        published.append(alert)

    return {"statusCode": 200, "body": json.dumps({"published": len(published)})}
