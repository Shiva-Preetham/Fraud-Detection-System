SELECT
    day_of_week(from_iso8601_timestamp(event_time)) AS day_of_week,
    hour(from_iso8601_timestamp(event_time)) AS event_hour,
    count(*) AS transaction_count,
    avg(risk_score) AS avg_risk_score,
    sum(CASE WHEN prediction = 'FRAUD' THEN 1 ELSE 0 END) AS fraud_alerts
FROM fraud_gold.scored_transactions
WHERE date(from_iso8601_timestamp(event_time)) >= current_date - interval '7' day
GROUP BY 1, 2
ORDER BY 1, 2;
