WITH scored AS (
    SELECT
        txn_id,
        card_id,
        amount,
        prediction,
        label,
        risk_score,
        event_time
    FROM fraud_gold.scored_transactions
    WHERE date(from_iso8601_timestamp(event_time)) >= current_date - interval '7' day
),
confusion AS (
    SELECT
        sum(CASE WHEN prediction = 'FRAUD' AND label = 1 THEN 1 ELSE 0 END) AS tp,
        sum(CASE WHEN prediction = 'FRAUD' AND label = 0 THEN 1 ELSE 0 END) AS fp,
        sum(CASE WHEN prediction = 'LEGIT' AND label = 1 THEN 1 ELSE 0 END) AS fn,
        count(*) AS total_txns,
        sum(CASE WHEN prediction = 'FRAUD' THEN amount ELSE 0 END) AS blocked_value
    FROM scored
)
SELECT
    total_txns,
    blocked_value,
    tp / nullif(tp + fp, 0) AS precision,
    tp / nullif(tp + fn, 0) AS recall,
    2 * (tp / nullif(tp + fp, 0)) * (tp / nullif(tp + fn, 0))
        / nullif((tp / nullif(tp + fp, 0)) + (tp / nullif(tp + fn, 0)), 0) AS f1_score
FROM confusion;
