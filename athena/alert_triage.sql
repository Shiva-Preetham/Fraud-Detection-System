SELECT
    txn_id,
    card_id,
    merchant_name,
    merchant_category,
    amount,
    city,
    country,
    risk_score,
    fraud_pattern,
    txn_count_last_10m,
    geo_distance_from_last_km,
    is_card_not_present,
    country_mismatch
FROM fraud_gold.scored_transactions
WHERE prediction = 'FRAUD'
ORDER BY risk_score DESC, from_iso8601_timestamp(event_time) DESC
LIMIT 100;
