from fraud_detection.model import HeuristicScorer


def test_heuristic_scorer_raises_risk_for_contextual_fraud_signals():
    scorer = HeuristicScorer()
    low_risk = {
        "amount": 20,
        "country_mismatch": 0,
        "is_card_not_present": 0,
        "is_new_device": 0,
        "is_new_merchant": 0,
        "txn_count_last_10m": 0,
        "distinct_merchants_last_1h": 0,
        "geo_distance_from_last_km": 0,
        "amount_z_score": 0,
    }
    high_risk = {
        **low_risk,
        "amount": 4500,
        "country_mismatch": 1,
        "is_card_not_present": 1,
        "is_new_device": 1,
        "txn_count_last_10m": 14,
        "geo_distance_from_last_km": 7500,
        "amount_z_score": 8,
    }

    low_score, high_score = scorer.predict_scores([low_risk, high_risk])

    assert low_score < 0.2
    assert high_score > 0.65
