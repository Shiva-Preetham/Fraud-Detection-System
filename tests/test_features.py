from fraud_detection.features import StatefulFeatureEngineer, haversine_km
from fraud_detection.synthetic_data import generate_transactions


def test_haversine_detects_long_distance():
    distance = haversine_km(19.0760, 72.8777, 51.5072, -0.1276)
    assert distance > 7000


def test_feature_engineer_adds_velocity_and_context():
    transactions = generate_transactions(8, fraud_rate=0, seed=12)
    card_id = transactions[0]["card_id"]
    for row in transactions:
        row["card_id"] = card_id

    engineer = StatefulFeatureEngineer()
    rows = [engineer.transform(row) for row in transactions]

    assert rows[0]["txn_count_last_10m"] == 0
    assert rows[-1]["txn_count_last_10m"] == 7
    assert "amount_z_score" in rows[-1]
    assert "geo_distance_from_last_km" in rows[-1]
