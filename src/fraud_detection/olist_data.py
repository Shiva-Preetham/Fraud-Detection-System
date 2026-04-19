from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from fraud_detection.features import haversine_km


def _mode_or_first(series: pd.Series):
    clean = series.dropna()
    if clean.empty:
        return None
    mode = clean.mode()
    return mode.iloc[0] if not mode.empty else clean.iloc[0]


def _safe_haversine(lat1, lon1, lat2, lon2) -> float:
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0.0
    return round(haversine_km(float(lat1), float(lon1), float(lat2), float(lon2)), 2)


def _load_csvs(base_dir: Path) -> dict[str, pd.DataFrame]:
    orders = pd.read_csv(
        base_dir / "olist_orders_dataset.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    payments = pd.read_csv(base_dir / "olist_order_payments_dataset.csv")
    items = pd.read_csv(base_dir / "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
    reviews = pd.read_csv(
        base_dir / "olist_order_reviews_dataset.csv",
        parse_dates=["review_creation_date", "review_answer_timestamp"],
    )
    products = pd.read_csv(base_dir / "olist_products_dataset.csv")
    sellers = pd.read_csv(base_dir / "olist_sellers_dataset.csv")
    translations = pd.read_csv(base_dir / "product_category_name_translation.csv")
    customers = pd.read_csv(base_dir / "olist_customers_dataset.csv")
    geolocation = pd.read_csv(base_dir / "olist_geolocation_dataset.csv")
    return {
        "orders": orders,
        "payments": payments,
        "items": items,
        "reviews": reviews,
        "products": products,
        "sellers": sellers,
        "translations": translations,
        "customers": customers,
        "geolocation": geolocation,
    }


def _prepare_order_frame(base_dir: Path) -> pd.DataFrame:
    frames = _load_csvs(base_dir)
    geo = (
        frames["geolocation"]
        .groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(
            geolocation_lat=("geolocation_lat", "mean"),
            geolocation_lng=("geolocation_lng", "mean"),
            geolocation_city=("geolocation_city", _mode_or_first),
            geolocation_state=("geolocation_state", _mode_or_first),
        )
    )

    customers = frames["customers"].merge(
        geo,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    )
    customers = customers.rename(
        columns={
            "geolocation_lat": "customer_lat",
            "geolocation_lng": "customer_lng",
        }
    )

    sellers = frames["sellers"].merge(
        geo,
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    )
    sellers = sellers.rename(columns={"geolocation_lat": "seller_lat", "geolocation_lng": "seller_lng"})

    products = frames["products"].merge(frames["translations"], on="product_category_name", how="left")
    products["product_category_name_english"] = products["product_category_name_english"].fillna(
        products["product_category_name"].fillna("unknown")
    )

    payment_agg = (
        frames["payments"]
        .groupby("order_id", as_index=False)
        .agg(
            payment_value=("payment_value", "sum"),
            payment_type=("payment_type", _mode_or_first),
            payment_installments=("payment_installments", "max"),
            payment_sequential=("payment_sequential", "max"),
        )
    )

    review_agg = (
        frames["reviews"]
        .groupby("order_id", as_index=False)
        .agg(
            review_score=("review_score", "mean"),
            review_comment_count=("review_comment_message", lambda s: int(s.notna().sum())),
        )
    )

    item_enriched = (
        frames["items"]
        .merge(products[["product_id", "product_category_name_english"]], on="product_id", how="left")
        .merge(
            sellers[["seller_id", "seller_state", "seller_city", "seller_lat", "seller_lng"]],
            on="seller_id",
            how="left",
        )
    )
    item_enriched["product_category_name_english"] = item_enriched["product_category_name_english"].fillna("unknown")

    item_agg = (
        item_enriched.groupby("order_id", as_index=False)
        .agg(
            price=("price", "sum"),
            freight_value=("freight_value", "sum"),
            item_count=("order_item_id", "count"),
            seller_count=("seller_id", "nunique"),
            primary_seller_id=("seller_id", _mode_or_first),
            seller_state=("seller_state", _mode_or_first),
            seller_city=("seller_city", _mode_or_first),
            seller_lat=("seller_lat", "mean"),
            seller_lng=("seller_lng", "mean"),
            product_category=("product_category_name_english", _mode_or_first),
        )
    )

    frame = (
        frames["orders"]
        .merge(customers, on="customer_id", how="left")
        .merge(payment_agg, on="order_id", how="left")
        .merge(item_agg, on="order_id", how="left")
        .merge(review_agg, on="order_id", how="left")
    )

    frame = frame.dropna(subset=["order_purchase_timestamp", "payment_value"]).copy()
    frame["event_time"] = frame["order_purchase_timestamp"]
    frame["approval_delay_hours"] = (
        frame["order_approved_at"] - frame["order_purchase_timestamp"]
    ).dt.total_seconds().div(3600)
    frame["delivery_delay_days"] = (
        frame["order_delivered_customer_date"] - frame["order_estimated_delivery_date"]
    ).dt.total_seconds().div(86400)
    frame["freight_ratio"] = (frame["freight_value"] / frame["price"].replace(0, pd.NA)).fillna(0)
    frame["seller_distance_km"] = frame.apply(
        lambda row: _safe_haversine(row["customer_lat"], row["customer_lng"], row["seller_lat"], row["seller_lng"]),
        axis=1,
    )
    frame["review_score"] = frame["review_score"].fillna(5.0)
    frame["payment_type"] = frame["payment_type"].fillna("unknown")
    frame["product_category"] = frame["product_category"].fillna("unknown")
    frame["seller_city"] = frame["seller_city"].fillna("unknown")
    frame["seller_state"] = frame["seller_state"].fillna("unknown")
    frame["customer_state"] = frame["customer_state"].fillna("unknown")
    frame["customer_city"] = frame["customer_city"].fillna("unknown")
    frame["price"] = frame["price"].fillna(frame["payment_value"])
    frame["freight_value"] = frame["freight_value"].fillna(0)
    frame["item_count"] = frame["item_count"].fillna(1)
    frame["seller_count"] = frame["seller_count"].fillna(1)
    frame["payment_installments"] = frame["payment_installments"].fillna(1)
    frame["payment_sequential"] = frame["payment_sequential"].fillna(1)
    frame = frame.sort_values("event_time").reset_index(drop=True)
    return frame


def _derive_proxy_reasons(frame: pd.DataFrame) -> pd.DataFrame:
    high_value = frame["payment_value"].quantile(0.9)
    ultra_value = frame["payment_value"].quantile(0.98)
    high_distance = frame["seller_distance_km"].quantile(0.95)
    high_freight_ratio = frame["freight_ratio"].quantile(0.9)
    severe_delay = max(5.0, frame["delivery_delay_days"].dropna().quantile(0.97))

    frame["proxy_reason"] = "normal"
    frame["proxy_label"] = 0

    order_counts_24h: list[int] = []
    order_counts_30d: list[int] = []
    spend_30d: list[float] = []
    gap_hours: list[float] = []
    is_new_seller: list[int] = []
    is_new_payment_type: list[int] = []
    txn_count_10m: list[int] = []
    txn_count_1h: list[int] = []
    distinct_sellers_1h: list[int] = []
    spend_1h: list[float] = []
    avg_amount_7d: list[float] = []
    amount_z_scores: list[float] = []
    geo_from_last: list[float] = []
    seen_sellers: dict[str, set[str]] = {}
    seen_channels: dict[str, set[str]] = {}
    customer_history: dict[str, list[dict]] = {}

    for row in frame.itertuples(index=False):
        customer = row.customer_unique_id
        history = customer_history.setdefault(customer, [])
        seen_seller_set = seen_sellers.setdefault(customer, set())
        seen_channel_set = seen_channels.setdefault(customer, set())

        last_10m = [
            h for h in history if (row.event_time - h["event_time"]).total_seconds() <= 600
        ]
        last_1h = [
            h for h in history if (row.event_time - h["event_time"]).total_seconds() <= 3600
        ]
        last_24h = [
            h for h in history if (row.event_time - h["event_time"]).total_seconds() <= 86400
        ]
        last_30d = [
            h for h in history if (row.event_time - h["event_time"]).total_seconds() <= 30 * 86400
        ]
        last_7d = [
            h for h in history if (row.event_time - h["event_time"]).total_seconds() <= 7 * 86400
        ]
        amounts = [h["amount"] for h in history]
        amount_mean = sum(amounts) / len(amounts) if amounts else row.payment_value
        amount_std = (
            (sum((value - amount_mean) ** 2 for value in amounts) / len(amounts)) ** 0.5
            if len(amounts) > 1
            else 1.0
        )
        previous = history[-1] if history else None

        order_counts_24h.append(len(last_24h))
        order_counts_30d.append(len(last_30d))
        spend_30d.append(round(sum(h["amount"] for h in last_30d), 2))
        gap_hours.append(
            round((row.event_time - previous["event_time"]).total_seconds() / 3600, 2) if previous else 0.0
        )
        is_new_seller.append(int(str(row.primary_seller_id) not in seen_seller_set))
        is_new_payment_type.append(int(str(row.payment_type) not in seen_channel_set))
        txn_count_10m.append(len(last_10m))
        txn_count_1h.append(len(last_1h))
        distinct_sellers_1h.append(len({h["seller_id"] for h in last_1h}))
        spend_1h.append(round(sum(h["amount"] for h in last_1h), 2))
        avg_amount_7d.append(round(sum(h["amount"] for h in last_7d) / len(last_7d), 2) if last_7d else row.payment_value)
        amount_z_scores.append(round((row.payment_value - amount_mean) / max(amount_std, 1.0), 4))
        geo_from_last.append(
            _safe_haversine(previous["seller_lat"], previous["seller_lng"], row.seller_lat, row.seller_lng)
            if previous
            else 0.0
        )

        seen_seller_set.add(str(row.primary_seller_id))
        seen_channel_set.add(str(row.payment_type))
        history.append(
            {
                "event_time": row.event_time,
                "amount": float(row.payment_value),
                "seller_id": str(row.primary_seller_id),
                "seller_lat": row.seller_lat,
                "seller_lng": row.seller_lng,
            }
        )

    frame["customer_orders_24h"] = order_counts_24h
    frame["customer_orders_30d"] = order_counts_30d
    frame["customer_spend_30d"] = spend_30d
    frame["gap_hours_since_prev"] = gap_hours
    frame["txn_count_last_10m"] = txn_count_10m
    frame["txn_count_last_1h"] = txn_count_1h
    frame["distinct_merchants_last_1h"] = distinct_sellers_1h
    frame["spend_sum_last_1h"] = spend_1h
    frame["avg_amount_last_7d"] = avg_amount_7d
    frame["amount_z_score"] = amount_z_scores
    frame["geo_distance_from_last_km"] = geo_from_last
    frame["is_new_merchant"] = is_new_seller
    frame["is_new_device"] = is_new_payment_type
    frame["minutes_since_last_txn"] = (frame["gap_hours_since_prev"] * 60).round(2)

    conditions = [
        (
            frame["customer_orders_24h"].ge(3) & frame["payment_value"].ge(high_value),
            "velocity_spike",
        ),
        (
            frame["payment_installments"].ge(8)
            & frame["payment_value"].ge(ultra_value)
            & frame["payment_sequential"].gt(1),
            "installment_abuse",
        ),
        (
            frame["delivery_delay_days"].fillna(-999).ge(severe_delay)
            & frame["review_score"].le(2)
            & frame["payment_value"].ge(high_value),
            "chargeback_risk",
        ),
        (
            frame["seller_distance_km"].ge(high_distance)
            & frame["freight_ratio"].ge(high_freight_ratio)
            & frame["payment_value"].ge(high_value),
            "geo_freight_anomaly",
        ),
        (
            frame["order_status"].isin(["canceled", "unavailable"]) & frame["payment_value"].ge(high_value),
            "order_failure_risk",
        ),
    ]

    for mask, reason in conditions:
        frame.loc[mask & (frame["proxy_label"] == 0), "proxy_reason"] = reason
        frame.loc[mask, "proxy_label"] = 1

    return frame


def _score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    high_value = max(frame["payment_value"].quantile(0.99), 1.0)
    high_distance = max(frame["seller_distance_km"].quantile(0.99), 1.0)
    high_spend_30d = max(frame["customer_spend_30d"].quantile(0.95), 1.0)

    scores = []
    for row in frame.itertuples(index=False):
        score = 0.06
        score += min(float(row.payment_value) / high_value, 1.0) * 0.22
        score += min(float(row.customer_orders_24h) / 4, 1.0) * 0.17
        score += min(float(row.payment_installments) / 12, 1.0) * 0.09
        score += 0.09 if row.payment_sequential > 1 else 0
        score += 0.1 if row.review_score <= 2 else 0
        score += min(max(float(row.delivery_delay_days or 0), 0) / 18, 1.0) * 0.08
        score += min(float(row.seller_distance_km) / high_distance, 1.0) * 0.07
        score += min(float(row.freight_ratio), 0.7) / 0.7 * 0.07
        score += min(max(float(row.amount_z_score), 0), 5) / 5 * 0.07
        score += min(float(row.customer_spend_30d) / high_spend_30d, 1.0) * 0.04
        score += 0.08 if row.order_status in {"canceled", "unavailable"} else 0
        score += 0.04 if row.proxy_reason == "velocity_spike" else 0
        scores.append(round(min(score, 0.995), 4))

    frame["risk_score"] = scores
    return frame


def _to_dashboard_records(frame: pd.DataFrame) -> tuple[list[dict], dict]:
    frame = frame.sort_values("event_time").reset_index(drop=True)
    records: list[dict] = []
    for row in frame.itertuples(index=False):
        records.append(
            {
                "txn_id": row.order_id,
                "event_time": row.event_time.isoformat(),
                "card_id": row.customer_unique_id,
                "amount": float(row.payment_value),
                "currency": "BRL",
                "merchant_id": str(row.primary_seller_id),
                "merchant_name": str(row.seller_city),
                "merchant_category": str(row.product_category),
                "channel": str(row.payment_type),
                "card_present": False,
                "device_id": str(row.payment_type),
                "ip_address": "olist-historical-replay",
                "city": str(row.customer_city),
                "country": "BR",
                "card_country": "BR",
                "latitude": float(row.customer_lat) if pd.notna(row.customer_lat) else 0.0,
                "longitude": float(row.customer_lng) if pd.notna(row.customer_lng) else 0.0,
                "label": int(row.proxy_label),
                "fraud_pattern": str(row.proxy_reason),
                "event_hour": int(row.event_time.hour),
                "event_day_of_week": int(row.event_time.weekday()),
                "is_night": int(row.event_time.hour < 5 or row.event_time.hour > 22),
                "is_card_not_present": 1,
                "country_mismatch": 0,
                "is_new_merchant": int(row.is_new_merchant),
                "is_new_device": int(row.is_new_device),
                "txn_count_last_10m": int(row.txn_count_last_10m),
                "txn_count_last_1h": int(row.txn_count_last_1h),
                "distinct_merchants_last_1h": int(row.distinct_merchants_last_1h),
                "spend_sum_last_1h": float(row.spend_sum_last_1h),
                "avg_amount_last_7d": float(row.avg_amount_last_7d),
                "amount_z_score": float(row.amount_z_score),
                "geo_distance_from_last_km": float(row.geo_distance_from_last_km),
                "minutes_since_last_txn": float(row.minutes_since_last_txn),
                "payment_installments": int(row.payment_installments),
                "payment_sequential": int(row.payment_sequential),
                "order_item_count": int(row.item_count),
                "seller_count": int(row.seller_count),
                "review_score": float(row.review_score),
                "delivery_delay_days": float(row.delivery_delay_days) if pd.notna(row.delivery_delay_days) else 0.0,
                "approval_delay_hours": float(row.approval_delay_hours) if pd.notna(row.approval_delay_hours) else 0.0,
                "freight_ratio": float(row.freight_ratio),
                "customer_orders_24h": int(row.customer_orders_24h),
                "customer_orders_30d": int(row.customer_orders_30d),
                "customer_spend_30d": float(row.customer_spend_30d),
                "seller_distance_km": float(row.seller_distance_km),
                "order_status": str(row.order_status),
                "proxy_reason": str(row.proxy_reason),
                "risk_score": float(row.risk_score),
                "prediction": "FRAUD" if row.risk_score >= 0.65 else "LEGIT",
            }
        )

    positives = int(frame["proxy_label"].sum())
    metadata = {
        "dataset_name": "Olist ecommerce orders",
        "source_dir": str(frame.attrs["source_dir"]),
        "rows": int(len(frame)),
        "proxy_positive_rows": positives,
        "proxy_positive_rate": round(positives / max(1, len(frame)), 4),
        "time_range": {
            "start": frame["event_time"].min().isoformat(),
            "end": frame["event_time"].max().isoformat(),
        },
        "label_note": "Proxy risk labels derived from weak supervision rules because Olist has no native fraud ground truth.",
    }
    return records, metadata


@lru_cache(maxsize=2)
def load_olist_replay(base_dir_str: str) -> tuple[list[dict], dict]:
    base_dir = Path(base_dir_str)
    frame = _prepare_order_frame(base_dir)
    frame.attrs["source_dir"] = str(base_dir)
    frame = _derive_proxy_reasons(frame)
    frame = _score_frame(frame)
    return _to_dashboard_records(frame)
