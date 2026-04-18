from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import BooleanType, DoubleType, IntegerType, StringType, StructField, StructType

from fraud_detection.alerting import AlertRouter
from fraud_detection.config import get_settings
from fraud_detection.features import build_feature_rows
from fraud_detection.model import load_model_or_heuristic


schema = StructType(
    [
        StructField("txn_id", StringType()),
        StructField("event_time", StringType()),
        StructField("card_id", StringType()),
        StructField("amount", DoubleType()),
        StructField("currency", StringType()),
        StructField("merchant_id", StringType()),
        StructField("merchant_name", StringType()),
        StructField("merchant_category", StringType()),
        StructField("channel", StringType()),
        StructField("card_present", BooleanType()),
        StructField("device_id", StringType()),
        StructField("ip_address", StringType()),
        StructField("city", StringType()),
        StructField("country", StringType()),
        StructField("card_country", StringType()),
        StructField("latitude", DoubleType()),
        StructField("longitude", DoubleType()),
        StructField("label", IntegerType()),
        StructField("fraud_pattern", StringType()),
    ]
)


def main() -> None:
    settings = get_settings()
    spark = (
        SparkSession.builder.appName("fraud-detection-stream")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    model = load_model_or_heuristic(settings.model_path)
    alerts = AlertRouter(settings.alert_log_path)

    kafka_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap_servers)
        .option("subscribe", settings.kafka_topic)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed_df = kafka_df.select(from_json(col("value").cast("string"), schema).alias("event")).select("event.*")

    def score_microbatch(batch_df, batch_id: int) -> None:
        if batch_df.rdd.isEmpty():
            return

        settings.bronze_path.mkdir(parents=True, exist_ok=True)
        settings.silver_path.mkdir(parents=True, exist_ok=True)
        settings.gold_path.mkdir(parents=True, exist_ok=True)
        batch_df.write.mode("append").parquet(str(settings.bronze_path))

        transactions = [row.asDict(recursive=True) for row in batch_df.collect()]
        feature_rows = build_feature_rows(transactions)
        scores = model.predict_scores(feature_rows)
        scored_rows = []
        for row, score in zip(feature_rows, scores):
            scored = {
                **row,
                "risk_score": float(score),
                "prediction": "FRAUD" if score >= settings.risk_threshold else "LEGIT",
                "batch_id": batch_id,
            }
            scored_rows.append(scored)
            if score >= settings.risk_threshold:
                alerts.emit(scored)

        spark.createDataFrame(feature_rows).write.mode("append").parquet(str(settings.silver_path))
        spark.createDataFrame(scored_rows).write.mode("append").parquet(str(settings.gold_path))

    query = (
        parsed_df.writeStream.foreachBatch(score_microbatch)
        .option("checkpointLocation", str(settings.checkpoint_path))
        .trigger(processingTime="5 seconds")
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()
