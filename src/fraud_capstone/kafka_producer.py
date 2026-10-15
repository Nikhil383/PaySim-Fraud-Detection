"""Kafka transaction producer and streaming consumer simulation."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from fraud_capstone.config import RAW_DATA_PATH
from fraud_capstone.data import load_raw_data
from fraud_capstone.predict import FraudPredictor

logger = logging.getLogger(__name__)


def run_producer(
    data_path: Path,
    topic: str = "paysim-transactions",
    bootstrap_servers: str = "localhost:9092",
    delay_seconds: float = 0.1,
    limit: Optional[int] = 100,
):
    """Replay transactions from CSV to a Kafka topic."""
    try:
        from kafka import KafkaProducer
    except ImportError:
        raise ImportError("kafka-python is not installed. Install via 'uv sync --extra streaming'.")

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    df = pd.read_csv(data_path, nrows=limit)
    logger.info(f"Streaming {len(df)} transactions to topic '{topic}'...")

    for idx, row in df.iterrows():
        event = row.to_dict()
        producer.send(topic, value=event)
        logger.info(f"Published transaction {idx}: {event.get('nameOrig')} -> {event.get('nameDest')}")
        time.sleep(delay_seconds)

    producer.flush()
    logger.info("Replay finished.")


def run_consumer(
    topic: str = "paysim-transactions",
    bootstrap_servers: str = "localhost:9092",
    group_id: str = "fraud-detector-group",
):
    """Consume transactions, score them with FraudPredictor, and handle malformed events."""
    try:
        from kafka import KafkaConsumer
    except ImportError:
        raise ImportError("kafka-python is not installed. Install via 'uv sync --extra streaming'.")

    predictor = FraudPredictor()
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="earliest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    processed_count = 0
    flagged_count = 0
    failed_count = 0

    logger.info(f"Listening for transactions on '{topic}'...")
    for message in consumer:
        try:
            event = message.value
            processed_count += 1
            result = predictor.predict_single(event)
            if result["is_fraud_prediction"] == 1:
                flagged_count += 1
                logger.warning(
                    f"⚠️ FRAUD DETECTED! Prob: {result['fraud_probability']:.4f} | Event: {event}"
                )
            else:
                logger.info(
                    f"Legitimate transaction | Prob: {result['fraud_probability']:.4f}"
                )
        except Exception as e:
            failed_count += 1
            logger.error(f"Error processing transaction message: {e}")

        logger.info(
            f"Stats -> Processed: {processed_count}, Flagged: {flagged_count}, Failed: {failed_count}"
        )


def main():
    parser = argparse.ArgumentParser(description="Kafka Streaming for PaySim")
    parser.add_argument("--mode", choices=["produce", "consume"], required=True)
    parser.add_argument("--data", type=str, default=str(RAW_DATA_PATH))
    parser.add_argument("--topic", type=str, default="paysim-transactions")
    parser.add_argument("--server", type=str, default="localhost:9092")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    if args.mode == "produce":
        run_producer(Path(args.data), topic=args.topic, bootstrap_servers=args.server, limit=args.limit)
    elif args.mode == "consume":
        run_consumer(topic=args.topic, bootstrap_servers=args.server)


if __name__ == "__main__":
    main()
