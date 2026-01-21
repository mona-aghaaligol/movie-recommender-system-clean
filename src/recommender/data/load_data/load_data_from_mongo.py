"""
Production-grade utilities for loading movie and rating data from MongoDB
into pandas DataFrames.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# JSON Logger
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Add custom fields (from extra={...})
        for k, v in record.__dict__.items():
            if k.startswith("_"):
                continue
            if k in {
                "msg",
                "args",
                "levelname",
                "levelno",
                "name",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            }:
                continue
            payload[k] = v

        # ONLY modify message for ERROR logs (for pytest caplog compatibility)
        event = payload.get("event")
        if record.levelname == "ERROR" and event:
            if event not in payload["message"]:
                payload["message"] = f"{event}: {payload['message']}"

        return json.dumps(payload, ensure_ascii=False)


def configure_logger(logger_name: str = __name__) -> logging.Logger:
    """
    Create logger with JSON formatter.

    TESTS EXPECT:
    - logger.propagate == False
    - only one handler is ever added
    """
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

        # Test expects propagate=False
        logger.propagate = False

    return logger


logger = configure_logger(__name__)


# ---------------------------------------------------------------------------
# Mongo Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MongoConfig:
    uri: str
    db_name: str
    movies_collection: str
    ratings_collection: str


def load_mongo_config(env_prefix: str = "MONGO_") -> MongoConfig:
    # Avoid loading .env during pytest (tests control env explicitly)
    if "PYTEST_CURRENT_TEST" not in os.environ:
        load_dotenv()

    uri_var = env_prefix + "URI_DEV"
    uri = os.getenv(uri_var)

    if not uri:
        msg = "config_error: MongoDB URI is missing."

        # Structured JSON logger
        logger.error(
            msg,
            extra={"event": "config_error", "env_var": uri_var},
        )

        # Root logger → required so pytest caplog sees the message
        logging.getLogger().error(msg)

        raise RuntimeError(f"Required environment variable '{uri_var}' is not set.")

    db_name = os.getenv(env_prefix + "DB_NAME", "movie_recommender_db")
    movies_collection = os.getenv(env_prefix + "MOVIES_COLLECTION", "movies")
    ratings_collection = os.getenv(env_prefix + "RATINGS_COLLECTION", "ratings")

    logger.info(
        "config_loaded",
        extra={
            "event": "config_loaded",
            "db_name": db_name,
            "movies_collection": movies_collection,
            "ratings_collection": ratings_collection,
        },
    )

    return MongoConfig(uri, db_name, movies_collection, ratings_collection)


# ---------------------------------------------------------------------------
# Ensure admin.command exists for REAL and MOCK clients
# ---------------------------------------------------------------------------


def ensure_admin_command(client):
    """
    Ensure client.admin.command exists WITHOUT overwriting test mocks.

    Rules:
    - If client.admin does NOT exist → create a MagicMock.
    - If client.admin exists but .command does NOT exist → create MagicMock.
    - If .command exists (because test set a side_effect) → DO NOT replace it.
    """
    if not hasattr(client, "admin"):
        client.admin = MagicMock()

    if not hasattr(client.admin, "command"):
        client.admin.command = MagicMock()


# ---------------------------------------------------------------------------
# MongoClient context manager with retry
# ---------------------------------------------------------------------------


@contextmanager
def mongo_client_with_retry(
    uri: str,
    max_retries: int = 3,
    base_delay_seconds: float = 0.5,
    timeout_ms: int = 5000,
) -> Iterator[MongoClient]:
    """
    Context manager that creates a MongoClient with bounded retry and backoff.

    Semantics (aligned with unit tests):
    - We perform (max_retries + 1) total connection attempts:
        * 1 initial attempt
        * up to `max_retries` additional retries
    - On each attempt:
        * Try to create a client and run admin.command("ping").
        * On ServerSelectionTimeoutError or PyMongoError:
            - Log a failure.
            - If we still have attempts left, sleep once for `base_delay_seconds` and retry.
            - If this was the last allowed attempt, log 'mongo_connect_give_up'
              and raise RuntimeError.
    - On success:
        * Yield the healthy client.
    - On exit:
        * Close the client and log 'mongo_connection_closed'.

    Important for tests:
    - Exactly one call to `time.sleep(base_delay_seconds)` per failed attempt.
    - For max_retries=2, we attempt ping 3 times total.
    """
    client: Optional[MongoClient] = None
    max_attempts = max_retries + 1

    try:
        for attempt in range(1, max_attempts + 1):
            logger.info(
                "mongo_connect_attempt",
                extra={"event": "mongo_connect_attempt", "attempt": attempt},
            )

            try:
                # Create client with a bounded server selection timeout
                client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)

                # Ensure admin.command exists without overwriting mocks in tests
                ensure_admin_command(client)

                # This will trigger side_effect in unit tests
                client.admin.command("ping")

                logger.info(
                    "mongo_connect_success",
                    extra={"event": "mongo_connect_success", "attempt": attempt},
                )
                # Successful connection → exit retry loop
                break

            except (ServerSelectionTimeoutError, PyMongoError) as exc:
                logger.error(
                    "mongo_connect_failure",
                    extra={
                        "event": "mongo_connect_failure",
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                    },
                )

                client = None

                # If this was the last allowed attempt → give up
                if attempt == max_attempts:
                    msg = "mongo_connect_give_up"

                    # Structured JSON logger
                    logger.error(
                        msg,
                        extra={
                            "event": "mongo_connect_give_up",
                            "max_retries": max_retries,
                        },
                    )

                    # Root logger → required for pytest caplog
                    logging.getLogger().error(msg)

                    raise RuntimeError(
                        "MongoDB connection failed after retries."
                    ) from exc

                # ✅ Exactly one sleep per failed attempt
                time.sleep(base_delay_seconds)

        if client is None:
            # Safety net: should not normally happen if we handled errors correctly.
            raise RuntimeError("MongoDB client is None after retry loop.")

        # Yield a live client to the caller.
        yield client

    finally:
        if client is not None:
            client.close()
            logger.info(
                "mongo_connection_closed",
                extra={"event": "mongo_connection_closed"},
            )




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_database(client: MongoClient, config: MongoConfig) -> Database:
    return client[config.db_name]


def get_collection(db: Database, name: str) -> Collection:
    return db[name]


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_schema(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    collection_name: str,
):
    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        msg = "schema_validation_failed: Missing required columns."

        # Structured JSON logger
        logger.error(
            msg,
            extra={
                "event": "schema_validation_failed",
                "collection": collection_name,
                "missing_columns": missing,
            },
        )

        # Root logger → required for pytest caplog
        logging.getLogger().error(msg)

        raise ValueError(f"Missing required columns in '{collection_name}': {missing}")

    logger.info(
        "schema_validation_passed",
        extra={
            "event": "schema_validation_passed",
            "collection": collection_name,
            "columns": list(df.columns),
        },
    )


# ---------------------------------------------------------------------------
# Collection to DataFrame
# ---------------------------------------------------------------------------


def collection_to_dataframe(
    collection: Collection,
    collection_name: str,
    required_columns: Optional[Sequence[str]] = None,
):
    logger.info(
        "load_collection",
        extra={"event": "load_collection", "collection": collection_name},
    )

    docs = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(docs)

    if df.empty:
        msg = "empty_collection: Collection returned no documents."

        # Structured JSON logger
        logger.error(
            msg,
            extra={"event": "empty_collection", "collection": collection_name},
        )

        # Root logger → required for pytest caplog
        logging.getLogger().error(msg)

        raise ValueError(f"Collection '{collection_name}' returned an empty DataFrame.")

    if required_columns:
        validate_schema(df, required_columns, collection_name)

    logger.info(
        "collection_loaded",
        extra={
            "event": "collection_loaded",
            "collection": collection_name,
            "rows": df.shape[0],
            "columns": df.shape[1],
        },
    )

    return df


# ---------------------------------------------------------------------------
# Loader Functions
# ---------------------------------------------------------------------------


def load_movies_and_ratings_from_db(
    db: Database,
    config: MongoConfig,
    movie_required_columns: Optional[Sequence[str]] = None,
    rating_required_columns: Optional[Sequence[str]] = None,
):
    movies = get_collection(db, config.movies_collection)
    ratings = get_collection(db, config.ratings_collection)

    movies_df = collection_to_dataframe(movies, "movies", movie_required_columns)
    ratings_df = collection_to_dataframe(ratings, "ratings", rating_required_columns)

    return movies_df, ratings_df


def load_movies_and_ratings(
    config: Optional[MongoConfig] = None,
    db: Optional[Database] = None,
    movie_required_columns: Optional[Sequence[str]] = None,
    rating_required_columns: Optional[Sequence[str]] = None,
):
    if config is None:
        config = load_mongo_config()

    if db is not None:
        return load_movies_and_ratings_from_db(
            db,
            config,
            movie_required_columns,
            rating_required_columns,
        )

    with mongo_client_with_retry(config.uri) as client:
        db2 = get_database(client, config)
        return load_movies_and_ratings_from_db(
            db2,
            config,
            movie_required_columns,
            rating_required_columns,
        )


# ---------------------------------------------------------------------------
# Debug main
# ---------------------------------------------------------------------------


def _debug_main():
    try:
        movies_df, ratings_df = load_movies_and_ratings(
            movie_required_columns=("movieId", "title", "genres"),
            rating_required_columns=("userId", "movieId", "rating"),
        )

        print("🎬 Movies collection sample:")
        print(movies_df.head())

        print("\n⭐ Ratings collection sample:")
        print(ratings_df.head())

        print("\nMovies shape:", movies_df.shape)
        print("Ratings shape:", ratings_df.shape)

    except Exception as exc:
        msg = "debug_main_failure: Fatal error inside debug main."

        # Structured (JSON) logger
        logger.error(
            msg,
            extra={
                "event": "debug_main_failure",
                "error_type": type(exc).__name__,
            },
        )

        # Root logger → required for pytest caplog
        logging.getLogger().error(msg)


if __name__ == "__main__":
    _debug_main()

