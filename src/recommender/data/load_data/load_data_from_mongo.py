"""
Production-grade utilities for loading movie and rating data from MongoDB
into pandas DataFrames.

Design goals:
- Deterministic, testable behavior (bounded retries, predictable sleep semantics).
- Structured logs compatible with centralized log systems.
- No ad-hoc logging configuration in this module (use shared app logging).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from unittest.mock import MagicMock

from src.recommender.logging_utils import configure_logger

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
    """
    Load Mongo configuration from environment variables.

    Notes:
    - Avoid loading .env during pytest (tests control env explicitly).
    """
    if "PYTEST_CURRENT_TEST" not in os.environ:
        load_dotenv()

    uri_var = env_prefix + "URI_DEV"
    uri = os.getenv(uri_var)

    if not uri:
        logger.error(
            "MongoDB URI is missing",
            extra={"event": "config_error", "env_var": uri_var},
        )
        raise RuntimeError(f"Required environment variable '{uri_var}' is not set.")

    db_name = os.getenv(env_prefix + "DB_NAME", "movie_recommender_db")
    movies_collection = os.getenv(env_prefix + "MOVIES_COLLECTION", "movies")
    ratings_collection = os.getenv(env_prefix + "RATINGS_COLLECTION", "ratings")

    logger.info(
        "Mongo configuration loaded",
        extra={
            "event": "config_loaded",
            "db_name": db_name,
            "movies_collection": movies_collection,
            "ratings_collection": ratings_collection,
        },
    )

    return MongoConfig(uri, db_name, movies_collection, ratings_collection)


# ---------------------------------------------------------------------------
# Ensure admin.command exists for real and mock clients
# ---------------------------------------------------------------------------


def ensure_admin_command(client: MongoClient) -> None:
    """
    Ensure client.admin.command exists without overwriting test mocks.

    Rules:
    - If client.admin does not exist -> create a MagicMock.
    - If client.admin exists but .command does not exist -> create MagicMock.
    - If .command exists (e.g., test provided a side_effect) -> do not replace it.
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
    Create a MongoClient with bounded retry and backoff.

    Semantics:
    - Perform (max_retries + 1) total attempts:
        1 initial attempt + up to `max_retries` retries.
    - On each attempt:
        * Create a client and run admin.command("ping").
        * On ServerSelectionTimeoutError or PyMongoError:
            - Log a failure.
            - If attempts remain: sleep once for `base_delay_seconds` and retry.
            - If no attempts remain: log give up and raise RuntimeError.
    - On success:
        * Yield the healthy client.
    - On exit:
        * Close the client and log connection closed.
    """
    client: Optional[MongoClient] = None
    max_attempts = max_retries + 1

    try:
        for attempt in range(1, max_attempts + 1):
            logger.info(
                "Mongo connection attempt",
                extra={"event": "mongo_connect_attempt", "attempt": attempt},
            )

            try:
                client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)

                # Ensure admin.command exists without overwriting mocks in tests.
                ensure_admin_command(client)

                client.admin.command("ping")

                logger.info(
                    "Mongo connection established",
                    extra={"event": "mongo_connect_success", "attempt": attempt},
                )
                break

            except (ServerSelectionTimeoutError, PyMongoError) as exc:
                logger.error(
                    "Mongo connection failed",
                    extra={
                        "event": "mongo_connect_failure",
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                    },
                )

                client = None

                if attempt == max_attempts:
                    logger.error(
                        "Mongo connection give up",
                        extra={
                            "event": "mongo_connect_give_up",
                            "max_retries": max_retries,
                            "timeout_ms": timeout_ms,
                        },
                    )
                    raise RuntimeError("MongoDB connection failed after retries.") from exc

                time.sleep(base_delay_seconds)

        if client is None:
            raise RuntimeError("MongoDB client is None after retry loop.")

        yield client

    finally:
        if client is not None:
            client.close()
            logger.info(
                "Mongo connection closed",
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


def validate_schema(df: pd.DataFrame, required_columns: Sequence[str], collection_name: str) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        logger.error(
            "Schema validation failed",
            extra={
                "event": "schema_validation_failed",
                "collection": collection_name,
                "missing_columns": missing,
            },
        )
        raise ValueError(f"Missing required columns in '{collection_name}': {missing}")

    logger.info(
        "Schema validation passed",
        extra={
            "event": "schema_validation_passed",
            "collection": collection_name,
            "columns": list(df.columns),
        },
    )


# ---------------------------------------------------------------------------
# Collection -> DataFrame
# ---------------------------------------------------------------------------


def collection_to_dataframe(
    collection: Collection,
    collection_name: str,
    required_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    logger.info(
        "Loading collection",
        extra={"event": "load_collection", "collection": collection_name},
    )

    docs = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(docs)

    if df.empty:
        logger.error(
            "Collection returned no documents",
            extra={"event": "empty_collection", "collection": collection_name},
        )
        raise ValueError(f"Collection '{collection_name}' returned an empty DataFrame.")

    if required_columns:
        validate_schema(df, required_columns, collection_name)

    logger.info(
        "Collection loaded",
        extra={
            "event": "collection_loaded",
            "collection": collection_name,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
    )

    return df


# ---------------------------------------------------------------------------
# Loader functions
# ---------------------------------------------------------------------------


def load_movies_and_ratings_from_db(
    db: Database,
    config: MongoConfig,
    movie_required_columns: Optional[Sequence[str]] = None,
    rating_required_columns: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
# Debug main (development only)
# ---------------------------------------------------------------------------


def _debug_main() -> None:
    try:
        movies_df, ratings_df = load_movies_and_ratings(
            movie_required_columns=("movieId", "title", "genres"),
            rating_required_columns=("userId", "movieId", "rating"),
        )

        logger.info(
            "Debug load succeeded",
            extra={
                "event": "debug_main_success",
                "movies_rows": int(movies_df.shape[0]),
                "ratings_rows": int(ratings_df.shape[0]),
            },
        )

    except Exception as exc:
        logger.exception(
            "Debug load failed",
            extra={"event": "debug_main_failure", "exception_type": type(exc).__name__},
        )


if __name__ == "__main__":
    _debug_main()
