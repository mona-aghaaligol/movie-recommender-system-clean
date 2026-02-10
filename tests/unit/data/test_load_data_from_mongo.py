from __future__ import annotations

import json
import logging
from io import StringIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

import src.recommender.data.load_data.load_data_from_mongo as mod
from src.recommender.data.load_data.load_data_from_mongo import (
    MongoConfig,
    collection_to_dataframe,
    get_collection,
    get_database,
    load_mongo_config,
    load_movies_and_ratings,
    load_movies_and_ratings_from_db,
    mongo_client_with_retry,
    validate_schema,
)

from src.recommender.logging_utils import JsonFormatter, configure_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attach_caplog_to_logger(caplog, logger: logging.Logger) -> None:
    """
    Attach caplog handler to a non-propagating logger.

    Our production logger is configured with propagate=False (correct). Tests that
    assert on logs should attach caplog.handler directly to the logger.
    """
    logger.addHandler(caplog.handler)


def _detach_caplog_from_logger(caplog, logger: logging.Logger) -> None:
    logger.removeHandler(caplog.handler)


def _has_event(caplog, event_name: str) -> bool:
    for rec in caplog.records:
        if getattr(rec, "event", None) == event_name:
            return True
    return False


# ---------------------------------------------------------------------------
# Fake test doubles
# ---------------------------------------------------------------------------

class FakeCollection:
    def __init__(self, documents):
        self._documents = documents
        self.find_calls = []

    def find(self, query, projection):
        cleaned = []
        for doc in self._documents:
            d = dict(doc)
            d.pop("_id", None)
            cleaned.append(d)
        self.find_calls.append((query, projection))
        return cleaned


class FakeDatabase:
    def __init__(self, collections):
        self._collections = collections
        self.getitem_calls = []

    def __getitem__(self, name):
        self.getitem_calls.append(name)
        return self._collections[name]


class FakeClient:
    def __init__(self, db):
        self._db = db
        self.closed = False
        self.admin = MagicMock()

    def __getitem__(self, name):
        return self._db

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# Central JsonFormatter / configure_logger Tests
# ---------------------------------------------------------------------------

def test_json_formatter_outputs_valid_json_and_expected_keys():
    logger = logging.getLogger("json_formatter_test")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())

    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info("hello", extra={"event": "test_event", "user_id": 55})
    payload = json.loads(stream.getvalue())

    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "json_formatter_test"
    assert payload["event"] == "test_event"
    assert payload["user_id"] == 55
    assert "lineno" not in payload


def test_configure_logger_is_idempotent():
    logger_name = "config_test_logger"
    logger = logging.getLogger(logger_name)
    logger.handlers = []

    l1 = configure_logger(logger_name)
    l2 = configure_logger(logger_name)

    assert l1 is l2
    assert len(l1.handlers) == 1
    assert isinstance(l1.handlers[0].formatter, JsonFormatter)
    assert l1.propagate is False


# ---------------------------------------------------------------------------
# load_mongo_config Tests
# ---------------------------------------------------------------------------

def test_load_mongo_config_reads_environment(monkeypatch):
    monkeypatch.setenv("MONGO_URI_DEV", "mongodb://test")
    monkeypatch.setenv("MONGO_DB_NAME", "testdb")
    monkeypatch.setenv("MONGO_MOVIES_COLLECTION", "films")
    monkeypatch.setenv("MONGO_RATINGS_COLLECTION", "scores")

    cfg = load_mongo_config()

    assert cfg.uri == "mongodb://test"
    assert cfg.db_name == "testdb"
    assert cfg.movies_collection == "films"
    assert cfg.ratings_collection == "scores"


def test_load_mongo_config_missing_uri_raises(monkeypatch, caplog):
    monkeypatch.delenv("MONGO_URI_DEV", raising=False)

    caplog.set_level(logging.ERROR)
    _attach_caplog_to_logger(caplog, mod.logger)
    try:
        with pytest.raises(RuntimeError):
            load_mongo_config()

        assert _has_event(caplog, "config_error")
    finally:
        _detach_caplog_from_logger(caplog, mod.logger)


# ---------------------------------------------------------------------------
# get_database / get_collection Tests
# ---------------------------------------------------------------------------

def test_get_database_returns_database_object():
    fake_db = MagicMock()
    client = MagicMock()
    client.__getitem__.return_value = fake_db

    cfg = MongoConfig("x", "testdb", "movies", "ratings")
    db = get_database(client, cfg)

    assert db is fake_db
    client.__getitem__.assert_called_once_with("testdb")


def test_get_collection_retrieves_named_collection():
    fake_coll = MagicMock()
    db = MagicMock()
    db.__getitem__.return_value = fake_coll

    result = get_collection(db, "movies")
    assert result is fake_coll
    db.__getitem__.assert_called_once_with("movies")


# ---------------------------------------------------------------------------
# validate_schema Tests
# ---------------------------------------------------------------------------

def test_validate_schema_success():
    df = pd.DataFrame([{"a": 1, "b": 2}])
    validate_schema(df, ("a", "b"), "test")


def test_validate_schema_failure_logs_and_raises(caplog):
    df = pd.DataFrame([{"a": 1}])

    caplog.set_level(logging.ERROR)
    _attach_caplog_to_logger(caplog, mod.logger)
    try:
        with pytest.raises(ValueError):
            validate_schema(df, ("a", "b"), "test")

        assert _has_event(caplog, "schema_validation_failed")
    finally:
        _detach_caplog_from_logger(caplog, mod.logger)


# ---------------------------------------------------------------------------
# collection_to_dataframe Tests
# ---------------------------------------------------------------------------

def test_collection_to_dataframe_success():
    docs = [
        {"_id": "x1", "movieId": 1, "title": "Toy Story", "genres": "Adventure"},
        {"_id": "x2", "movieId": 2, "title": "Jumanji", "genres": "Adventure"},
    ]
    coll = FakeCollection(docs)

    df = collection_to_dataframe(
        coll,
        "movies",
        required_columns=("movieId", "title", "genres"),
    )

    assert len(df) == 2
    assert set(df.columns) == {"movieId", "title", "genres"}
    assert coll.find_calls == [({}, {"_id": 0})]


def test_collection_to_dataframe_empty_raises(caplog):
    coll = FakeCollection([])

    caplog.set_level(logging.ERROR)
    _attach_caplog_to_logger(caplog, mod.logger)
    try:
        with pytest.raises(ValueError):
            collection_to_dataframe(coll, "movies")

        assert _has_event(caplog, "empty_collection")
    finally:
        _detach_caplog_from_logger(caplog, mod.logger)


# ---------------------------------------------------------------------------
# mongo_client_with_retry Tests
# ---------------------------------------------------------------------------

def _make_mock_client_with_side_effects(side_effects):
    mock_client = MagicMock()
    mock_client.admin = MagicMock()
    mock_client.admin.command.side_effect = side_effects
    return mock_client


def test_mongo_retry_succeeds_on_first_attempt():
    mock_client = _make_mock_client_with_side_effects([{"ok": 1}])

    with patch("src.recommender.data.load_data.load_data_from_mongo.MongoClient", return_value=mock_client), \
         patch("src.recommender.data.load_data.load_data_from_mongo.time.sleep") as mock_sleep:

        with mongo_client_with_retry("mongodb://x") as client:
            assert client is mock_client

        mock_sleep.assert_not_called()
        mock_client.admin.command.assert_called_once_with("ping")


def test_mongo_retry_succeeds_on_second_attempt_with_backoff():
    mock_client = _make_mock_client_with_side_effects([
        ServerSelectionTimeoutError("fail1"),
        {"ok": 1},
    ])

    with patch("src.recommender.data.load_data.load_data_from_mongo.MongoClient", return_value=mock_client), \
         patch("src.recommender.data.load_data.load_data_from_mongo.time.sleep") as mock_sleep:

        with mongo_client_with_retry("mongodb://x", base_delay_seconds=0.5):
            pass

        assert mock_client.admin.command.call_count == 2
        mock_sleep.assert_called_once_with(0.5)


def test_mongo_retry_exhausts_all_attempts_and_raises(caplog):
    mock_client = _make_mock_client_with_side_effects([
        ServerSelectionTimeoutError("f1"),
        PyMongoError("f2"),
        ServerSelectionTimeoutError("f3"),
    ])

    caplog.set_level(logging.ERROR)
    _attach_caplog_to_logger(caplog, mod.logger)
    try:
        with patch("src.recommender.data.load_data.load_data_from_mongo.MongoClient", return_value=mock_client), \
             patch("src.recommender.data.load_data.load_data_from_mongo.time.sleep"):

            with pytest.raises(RuntimeError):
                with mongo_client_with_retry("mongodb://x", max_retries=2):
                    pass

        assert mock_client.admin.command.call_count == 3
        assert _has_event(caplog, "mongo_connect_give_up")
    finally:
        _detach_caplog_from_logger(caplog, mod.logger)


# ---------------------------------------------------------------------------
# load_movies_and_ratings_from_db Tests
# ---------------------------------------------------------------------------

def test_load_movies_and_ratings_from_db_success():
    movies_docs = [
        {"movieId": 1, "title": "Toy Story", "genres": "Animation"},
        {"movieId": 2, "title": "Jumanji", "genres": "Adventure"},
    ]
    ratings_docs = [
        {"userId": 10, "movieId": 1, "rating": 4.5},
        {"userId": 20, "movieId": 2, "rating": 3.0},
    ]

    fake_db = FakeDatabase({
        "movies": FakeCollection(movies_docs),
        "ratings": FakeCollection(ratings_docs),
    })

    cfg = MongoConfig("x", "testdb", "movies", "ratings")

    mdf, rdf = load_movies_and_ratings_from_db(
        fake_db,
        cfg,
        movie_required_columns=("movieId", "title", "genres"),
        rating_required_columns=("userId", "movieId", "rating"),
    )

    assert len(mdf) == 2
    assert len(rdf) == 2
    assert fake_db.getitem_calls == ["movies", "ratings"]


# ---------------------------------------------------------------------------
# load_movies_and_ratings Tests
# ---------------------------------------------------------------------------

def test_load_movies_and_ratings_with_injected_database():
    fake_db = FakeDatabase({
        "movies": FakeCollection([{"movieId": 1, "title": "Toy Story", "genres": "Animation"}]),
        "ratings": FakeCollection([{"userId": 10, "movieId": 1, "rating": 5.0}]),
    })

    cfg = MongoConfig("unused", "db", "movies", "ratings")

    with patch("src.recommender.data.load_data.load_data_from_mongo.mongo_client_with_retry") as mock_ctx:
        mdf, rdf = load_movies_and_ratings(
            config=cfg,
            db=fake_db,
            movie_required_columns=("movieId", "title", "genres"),
            rating_required_columns=("userId", "movieId", "rating"),
        )

    assert len(mdf) == 1
    assert len(rdf) == 1
    mock_ctx.assert_not_called()


def test_load_movies_and_ratings_uses_context_manager_when_no_db_injected():
    fake_db = FakeDatabase({
        "movies": FakeCollection([{"movieId": 1, "title": "Toy Story", "genres": "Animation"}]),
        "ratings": FakeCollection([{"userId": 10, "movieId": 1, "rating": 4.0}]),
    })
    fake_client = FakeClient(fake_db)

    cfg = MongoConfig("mongodb://x", "db", "movies", "ratings")

    with patch("src.recommender.data.load_data.load_data_from_mongo.mongo_client_with_retry") as mock_ctx:
        mock_ctx.return_value.__enter__.return_value = fake_client

        mdf, rdf = load_movies_and_ratings(
            config=cfg,
            movie_required_columns=("movieId", "title", "genres"),
            rating_required_columns=("userId", "movieId", "rating"),
        )

    assert len(mdf) == 1
    assert len(rdf) == 1
    mock_ctx.assert_called_once()


# ---------------------------------------------------------------------------
# _debug_main Tests
# ---------------------------------------------------------------------------

def test_debug_main_emits_success_event(caplog):
    fake_movies = pd.DataFrame([{"movieId": 1, "title": "Toy Story", "genres": "Animation"}])
    fake_ratings = pd.DataFrame([{"userId": 10, "movieId": 1, "rating": 4.0}])

    caplog.set_level(logging.INFO)
    _attach_caplog_to_logger(caplog, mod.logger)
    try:
        with patch.object(mod, "load_movies_and_ratings", return_value=(fake_movies, fake_ratings)):
            mod._debug_main()

        assert _has_event(caplog, "debug_main_success")
    finally:
        _detach_caplog_from_logger(caplog, mod.logger)


def test_debug_main_logs_error_on_failure(caplog):
    caplog.set_level(logging.ERROR)
    _attach_caplog_to_logger(caplog, mod.logger)
    try:
        with patch.object(mod, "load_movies_and_ratings", side_effect=RuntimeError("boom")):
            mod._debug_main()

        assert _has_event(caplog, "debug_main_failure")
    finally:
        _detach_caplog_from_logger(caplog, mod.logger)
