"""Tests for accuracy.persistence — PredictionStore."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from kael_trading_bot.accuracy.models import PredictionRecord
from kael_trading_bot.accuracy.persistence import PredictionStore


def _make_record(
    pair: str = "EURUSD=X",
    timeframe: str = "1d",
    direction: str = "buy",
    predicted_price: float = 1.1050,
    predicted_at: str = "2025-01-01T00:00:00+00:00",
    horizon_at: str = "2025-01-02T00:00:00+00:00",
    **kwargs,
) -> PredictionRecord:
    return PredictionRecord(
        id=uuid.uuid4().hex,
        pair=pair,
        timeframe=timeframe,
        direction=direction,
        predicted_price=predicted_price,
        predicted_at=predicted_at,
        horizon_at=horizon_at,
        model_name=kwargs.get("model_name", "rf"),
        model_version=kwargs.get("model_version", "1.0"),
        generation_ts=kwargs.get("generation_ts", "2025-01-01T00:00:00+00:00"),
    )


class TestPredictionStore:
    """Tests for PredictionStore CRUD operations."""

    def test_save_and_query(self, tmp_path: Path):
        store = PredictionStore(directory=tmp_path)
        rec = _make_record()
        assert store.save(rec) is True

        results = store.query()
        assert len(results) == 1
        assert results[0].pair == "EURUSD=X"
        assert results[0].id == rec.id

    def test_save_duplicate_returns_false(self, tmp_path: Path):
        store = PredictionStore(directory=tmp_path)
        rec = _make_record()
        assert store.save(rec) is True
        assert store.save(rec) is False

        results = store.query()
        assert len(results) == 1

    def test_save_batch(self, tmp_path: Path):
        store = PredictionStore(directory=tmp_path)
        records = [_make_record(pair=f"PAIR{i}") for i in range(5)]
        count = store.save_batch(records)
        assert count == 5
        assert len(store.query()) == 5

    def test_query_filter_by_pair(self, tmp_path: Path):
        store = PredictionStore(directory=tmp_path)
        store.save_batch([
            _make_record(pair="EURUSD=X"),
            _make_record(pair="GBPUSD=X"),
            _make_record(pair="EURUSD=X", predicted_at="2025-02-01T00:00:00+00:00"),
        ])
        results = store.query(pair="EURUSD=X")
        assert len(results) == 2

    def test_query_filter_by_timeframe(self, tmp_path: Path):
        store = PredictionStore(directory=tmp_path)
        store.save_batch([
            _make_record(timeframe="1d"),
            _make_record(timeframe="1h"),
            _make_record(timeframe="1d", predicted_at="2025-02-01T00:00:00+00:00"),
        ])
        results = store.query(timeframe="1d")
        assert len(results) == 2

    def test_query_filter_by_model_name(self, tmp_path: Path):
        store = PredictionStore(directory=tmp_path)
        store.save_batch([
            _make_record(model_name="rf"),
            _make_record(model_name="xgboost", predicted_at="2025-02-01T00:00:00+00:00"),
        ])
        results = store.query(model_name="rf")
        assert len(results) == 1

    def test_query_limit(self, tmp_path: Path):
        store = PredictionStore(directory=tmp_path)
        store.save_batch([_make_record(predicted_at=f"2025-0{i:02d}01T00:00:00+00:00") for i in range(1, 11)])
        results = store.query(limit=3)
        assert len(results) == 3

    def test_query_reverse_chronological(self, tmp_path: Path):
        store = PredictionStore(directory=tmp_path)
        store.save_batch([
            _make_record(generation_ts="2025-01-01T00:00:00+00:00"),
            _make_record(
                generation_ts="2025-01-03T00:00:00+00:00",
                predicted_at="2025-02-01T00:00:00+00:00",
            ),
            _make_record(
                generation_ts="2025-01-02T00:00:00+00:00",
                predicted_at="2025-03-01T00:00:00+00:00",
            ),
        ])
        results = store.query()
        assert results[0].generation_ts == "2025-01-03T00:00:00+00:00"
        assert results[1].generation_ts == "2025-01-02T00:00:00+00:00"
        assert results[2].generation_ts == "2025-01-01T00:00:00+00:00"

    def test_get_by_id(self, tmp_path: Path):
        store = PredictionStore(directory=tmp_path)
        rec = _make_record()
        store.save(rec)

        found = store.get_by_id(rec.id)
        assert found is not None
        assert found.id == rec.id

        not_found = store.get_by_id("nonexistent")
        assert not_found is None

    def test_update_actual_price(self, tmp_path: Path):
        store = PredictionStore(directory=tmp_path)
        rec = _make_record()
        store.save(rec)

        assert store.update_actual_price(rec.id, 1.1100) is True

        updated = store.get_by_id(rec.id)
        assert updated is not None
        assert updated.actual_price == 1.1100

    def test_update_actual_price_nonexistent(self, tmp_path: Path):
        store = PredictionStore(directory=tmp_path)
        assert store.update_actual_price("nonexistent", 1.1100) is False

    def test_clear(self, tmp_path: Path):
        store = PredictionStore(directory=tmp_path)
        store.save_batch([_make_record(), _make_record(predicted_at="2025-02-01T00:00:00+00:00")])
        count = store.clear()
        assert count == 2
        assert store.query() == []

    def test_empty_store(self, tmp_path: Path):
        store = PredictionStore(directory=tmp_path)
        assert store.query() == []
        assert store.get_by_id("abc") is None
        assert store.clear() == 0

    def test_save_missing_fields_skips(self, tmp_path: Path):
        store = PredictionStore(directory=tmp_path)
        bad = PredictionRecord(
            pair="",
            timeframe="1d",
            direction="buy",
            predicted_price=1.0,
            predicted_at="",
            horizon_at="",
        )
        assert store.save(bad) is False
        assert store.query() == []

    def test_creates_directory(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        store = PredictionStore(directory=nested)
        rec = _make_record()
        assert store.save(rec) is True
        assert len(store.query()) == 1

    def test_jsonl_format(self, tmp_path: Path):
        store = PredictionStore(directory=tmp_path)
        rec = _make_record()
        store.save(rec)

        lines = (tmp_path / "predictions.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["pair"] == "EURUSD=X"
        assert data["id"] == rec.id
