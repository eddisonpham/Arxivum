"""Unit tests for src.utils — track_activity context manager + run_async."""

from __future__ import annotations

import pytest

from src.db.models import Database
from src.utils import run_async, track_activity


@pytest.fixture
def db() -> Database:
    return Database(":memory:")


class TestTrackActivity:
    """Tests for the track_activity context manager."""

    def test_completed_status_on_success(self, db: Database) -> None:
        with track_activity(db, "search", query="transformers"):
            pass
        rows = db.list_activity(limit=10, action_type="search")
        assert len(rows) == 1
        assert rows[0].action_type == "search"
        assert rows[0].status == "completed"
        assert rows[0].query == "transformers"

    def test_failed_status_on_exception(self, db: Database) -> None:
        with pytest.raises(ValueError, match="boom"):
            with track_activity(db, "summarize", arxiv_id="2106.00001"):
                raise ValueError("boom")
        rows = db.list_activity(limit=10, action_type="summarize")
        assert len(rows) == 1
        assert rows[0].status == "failed"
        assert rows[0].arxiv_id == "2106.00001"

    def test_metadata_json_stored(self, db: Database) -> None:
        with track_activity(
            db, "idea", arxiv_id="2106.00001",
            metadata_json={"num_ideas": 3, "focus_area": "methodological"},
        ):
            pass
        rows = db.list_activity(limit=10, action_type="idea")
        assert rows[0].metadata_json == {
            "num_ideas": 3,
            "focus_area": "methodological",
        }

    def test_exception_type_preserved(self, db: Database) -> None:
        class CustomError(Exception):
            pass

        with pytest.raises(CustomError):
            with track_activity(db, "query", query="test"):
                raise CustomError("custom")

    def test_arxiv_id_only(self, db: Database) -> None:
        with track_activity(db, "enrich", arxiv_id="1234.5678"):
            pass
        rows = db.list_activity(limit=10, action_type="enrich")
        assert rows[0].arxiv_id == "1234.5678"
        assert rows[0].status == "completed"

    def test_no_query_field_when_not_provided(self, db: Database) -> None:
        with track_activity(db, "remove", arxiv_id="0001.0001"):
            pass
        rows = db.list_activity(limit=10, action_type="remove")
        assert rows[0].query is None


class TestRunAsync:
    """Tests for the run_async helper."""

    def test_run_async_no_running_loop(self) -> None:
        async def _coro() -> str:
            return "result"

        assert run_async(_coro()) == "result"
