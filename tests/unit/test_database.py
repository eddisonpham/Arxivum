"""Unit tests for the SQLite database layer."""

from src.db.models import (
    ActivityRow, Database, IdeaRow, MetricsRow,
    NoveltyCheckRow, PaperRow, SummaryRow,
)


class TestDatabaseCRUD:
    def test_create_tables(self, db):
        """All tables should exist after init."""
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {r[0] for r in tables}
        assert "papers" in names
        assert "metrics" in names
        assert "summaries" in names
        assert "ideas" in names
        assert "novelty_checks" in names
        assert "activity_log" in names

    def test_upsert_and_get_paper(self, db):
        p = PaperRow(
            arxiv_id="2106.00001", title="Test Paper",
            authors=["Alice", "Bob"], abstract="An abstract.",
            primary_category="cs.LG",
        )
        db.upsert_paper(p)
        got = db.get_paper("2106.00001")
        assert got is not None
        assert got.title == "Test Paper"
        assert got.authors == ["Alice", "Bob"]
        assert got.abstract == "An abstract."
        assert got.primary_category == "cs.LG"

    def test_upsert_paper_updates_existing(self, db):
        p = PaperRow(arxiv_id="2106.00001", title="V1", authors=["A"], abstract="abs")
        db.upsert_paper(p)
        p2 = PaperRow(arxiv_id="2106.00001", title="V2", authors=["A"], abstract="abs2")
        db.upsert_paper(p2)
        got = db.get_paper("2106.00001")
        assert got.title == "V2"
        assert got.abstract == "abs2"

    def test_get_nonexistent_paper(self, db):
        assert db.get_paper("9999.99999") is None

    def test_delete_paper_cascades(self, db):
        p = PaperRow(arxiv_id="2106.00001", title="T", authors=["A"], abstract="abs")
        db.upsert_paper(p)
        db.upsert_metrics(MetricsRow(arxiv_id="2106.00001", citation_count=5))
        db.upsert_summary(SummaryRow(id=None, arxiv_id="2106.00001", section="overall", content="test"))
        assert db.get_paper("2106.00001") is not None
        deleted = db.delete_paper("2106.00001")
        assert deleted is True
        assert db.get_paper("2106.00001") is None
        assert db.get_metrics("2106.00001") is None
        assert db.get_summaries("2106.00001") == []

    def test_delete_nonexistent_paper(self, db):
        assert db.delete_paper("9999.99999") is False

    def test_count_papers(self, db):
        assert db.count_papers() == 0
        db.upsert_paper(PaperRow(arxiv_id="1.1", title="A", authors=["X"], abstract="a"))
        db.upsert_paper(PaperRow(arxiv_id="2.2", title="B", authors=["Y"], abstract="b"))
        assert db.count_papers() == 2

    def test_list_papers_pagination(self, db):
        for i in range(5):
            db.upsert_paper(PaperRow(
                arxiv_id=f"2106.{i:05d}", title=f"Paper {i}",
                authors=["A"], abstract=f"abstract {i}",
            ))
        page1 = db.list_papers(limit=2, offset=0)
        page2 = db.list_papers(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        ids1 = {p.arxiv_id for p in page1}
        ids2 = {p.arxiv_id for p in page2}
        assert ids1.isdisjoint(ids2)


class TestMetrics:
    def test_upsert_and_get_metrics(self, db):
        db.upsert_paper(PaperRow(arxiv_id="2106.00001", title="T", authors=["A"], abstract="a"))
        db.upsert_metrics(MetricsRow(
            arxiv_id="2106.00001", citation_count=42,
            influential_citation_count=5, venue="NeurIPS",
            s2_paper_id="abc",
        ))
        m = db.get_metrics("2106.00001")
        assert m is not None
        assert m.citation_count == 42
        assert m.venue == "NeurIPS"
        assert m.s2_paper_id == "abc"

    def test_upsert_metrics_updates(self, db):
        db.upsert_paper(PaperRow(arxiv_id="2106.00001", title="T", authors=["A"], abstract="a"))
        db.upsert_metrics(MetricsRow(arxiv_id="2106.00001", citation_count=10))
        db.upsert_metrics(MetricsRow(arxiv_id="2106.00001", citation_count=99, venue="ICML"))
        m = db.get_metrics("2106.00001")
        assert m.citation_count == 99
        assert m.venue == "ICML"

    def test_get_metrics_nonexistent(self, db):
        assert db.get_metrics("9999.99999") is None


class TestSummaries:
    def test_upsert_and_get_summary(self, db):
        db.upsert_paper(PaperRow(arxiv_id="2106.00001", title="T", authors=["A"], abstract="a"))
        sid = db.upsert_summary(SummaryRow(
            id=None, arxiv_id="2106.00001", section="overall", content="A summary.",
            model_used="stub",
        ))
        assert sid > 0
        s = db.get_summary("2106.00001", "overall")
        assert s is not None
        assert s.content == "A summary."
        assert s.model_used == "stub"

    def test_upsert_summary_updates_existing(self, db):
        db.upsert_paper(PaperRow(arxiv_id="2106.00001", title="T", authors=["A"], abstract="a"))
        db.upsert_summary(SummaryRow(id=None, arxiv_id="2106.00001", section="overall", content="v1"))
        db.upsert_summary(SummaryRow(id=None, arxiv_id="2106.00001", section="overall", content="v2"))
        s = db.get_summary("2106.00001", "overall")
        assert s.content == "v2"

    def test_get_summaries_multiple(self, db):
        db.upsert_paper(PaperRow(arxiv_id="2106.00001", title="T", authors=["A"], abstract="a"))
        for sec in ["problem_statement", "methodology", "overall"]:
            db.upsert_summary(SummaryRow(id=None, arxiv_id="2106.00001", section=sec, content=f"content {sec}"))
        summaries = db.get_summaries("2106.00001")
        assert len(summaries) == 3
        sections = {s.section for s in summaries}
        assert sections == {"problem_statement", "methodology", "overall"}


class TestIdeas:
    def test_add_and_get_idea(self, db):
        db.upsert_paper(PaperRow(arxiv_id="2106.00001", title="T", authors=["A"], abstract="a"))
        iid = db.add_idea(IdeaRow(
            id=None, arxiv_id="2106.00001", idea_text="A novel idea",
            constraints_used={"assumptions": ["x"]}, status="pending",
            search_queries=["query1", "query2"],
        ))
        assert iid > 0
        idea = db.get_idea(iid)
        assert idea is not None
        assert idea.idea_text == "A novel idea"
        assert idea.status == "pending"
        assert idea.search_queries == ["query1", "query2"]
        assert idea.constraints_used == {"assumptions": ["x"]}

    def test_list_ideas(self, db):
        db.upsert_paper(PaperRow(arxiv_id="2106.00001", title="T", authors=["A"], abstract="a"))
        for i in range(3):
            db.add_idea(IdeaRow(id=None, arxiv_id="2106.00001", idea_text=f"idea {i}"))
        ideas = db.list_ideas("2106.00001")
        assert len(ideas) == 3

    def test_update_idea_status(self, db):
        db.upsert_paper(PaperRow(arxiv_id="2106.00001", title="T", authors=["A"], abstract="a"))
        iid = db.add_idea(IdeaRow(id=None, arxiv_id="2106.00001", idea_text="idea"))
        assert db.update_idea_status(iid, "approved") is True
        idea = db.get_idea(iid)
        assert idea.status == "approved"

    def test_update_nonexistent_idea(self, db):
        assert db.update_idea_status(999, "approved") is False


class TestNoveltyChecks:
    def test_add_and_get_novelty(self, db):
        db.upsert_paper(PaperRow(arxiv_id="2106.00001", title="T", authors=["A"], abstract="a"))
        iid = db.add_idea(IdeaRow(id=None, arxiv_id="2106.00001", idea_text="idea"))
        nid = db.add_novelty_check(NoveltyCheckRow(
            id=None, idea_id=iid, query_terms=["q1"],
            similar_arxiv_ids=["2106.00002"], verdict="needs_review",
            notes="Some overlap found",
        ))
        assert nid > 0
        checks = db.get_novelty_checks(iid)
        assert len(checks) == 1
        assert checks[0].verdict == "needs_review"
        assert checks[0].similar_arxiv_ids == ["2106.00002"]


class TestActivityLog:
    def test_log_activity(self, db):
        lid = db.log_activity(ActivityRow(
            id=None, action_type="search", query="test query", status="started",
        ))
        assert lid > 0

    def test_update_activity_status(self, db):
        lid = db.log_activity(ActivityRow(id=None, action_type="search", status="started"))
        db.update_activity_status(lid, "completed")
        activities = db.list_activity(limit=10)
        assert len(activities) == 1
        assert activities[0].status == "completed"

    def test_list_activity_limit(self, db):
        for i in range(10):
            db.log_activity(ActivityRow(id=None, action_type="search", status="completed"))
        activities = db.list_activity(limit=5)
        assert len(activities) == 5

    def test_list_activity_filter_by_type(self, db):
        db.log_activity(ActivityRow(id=None, action_type="search", status="completed"))
        db.log_activity(ActivityRow(id=None, action_type="import", status="completed"))
        searches = db.list_activity(limit=10, action_type="search")
        assert len(searches) == 1
        assert searches[0].action_type == "search"
