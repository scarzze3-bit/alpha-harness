"""Tests for Phase 5 Task D: Persistence and Recovery.

Verifies:
1. Experiment records and simulation records survive application restarts.
2. Interrupted RUNNING jobs are safely resumed and stale PENDING jobs are orphaned on restart.
3. Orphan reconciliation correctly adopts matching alphas and populates dedup cache.
4. Orphan reconciliation requeues lost sends and abandons after MAX_ORPHAN_REQUEUES.
5. Repeated processing of the same completion event is idempotent.
6. Cancellation races preserve uncancelled simulations by reverting to RUNNING.
7. Transient API failures (429, 5xx) yield retry outcomes, never COMPLETE or ERROR.
8. Failed simulations/errors are TrialState.FAIL and not recorded as valid low-quality alphas.
"""

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent))
# No stubs needed; real models are imported

from sqlalchemy import select

from alpha_harness.brain.schemas import SimulationStatus
from alpha_harness.db.models import (
    DedupEntry,
    SimStatus,
    SimulationRecord,
    Study,
    StudyStatus,
    Trial,
    TrialState,
    utcnow,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.engine.lifecycle import (
    Outcome,
    cancel_after_lost_race,
    finish,
    read_outcome,
)
from alpha_harness.engine.reconcile import (
    MAX_ORPHAN_REQUEUES,
    WINDOW,
    reconcile_orphans,
)
from alpha_harness.engine.tracker import (
    PENDING_GRACE_SECONDS,
    SimulationTracker,
)
from alpha_harness.labs.scheduler import _search_memory


def _make_sim_record(
    request_hash: str,
    status: SimStatus,
    *,
    task: str = "test",
    expression: str = "rank(close)",
    platform_id: str | None = None,
    alpha_id: str | None = None,
    orphan_requeues: int = 0,
    created_at: datetime | None = None,
    submitted_at: datetime | None = None,
    sent_at: datetime | None = None,
    payload: dict[str, Any] | None = None,
) -> SimulationRecord:
    now = utcnow()
    wire = payload or {
        "type": "REGULAR",
        "regular": expression,
        "settings": {"region": "USA", "delay": 1},
    }
    return SimulationRecord(
        request_hash=request_hash,
        payload=wire,
        expression=expression,
        sim_type="REGULAR",
        instrument_type="EQUITY",
        region="USA",
        delay=1,
        language="FASTEXPR",
        universe="TOP3000",
        task=task,
        status=status,
        platform_id=platform_id,
        alpha_id=alpha_id,
        orphan_requeues=orphan_requeues,
        created_at=created_at or now,
        submitted_at=submitted_at,
        sent_at=sent_at,
    )


class TestPersistenceRecovery(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_harness.db"
        self.db = Database(self.db_path)
        await self.db.create_all()

    async def asyncTearDown(self) -> None:
        await self.db.dispose()
        self.temp_dir.cleanup()

    async def test_records_survive_application_restart(self) -> None:
        """Verify that Studies, Trials, and SimulationRecords survive application restart."""
        now = utcnow()
        async with self.db.session() as session:
            study = Study(
                name="alpha-search-v1",
                template_source="def alpha(): pass",
                task="study",
                status=StudyStatus.RUNNING,
                max_trials=80,
            )
            session.add(study)
            await session.flush()

            t1 = Trial(
                study_id=study.id,
                number=0,
                state=TrialState.COMPLETE,
                expression="rank(close)",
                values=[1.85],
                result={"sharpe": 1.85, "turnover": 0.12},
                alpha_id="alpha-101",
            )
            t2 = Trial(
                study_id=study.id,
                number=1,
                state=TrialState.FAIL,
                expression="rank(invalid_field)",
                values=None,
                message="BrainError: Invalid field",
            )
            session.add_all([t1, t2])

            r1 = _make_sim_record(
                request_hash="hash-running-1",
                status=SimStatus.RUNNING,
                platform_id="sim-brain-001",
                expression="rank(close)",
                created_at=now,
                submitted_at=now,
            )
            r2 = _make_sim_record(
                request_hash="hash-pending-stale",
                status=SimStatus.PENDING,
                expression="rank(volume)",
                created_at=now - timedelta(seconds=PENDING_GRACE_SECONDS + 60),
            )
            session.add_all([r1, r2])

        # Simulate complete restart by disposing and reopening the DB file
        await self.db.dispose()
        db_restarted = Database(self.db_path)

        async with db_restarted.session() as session:
            loaded_study = await session.scalar(
                select(Study).where(Study.name == "alpha-search-v1")
            )
            self.assertIsNotNone(loaded_study)
            self.assertEqual(loaded_study.status, StudyStatus.RUNNING)
            self.assertEqual(loaded_study.max_trials, 80)

            trials = list(
                await session.scalars(
                    select(Trial).where(Trial.study_id == loaded_study.id).order_by(Trial.number)
                )
            )
            self.assertEqual(len(trials), 2)
            self.assertEqual(trials[0].state, TrialState.COMPLETE)
            self.assertEqual(trials[0].values, [1.85])
            self.assertEqual(trials[0].alpha_id, "alpha-101")
            self.assertEqual(trials[1].state, TrialState.FAIL)
            self.assertIsNone(trials[1].values)
            self.assertEqual(trials[1].message, "BrainError: Invalid field")

            sims = list(
                await session.scalars(select(SimulationRecord).order_by(SimulationRecord.id))
            )
            self.assertEqual(len(sims), 2)
            self.assertEqual(sims[0].status, SimStatus.RUNNING)
            self.assertEqual(sims[0].platform_id, "sim-brain-001")
            self.assertEqual(sims[1].status, SimStatus.PENDING)

        await db_restarted.dispose()

    async def test_interrupted_jobs_reconciled_on_startup(self) -> None:
        """Verify tracker.reconcile resumes RUNNING and marks stale PENDING as ORPHANED."""
        now = utcnow()
        async with self.db.session() as session:
            r_running = _make_sim_record(
                request_hash="hash-run",
                status=SimStatus.RUNNING,
                platform_id="sim-live-99",
                created_at=now,
                submitted_at=now,
            )
            r_stale = _make_sim_record(
                request_hash="hash-stale",
                status=SimStatus.PENDING,
                created_at=now - timedelta(seconds=PENDING_GRACE_SECONDS + 30),
            )
            session.add_all([r_running, r_stale])

        mock_endpoints = MagicMock()
        tracker = SimulationTracker(
            self.db,
            mock_endpoints,
            on_change=AsyncMock(),
            on_unauthorized=AsyncMock(),
            on_alpha=AsyncMock(),
        )

        counts = await tracker.reconcile()
        self.assertEqual(counts["resumed"], 1)
        self.assertEqual(counts["orphaned"], 1)

        async with self.db.session() as session:
            running_rec = await session.scalar(
                select(SimulationRecord).where(SimulationRecord.request_hash == "hash-run")
            )
            stale_rec = await session.scalar(
                select(SimulationRecord).where(SimulationRecord.request_hash == "hash-stale")
            )
            self.assertIsNotNone(running_rec)
            self.assertEqual(running_rec.status, SimStatus.RUNNING)
            self.assertIsNotNone(stale_rec)
            self.assertEqual(stale_rec.status, SimStatus.ORPHANED)
            self.assertIn("Matching it against your alphas", stale_rec.message or "")

    async def test_orphan_reconcile_adopts_matching_alpha(self) -> None:
        """Verify reconcile_orphans adopts an alpha when BRAIN completed the lost send."""
        now = utcnow()
        payload = {
            "type": "REGULAR",
            "regular": "rank(close) - rank(open)",
            "settings": {"region": "USA", "delay": 1},
        }
        async with self.db.session() as session:
            orphan = _make_sim_record(
                request_hash="hash-orphan-adopt",
                status=SimStatus.ORPHANED,
                payload=payload,
                sent_at=now - timedelta(minutes=5),
                created_at=now - timedelta(minutes=5),
            )
            session.add(orphan)

        created_str = (now - timedelta(minutes=4)).isoformat()
        mock_endpoints = MagicMock()
        mock_endpoints.list_alphas = AsyncMock(
            return_value={
                "results": [
                    {
                        "id": "alpha-recovered-123",
                        "type": "REGULAR",
                        "regular": "rank(close) - rank(open)\n",
                        "settings": {"region": "USA", "delay": 1},
                        "dateCreated": created_str,
                    }
                ]
            }
        )

        landed_alphas: list[str] = []
        counts = await reconcile_orphans(
            self.db,
            mock_endpoints,
            on_alpha=lambda a_id: landed_alphas.append(a_id),
        )

        self.assertEqual(counts.get("adopted", 0), 1)
        self.assertEqual(landed_alphas, ["alpha-recovered-123"])

        async with self.db.session() as session:
            rec = await session.scalar(
                select(SimulationRecord).where(SimulationRecord.request_hash == "hash-orphan-adopt")
            )
            self.assertIsNotNone(rec)
            self.assertEqual(rec.status, SimStatus.COMPLETE)
            self.assertEqual(rec.alpha_id, "alpha-recovered-123")
            self.assertEqual(rec.progress, 1.0)

            # Check that dedup cache was populated
            dedup = await session.scalar(
                select(DedupEntry).where(DedupEntry.request_hash == "hash-orphan-adopt")
            )
            self.assertIsNotNone(dedup)
            self.assertEqual(dedup.alpha_id, "alpha-recovered-123")

    async def test_orphan_reconcile_requeues_and_eventually_abandons(self) -> None:
        """Verify lost sends are requeued until MAX_ORPHAN_REQUEUES, then abandoned to ERROR."""
        now = utcnow()
        past_window = now - WINDOW - timedelta(minutes=5)
        async with self.db.session() as session:
            orphan_requeue = _make_sim_record(
                request_hash="hash-orphan-rq",
                status=SimStatus.ORPHANED,
                orphan_requeues=0,
                sent_at=past_window,
                created_at=past_window,
            )
            orphan_abandon = _make_sim_record(
                request_hash="hash-orphan-ab",
                status=SimStatus.ORPHANED,
                orphan_requeues=MAX_ORPHAN_REQUEUES,
                sent_at=past_window,
                created_at=past_window,
            )
            session.add_all([orphan_requeue, orphan_abandon])

        mock_endpoints = MagicMock()
        mock_endpoints.list_alphas = AsyncMock(return_value={"results": []})

        counts = await reconcile_orphans(
            self.db,
            mock_endpoints,
            on_alpha=lambda _: None,
        )

        self.assertEqual(counts.get("requeued", 0), 1)
        self.assertEqual(counts.get("abandoned", 0), 1)

        async with self.db.session() as session:
            rq_rec = await session.scalar(
                select(SimulationRecord).where(SimulationRecord.request_hash == "hash-orphan-rq")
            )
            ab_rec = await session.scalar(
                select(SimulationRecord).where(SimulationRecord.request_hash == "hash-orphan-ab")
            )
            self.assertIsNotNone(rq_rec)
            self.assertEqual(rq_rec.status, SimStatus.QUEUED)
            self.assertEqual(rq_rec.orphan_requeues, 1)

            self.assertIsNotNone(ab_rec)
            self.assertEqual(ab_rec.status, SimStatus.ERROR)
            self.assertIn("lost 3 times", ab_rec.message or "")

    async def test_repeated_processing_of_same_completion_event_is_idempotent(self) -> None:
        """Verify processing same completion event does not double-write or notify."""
        now = utcnow()
        async with self.db.session() as session:
            record = _make_sim_record(
                request_hash="hash-finish-race",
                status=SimStatus.RUNNING,
                platform_id="sim-p-888",
                created_at=now,
                submitted_at=now,
            )
            session.add(record)
            await session.flush()
            record_id = record.id

        outcome = Outcome(
            kind="finished",
            status=SimStatus.COMPLETE,
            platform_status=SimulationStatus.COMPLETE,
            alpha_id="alpha-winner-99",
            message="Simulation finished.",
        )

        # First call wins and transitions RUNNING -> COMPLETE
        async with self.db.session() as session:
            rec1 = await session.get(SimulationRecord, record_id)
            self.assertIsNotNone(rec1)
            won_alpha_id = await finish(session, rec1, outcome)
            self.assertEqual(won_alpha_id, "alpha-winner-99")

        # Second call on the now-COMPLETE row must return None (lost race / already finished)
        async with self.db.session() as session:
            rec2 = await session.get(SimulationRecord, record_id)
            self.assertIsNotNone(rec2)
            second_alpha_id = await finish(session, rec2, outcome)
            self.assertIsNone(second_alpha_id)

    async def test_cancellation_and_lost_race_handling(self) -> None:
        """Verify cancel race handling reverts to RUNNING if platform refuses to cancel."""
        now = utcnow()
        async with self.db.session() as session:
            rec = _make_sim_record(
                request_hash="hash-cancel-race",
                status=SimStatus.CANCELLED,
                platform_id="sim-race-1",
                created_at=now,
            )
            session.add(rec)
            await session.flush()
            rec_id = rec.id

        mock_endpoints = MagicMock()
        # Case A: BRAIN refuses cancellation
        mock_endpoints.cancel_simulation = AsyncMock(return_value=False)

        stopped = await cancel_after_lost_race(self.db, mock_endpoints, rec_id, "sim-race-1")
        self.assertFalse(stopped)

        async with self.db.session() as session:
            refreshed = await session.get(SimulationRecord, rec_id)
            self.assertIsNotNone(refreshed)
            self.assertEqual(refreshed.status, SimStatus.RUNNING)
            self.assertIn("would not cancel it, so it is still running", refreshed.message or "")

        # Case B: BRAIN accepts cancellation
        mock_endpoints.cancel_simulation = AsyncMock(return_value=True)
        stopped_success = await cancel_after_lost_race(
            self.db, mock_endpoints, rec_id, "sim-race-1"
        )
        self.assertTrue(stopped_success)

    def test_read_outcome_transient_api_failures_and_partial_results(self) -> None:
        """Verify transient HTTP errors produce retry outcomes, not COMPLETE or ERROR."""
        # 500 Internal Server Error
        o_500 = read_outcome(500, {"detail": "Platform internal error"}, retry_after=5.0)
        self.assertEqual(o_500.kind, "retry")
        self.assertEqual(o_500.delay, 5.0)
        self.assertEqual(o_500.message, "Platform internal error")

        # 429 Too Many Requests
        o_429 = read_outcome(429, {"detail": "Rate limit exceeded"}, retry_after=12.0)
        self.assertEqual(o_429.kind, "retry")
        self.assertEqual(o_429.delay, 12.0)

        # 401 Unauthorized
        o_401 = read_outcome(401, {}, retry_after=None)
        self.assertEqual(o_401.kind, "unauthorized")

        # Partial progress (200 OK with progress)
        o_prog = read_outcome(200, {"progress": 0.65}, retry_after=3.0)
        self.assertEqual(o_prog.kind, "pending")
        self.assertEqual(o_prog.progress, 0.65)
        self.assertEqual(o_prog.delay, 3.0)

        # Multi-simulation fanout
        o_fan = read_outcome(200, {"children": ["child-1", "child-2"]}, retry_after=None)
        self.assertEqual(o_fan.kind, "fanout")
        self.assertEqual(o_fan.children, ("child-1", "child-2"))

        # Terminal error from platform
        o_err = read_outcome(
            200, {"status": "ERROR", "message": "Division by zero"}, retry_after=None
        )
        self.assertEqual(o_err.kind, "finished")
        self.assertEqual(o_err.status, SimStatus.ERROR)
        self.assertIsNone(o_err.alpha_id)

    def test_failed_or_transient_api_trial_never_recorded_as_valid_low_quality_alpha(self) -> None:
        """Verify failed trials (values=None) map to seen=False and are excluded from rankings."""
        # Mock trial rows as stored in DB: (state, params, expression, settings, values)
        trial_rows = [
            (
                TrialState.COMPLETE,
                json.dumps({"field": "close"}),
                "rank(close)",
                json.dumps({"region": "USA", "delay": 1}),
                json.dumps([0.15]),  # Real low-quality alpha
            ),
            (
                TrialState.FAIL,
                json.dumps({"field": "bad_field"}),
                "rank(bad_field)",
                json.dumps({"region": "USA", "delay": 1}),
                None,  # API failure or execution error has NO values
            ),
        ]

        tried, seen = _search_memory(trial_rows, "field")
        self.assertIn("close", tried)
        self.assertIn("bad_field", tried)

        seen_values = list(seen.values())
        self.assertIn(0.15, seen_values)
        self.assertIn(False, seen_values)

        # A failed trial NEVER receives a float score (e.g. 0.0 or positive)
        for score in seen.values():
            if isinstance(score, float):
                self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
