"""Tests verifying legacy request-hash compatibility, deduplication, and safe migration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from alpha_harness.brain.schemas import (
    SimulationRequest,
    SimulationSettings,
    SimulationType,
)
from alpha_harness.db.models import DedupEntry, SimStatus
from alpha_harness.db.sqlite import Database
from alpha_harness.engine.lifecycle import (
    legacy_request_hash,
    new_record,
    rekey_historical_dedup,
    request_hash,
)
from alpha_harness.engine.slots import BatchEngine


class TestLegacyHashCompatibility(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.settings = SimulationSettings(
            instrument_type="EQUITY",
            region="USA",
            universe="TOP3000",
            delay=1,
            decay=4,
            neutralization="INDUSTRY",
            truncation=0.08,
            pasteurization="ON",
            unit_handling="VERIFY",
            nan_handling="OFF",
            language="FASTEXPR",
            visualization=False,
        )

    def test_canonical_expression_hash_identity(self) -> None:
        """Standard formula without comments produces identical legacy and canonical hashes."""
        req = SimulationRequest(
            type=SimulationType.REGULAR,
            settings=self.settings,
            regular="rank(close) - rank(open)",
        )
        leg_hash = legacy_request_hash(req)
        can_hash = request_hash(req)
        self.assertEqual(
            leg_hash,
            can_hash,
            "Clean formulas must produce identical hashes across paths.",
        )

    def test_formatting_variations_differ_in_legacy_but_unify_in_canonical(self) -> None:
        """Formulas with whitespace or comments differ in legacy, but match in canonical."""
        clean = SimulationRequest(
            type=SimulationType.REGULAR,
            settings=self.settings,
            regular="rank(close) - rank(open)",
        )
        messy_space = SimulationRequest(
            type=SimulationType.REGULAR,
            settings=self.settings,
            regular="  rank( close )   -   rank( open )  ",
        )
        messy_comment = SimulationRequest(
            type=SimulationType.REGULAR,
            settings=self.settings,
            regular="# daily rank difference\nrank(close) - rank(open);",
        )

        # Legacy hashes reflect raw string representation differences
        self.assertNotEqual(legacy_request_hash(clean), legacy_request_hash(messy_space))
        self.assertNotEqual(legacy_request_hash(clean), legacy_request_hash(messy_comment))

        # Canonical hashes unify semantic equivalence
        self.assertEqual(request_hash(clean), request_hash(messy_space))
        self.assertEqual(request_hash(clean), request_hash(messy_comment))

    def test_different_execution_settings_never_collide(self) -> None:
        """Requests changing simulation behavior produce different hashes in both modes."""
        base_req = SimulationRequest(
            type=SimulationType.REGULAR,
            settings=self.settings,
            regular="rank(close) - rank(open)",
        )
        r_decay = SimulationRequest(
            type=SimulationType.REGULAR,
            settings=self.settings.model_copy(update={"decay": 10}),
            regular="rank(close) - rank(open)",
        )
        r_neut = SimulationRequest(
            type=SimulationType.REGULAR,
            settings=self.settings.model_copy(update={"neutralization": "SUBINDUSTRY"}),
            regular="rank(close) - rank(open)",
        )
        r_univ = SimulationRequest(
            type=SimulationType.REGULAR,
            settings=self.settings.model_copy(update={"universe": "TOP1000"}),
            regular="rank(close) - rank(open)",
        )
        r_delay = SimulationRequest(
            type=SimulationType.REGULAR,
            settings=self.settings.model_copy(update={"delay": 0}),
            regular="rank(close) - rank(open)",
        )

        for variant in (r_decay, r_neut, r_univ, r_delay):
            self.assertNotEqual(request_hash(base_req), request_hash(variant))
            self.assertNotEqual(legacy_request_hash(base_req), legacy_request_hash(variant))

    async def test_bounded_legacy_hash_lookup_on_enqueue(self) -> None:
        """Incoming messy requests match historical DedupEntry records keyed by legacy_request_hash,
        preventing duplicate simulations and re-keying canonical hash without deletion."""
        temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(temp_dir.name) / "test_dedup_bounded.db"
        db = Database(db_path)
        await db.create_all()

        messy_req = SimulationRequest(
            type=SimulationType.REGULAR,
            settings=self.settings,
            regular="# historical formula with comments\nrank(close) - rank(open);",
        )
        legacy_hash = legacy_request_hash(messy_req)
        canonical_hash = request_hash(messy_req)
        self.assertNotEqual(legacy_hash, canonical_hash)

        # Store historical DedupEntry under legacy hash
        async with db.session() as session:
            session.add(DedupEntry(request_hash=legacy_hash, alpha_id="alpha-hist-bounded"))

        tracker = MagicMock()
        tracker.notify = AsyncMock()
        engine = BatchEngine(
            db,
            MagicMock(),
            tracker,
            on_unauthorized=AsyncMock(return_value=True),
        )

        # Enqueue incoming messy request -> bounded lookup matches legacy_hash
        result = await engine.enqueue([messy_req], task="test")
        skipped = result.get("skipped", [])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["alphaId"], "alpha-hist-bounded")

        async with db.session() as session:
            # Both legacy and canonical entries exist now
            legacy_entry = await session.get(DedupEntry, legacy_hash)
            canonical_entry = await session.get(DedupEntry, canonical_hash)
            self.assertIsNotNone(legacy_entry, "Legacy dedup record must NOT be deleted!")
            self.assertIsNotNone(canonical_entry, "Canonical dedup record must be added!")
            self.assertEqual(canonical_entry.alpha_id, "alpha-hist-bounded")

        await db.dispose()
        temp_dir.cleanup()

    async def test_historical_dedup_rekey_migration_and_restart(self) -> None:
        """Historical SimulationRecords are rekeyed on startup, ensuring that clean incoming
        requests recognize historical results without deleting historical dedup rows."""
        temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(temp_dir.name) / "test_dedup_rekey.db"
        db = Database(db_path)
        await db.create_all()

        messy_req = SimulationRequest(
            type=SimulationType.REGULAR,
            settings=self.settings,
            regular="# legacy comment\nrank(close) - rank(open);",
        )
        legacy_hash = legacy_request_hash(messy_req)
        canonical_hash = request_hash(messy_req)

        # Historical state: SimulationRecord and DedupEntry created before Phase 5
        async with db.session() as session:
            session.add(DedupEntry(request_hash=legacy_hash, alpha_id="alpha-hist-migration"))
            rec = new_record(
                messy_req,
                task="manual",
                status=SimStatus.COMPLETE,
                alpha_id="alpha-hist-migration",
            )
            rec.request_hash = legacy_hash  # Simulate pre-Phase 5 historical hash
            session.add(rec)

        # Run rekey migration (which runs during state.startup)
        rekeyed_count = await rekey_historical_dedup(db)
        self.assertEqual(rekeyed_count, 1)

        # Restart database to test persistence and recovery
        await db.dispose()
        db_restarted = Database(db_path)
        await db_restarted.create_all()

        # Submit a clean, canonical request for the same formula
        clean_req = SimulationRequest(
            type=SimulationType.REGULAR,
            settings=self.settings,
            regular="rank(close) - rank(open)",
        )
        self.assertEqual(request_hash(clean_req), canonical_hash)

        tracker = MagicMock()
        tracker.notify = AsyncMock()
        engine = BatchEngine(
            db_restarted,
            MagicMock(),
            tracker,
            on_unauthorized=AsyncMock(return_value=True),
        )

        result = await engine.enqueue([clean_req], task="test")
        skipped = result.get("skipped", [])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["alphaId"], "alpha-hist-migration")

        async with db_restarted.session() as session:
            legacy_entry = await session.get(DedupEntry, legacy_hash)
            canonical_entry = await session.get(DedupEntry, canonical_hash)
            self.assertIsNotNone(legacy_entry, "Legacy entry preserved across restart!")
            self.assertIsNotNone(canonical_entry, "Canonical entry present across restart!")

        await db_restarted.dispose()
        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
