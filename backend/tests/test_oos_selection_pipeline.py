"""End-to-end integration tests verifying that real persisted out-of-sample metrics
drive genetic algorithm parent selection and ranking."""

import math
import tempfile
import unittest
from pathlib import Path

from alpha_harness.brain.schemas import (
    Alpha,
    AlphaCode,
    Check,
    CheckResult,
    SampleStats,
    SimulationSettings,
    SimulationType,
)
from alpha_harness.db.duck import Catalog
from alpha_harness.labs.ga import (
    HEURISTIC_OOS_PENALTY,
    seed_score,
    utility,
)
from alpha_harness.vault.store import SEEDABLE, AlphaVault
from alpha_harness.vault.yields import SUBMITTED


class TestOOSSelectionPipeline(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temp_dir.name) / "test_catalog.duckdb"
        self.catalog = Catalog(self.catalog_path)
        await self.catalog.open()
        self.vault = AlphaVault(self.catalog)

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

    async def asyncTearDown(self) -> None:
        await self.catalog.close()
        self.temp_dir.cleanup()

    async def test_end_to_end_oos_metrics_invert_parent_ranking(self) -> None:
        """Alphas with real persisted out-of-sample metrics loaded through the actual catalog
        query path are scored such that an OOS-stable Alpha beats an overfit Alpha with
        higher in-sample Sharpe."""

        # 1. Overfit Alpha: High in-sample Sharpe (2.5), collapses OOS (0.3 -> ratio 0.12 < 0.25)
        overfit_checks = [
            Check(name="CHECK_SHARPE", result=CheckResult.PASS),
            Check(name="CHECK_FITNESS", result=CheckResult.PASS),
            Check(name="CHECK_TURNOVER", result=CheckResult.PASS),
        ]
        alpha_overfit = Alpha(
            id="alpha-overfit-001",
            type=SimulationType.REGULAR,
            name="Overfit Alpha",
            status="UNSUBMITTED",
            settings=self.settings,
            regular=AlphaCode(code="rank(close) - rank(open)", operator_count=3),
            in_sample=SampleStats(
                sharpe=2.5,
                fitness=1.8,
                turnover=0.15,
                returns=0.20,
                drawdown=0.08,
                margin=0.002,
                checks=overfit_checks,
            ),
            train=SampleStats(sharpe=2.5, fitness=1.8),
            test=SampleStats(sharpe=0.3, fitness=0.2, start_date="2024-01-01", turnover=0.15),
        )

        # 2. Robust Alpha: Moderate in-sample Sharpe (1.8), stable OOS (1.5 -> ratio 0.833 >= 0.50)
        robust_checks = [
            Check(name="CHECK_SHARPE", result=CheckResult.PASS),
            Check(name="CHECK_FITNESS", result=CheckResult.PASS),
            Check(name="CHECK_TURNOVER", result=CheckResult.PASS),
        ]
        alpha_robust = Alpha(
            id="alpha-robust-002",
            type=SimulationType.REGULAR,
            name="Robust Alpha",
            status="UNSUBMITTED",
            settings=self.settings,
            regular=AlphaCode(code="ts_rank(volume, 10)", operator_count=2),
            in_sample=SampleStats(
                sharpe=1.8,
                fitness=1.4,
                turnover=0.10,
                returns=0.15,
                drawdown=0.05,
                margin=0.0015,
                checks=robust_checks,
            ),
            train=SampleStats(sharpe=1.8, fitness=1.4),
            test=SampleStats(sharpe=1.5, fitness=1.2, start_date="2024-01-01", turnover=0.10),
        )

        # 3. Failed Alpha: Negative Sharpe / failed check -> excluded by selection query
        alpha_failed = Alpha(
            id="alpha-failed-003",
            type=SimulationType.REGULAR,
            name="Failed Alpha",
            status="UNSUBMITTED",
            settings=self.settings,
            regular=AlphaCode(code="rank(close)", operator_count=1),
            in_sample=SampleStats(
                sharpe=-0.5,
                fitness=None,
                turnover=0.10,
                returns=-0.05,
                drawdown=0.25,
                margin=0.0,
                checks=[Check(name="CHECK_SHARPE", result=CheckResult.FAIL)],
            ),
            train=SampleStats(sharpe=-0.5, fitness=None),
            test=SampleStats(sharpe=-0.8, fitness=None),
        )

        # 4. Save candidates into DuckDB through the real AlphaVault persistence pipeline
        saved_count = await self.vault.save_alphas([alpha_overfit, alpha_robust, alpha_failed])
        self.assertEqual(saved_count, 3)

        # 5. Query candidates using the REAL SQL query from ga.auto_seeds
        query = f"""
            SELECT a.* FROM alpha a
            WHERE coalesce(a.instrument_type, 'EQUITY') = 'EQUITY'
              AND a.region = ? AND a.delay = ? AND a.universe = ?
              AND NOT {SUBMITTED} AND {SEEDABLE} AND a.expression IS NOT NULL
        """  # noqa: S608
        found = await self.catalog.query(
            query,
            [self.settings.region, self.settings.delay, self.settings.universe],
        )

        # Verify that train_sharpe and test_sharpe are real returned columns from DuckDB
        candidate_ids = {r["alpha_id"] for r in found}
        self.assertIn("alpha-overfit-001", candidate_ids)
        self.assertIn("alpha-robust-002", candidate_ids)
        # alpha_failed has fitness IS NULL, so SEEDABLE query excludes it
        self.assertNotIn("alpha-failed-003", candidate_ids)

        row_overfit = next(r for r in found if r["alpha_id"] == "alpha-overfit-001")
        row_robust = next(r for r in found if r["alpha_id"] == "alpha-robust-002")

        self.assertEqual(row_overfit["train_sharpe"], 2.5)
        self.assertEqual(row_overfit["test_sharpe"], 0.3)
        self.assertEqual(row_robust["train_sharpe"], 1.8)
        self.assertEqual(row_robust["test_sharpe"], 1.5)

        # 6. Apply seed_score to candidates
        score_overfit = seed_score(row_overfit)
        score_robust = seed_score(row_robust)

        # Overfit alpha receives HEURISTIC_OOS_PENALTY (0.50)
        # Robust alpha receives HEURISTIC_OOS_BONUS (1.25)
        self.assertGreater(
            score_robust,
            score_overfit,
            f"OOS-stable alpha ({score_robust}) must rank above overfit alpha ({score_overfit})",
        )

        # Verify ranking order in sorted candidate pool
        ranked = sorted(found, key=seed_score, reverse=True)
        self.assertEqual(ranked[0]["alpha_id"], "alpha-robust-002")
        self.assertEqual(ranked[1]["alpha_id"], "alpha-overfit-001")

    def test_seed_score_boundary_conditions(self) -> None:
        """Verify robust handling of NaN, Inf, absent validation metrics, negative test Sharpe,
        and invalid data types."""
        base_row = {
            "sharpe": 2.0,
            "fitness": 1.5,
            "turnover": 0.1,
            "margin": 0.001,
            "checks": None,
        }
        expected_base = utility(base_row)

        # Absent validation data -> baseline score preserved without modification
        score_no_val = seed_score(base_row)
        self.assertAlmostEqual(score_no_val, expected_base)

        # None train or test -> baseline score preserved
        score_none_test = seed_score({**base_row, "train_sharpe": 2.0, "test_sharpe": None})
        self.assertAlmostEqual(score_none_test, expected_base)

        score_none_train = seed_score({**base_row, "train_sharpe": None, "test_sharpe": 1.5})
        self.assertAlmostEqual(score_none_train, expected_base)

        # Negative test Sharpe -> penalized with HEURISTIC_OOS_PENALTY
        score_neg_test = seed_score({**base_row, "train_sharpe": 2.0, "test_sharpe": -0.5})
        self.assertAlmostEqual(score_neg_test, expected_base * HEURISTIC_OOS_PENALTY)

        # Zero train Sharpe -> protected against ZeroDivisionError, preserves baseline score
        score_zero_train = seed_score({**base_row, "train_sharpe": 0.0, "test_sharpe": 1.5})
        self.assertAlmostEqual(score_zero_train, expected_base)

        # NaN test Sharpe -> handled cleanly, preserves baseline score
        score_nan_test = seed_score({**base_row, "train_sharpe": 2.0, "test_sharpe": float("nan")})
        self.assertAlmostEqual(score_nan_test, expected_base)

        # Inf test Sharpe -> handled cleanly, preserves baseline score
        score_inf_test = seed_score({**base_row, "train_sharpe": 2.0, "test_sharpe": float("inf")})
        self.assertAlmostEqual(score_inf_test, expected_base)

        # String invalid type -> handled cleanly, preserves baseline score
        score_str_test = seed_score({**base_row, "train_sharpe": "invalid", "test_sharpe": "abc"})
        self.assertAlmostEqual(score_str_test, expected_base)

        # Non-finite base utility
        self.assertTrue(math.isnan(seed_score({"sharpe": float("nan")})))


if __name__ == "__main__":
    unittest.main()
