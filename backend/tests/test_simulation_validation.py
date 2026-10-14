"""Comprehensive model-level validation tests for SimulationRequest across all simulation types:
- REGULAR
- REGION_AGNOSTIC
- COMBO
- SUPER

Tests valid expressions, missing required expressions, empty/whitespace expressions,
malformed expressions, companion constraints, dotted attributes, and recipes.
"""

import unittest

from pydantic import ValidationError

from alpha_harness.brain.schemas import (
    FULL_MODE,
    REGION_AGNOSTIC_REGION,
    SimulationRequest,
    SimulationSettings,
    SimulationType,
)
from alpha_harness.labs.super_alpha import COMBOS, selections


class TestSimulationRequestValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SimulationSettings(
            region="USA",
            delay=1,
            universe="TOP3000",
            neutralization="SUBINDUSTRY",
            simulation_mode=FULL_MODE,
        )

    # --- 1. REGULAR SIMULATION TESTS ---

    def test_regular_valid_expression(self) -> None:
        req = SimulationRequest(
            type=SimulationType.REGULAR,
            settings=self.settings,
            regular="rank(close) - rank(open)",
        )
        self.assertEqual(req.type, SimulationType.REGULAR)
        self.assertEqual(req.regular, "rank(close) - rank(open)")

    def test_regular_missing_expression_fails(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            SimulationRequest(
                type=SimulationType.REGULAR,
                settings=self.settings,
            )
        self.assertIn("requires a non-empty 'regular' expression", str(ctx.exception))

    def test_regular_empty_and_whitespace_fails(self) -> None:
        for empty_val in ("", "   ", "\t\n"):
            with self.subTest(empty=empty_val):
                with self.assertRaises(ValidationError) as ctx:
                    SimulationRequest(
                        type=SimulationType.REGULAR,
                        settings=self.settings,
                        regular=empty_val,
                    )
                self.assertIn("requires a non-empty 'regular' expression", str(ctx.exception))

    def test_regular_malformed_expression_fails(self) -> None:
        malformed = ["rank(close) + ", "ts_rank(close, ", "((close - open)", "close / * open"]
        for expr in malformed:
            with self.subTest(expr=expr):
                with self.assertRaises(ValidationError) as ctx:
                    SimulationRequest(
                        type=SimulationType.REGULAR,
                        settings=self.settings,
                        regular=expr,
                    )
                self.assertIn("Malformed regular expression", str(ctx.exception))

    def test_regular_rejects_conflicting_companion_expressions(self) -> None:
        # REGULAR should not accept combo or selection
        with self.assertRaises(ValidationError) as ctx:
            SimulationRequest(
                type=SimulationType.REGULAR,
                settings=self.settings,
                regular="rank(close)",
                combo="1",
            )
        self.assertIn("does not accept a 'combo' expression", str(ctx.exception))

        with self.assertRaises(ValidationError) as ctx:
            SimulationRequest(
                type=SimulationType.REGULAR,
                settings=self.settings,
                regular="rank(close)",
                selection="operator_count <= 800",
            )
        self.assertIn("does not accept a 'selection' expression", str(ctx.exception))

    # --- 2. REGION_AGNOSTIC SIMULATION TESTS ---

    def test_region_agnostic_valid_expression(self) -> None:
        ra_settings = self.settings.model_copy(update={"region": REGION_AGNOSTIC_REGION})
        req = SimulationRequest(
            type=SimulationType.REGION_AGNOSTIC,
            settings=ra_settings,
            regular="rank(close) / (1 + ts_std_dev(close, 20))",
        )
        self.assertEqual(req.type, SimulationType.REGION_AGNOSTIC)
        self.assertIsNotNone(req.regular)

    def test_region_agnostic_auto_derived_from_all_region(self) -> None:
        ra_settings = self.settings.model_copy(update={"region": REGION_AGNOSTIC_REGION})
        req = SimulationRequest(
            settings=ra_settings,
            regular="rank(close)",
        )
        # Type is automatically derived as REGION_AGNOSTIC when region == "ALL"
        self.assertEqual(req.type, SimulationType.REGION_AGNOSTIC)

    def test_region_agnostic_missing_expression_fails(self) -> None:
        ra_settings = self.settings.model_copy(update={"region": REGION_AGNOSTIC_REGION})
        with self.assertRaises(ValidationError) as ctx:
            SimulationRequest(
                type=SimulationType.REGION_AGNOSTIC,
                settings=ra_settings,
            )
        self.assertIn("requires a non-empty 'regular' expression", str(ctx.exception))

    def test_region_agnostic_malformed_expression_fails(self) -> None:
        ra_settings = self.settings.model_copy(update={"region": REGION_AGNOSTIC_REGION})
        with self.assertRaises(ValidationError) as ctx:
            SimulationRequest(
                type=SimulationType.REGION_AGNOSTIC,
                settings=ra_settings,
                regular="rank(close) - ",
            )
        self.assertIn("Malformed regular expression", str(ctx.exception))

    # --- 3. COMBO SIMULATION TESTS ---

    def test_combo_valid_simple_and_multi_statement(self) -> None:
        # Simple combo formula
        req1 = SimulationRequest(
            type=SimulationType.COMBO,
            settings=self.settings,
            combo="1",
        )
        self.assertEqual(req1.type, SimulationType.COMBO)
        self.assertEqual(req1.combo, "1")

        # Multi-statement combo formula with assignments and dotted attribute access
        combo_multi = "stats = generate_stats(alpha);\nmax(ts_ir(stats.returns, 250), 0)"
        req2 = SimulationRequest(
            type=SimulationType.COMBO,
            settings=self.settings,
            combo=combo_multi,
        )
        self.assertEqual(req2.type, SimulationType.COMBO)
        self.assertEqual(req2.combo, combo_multi)

    def test_combo_missing_expression_fails(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            SimulationRequest(
                type=SimulationType.COMBO,
                settings=self.settings,
            )
        self.assertIn("requires a non-empty 'combo' expression", str(ctx.exception))

    def test_combo_empty_and_whitespace_fails(self) -> None:
        for empty_val in ("", "   ", "\t\n"):
            with self.subTest(empty=empty_val):
                with self.assertRaises(ValidationError) as ctx:
                    SimulationRequest(
                        type=SimulationType.COMBO,
                        settings=self.settings,
                        combo=empty_val,
                    )
                self.assertIn("requires a non-empty 'combo' expression", str(ctx.exception))

    def test_combo_malformed_expression_fails(self) -> None:
        malformed = ["stats = ", "max(ts_ir(stats.returns, 250)", "stats.returns + "]
        for expr in malformed:
            with self.subTest(expr=expr):
                with self.assertRaises(ValidationError) as ctx:
                    SimulationRequest(
                        type=SimulationType.COMBO,
                        settings=self.settings,
                        combo=expr,
                    )
                self.assertIn("Malformed combo expression", str(ctx.exception))

    def test_combo_rejects_conflicting_companion_expressions(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            SimulationRequest(
                type=SimulationType.COMBO,
                settings=self.settings,
                combo="1",
                regular="rank(close)",
            )
        self.assertIn("does not accept a 'regular' expression", str(ctx.exception))

        with self.assertRaises(ValidationError) as ctx:
            SimulationRequest(
                type=SimulationType.COMBO,
                settings=self.settings,
                combo="1",
                selection="operator_count <= 800",
            )
        self.assertIn("does not accept a 'selection' expression", str(ctx.exception))

    # --- 4. SUPER ALPHA (SUPER) SIMULATION TESTS ---

    def test_super_valid_combinations(self) -> None:
        # Test every library selection paired with every combo recipe from super_alpha.py
        for sel in selections(10):
            for combo in COMBOS:
                with self.subTest(selection=sel.name, combo=combo.name):
                    req = SimulationRequest(
                        type=SimulationType.SUPER,
                        settings=self.settings,
                        selection=sel.code,
                        combo=combo.code,
                    )
                    self.assertEqual(req.type, SimulationType.SUPER)
                    self.assertEqual(req.selection, sel.code)
                    self.assertEqual(req.combo, combo.code)

    def test_super_missing_selection_fails(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            SimulationRequest(
                type=SimulationType.SUPER,
                settings=self.settings,
                combo="1",
            )
        self.assertIn("requires a non-empty 'selection' expression", str(ctx.exception))

    def test_super_missing_combo_fails(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            SimulationRequest(
                type=SimulationType.SUPER,
                settings=self.settings,
                selection="(operator_count <= 800) * (self_correlation < 0.7)",
            )
        self.assertIn("requires a non-empty 'combo' expression", str(ctx.exception))

    def test_super_empty_and_whitespace_fails(self) -> None:
        valid_sel = "(operator_count <= 800)"
        valid_combo = "1"

        # Empty selection
        for empty_val in ("", "   ", "\n"):
            with self.subTest(empty_sel=empty_val):
                with self.assertRaises(ValidationError) as ctx:
                    SimulationRequest(
                        type=SimulationType.SUPER,
                        settings=self.settings,
                        selection=empty_val,
                        combo=valid_combo,
                    )
                self.assertIn("requires a non-empty 'selection' expression", str(ctx.exception))

        # Empty combo
        for empty_val in ("", "   ", "\n"):
            with self.subTest(empty_combo=empty_val):
                with self.assertRaises(ValidationError) as ctx:
                    SimulationRequest(
                        type=SimulationType.SUPER,
                        settings=self.settings,
                        selection=valid_sel,
                        combo=empty_val,
                    )
                self.assertIn("requires a non-empty 'combo' expression", str(ctx.exception))

    def test_super_malformed_expressions_fail(self) -> None:
        valid_sel = "(operator_count <= 800)"
        valid_combo = "1"

        # Malformed selection
        with self.assertRaises(ValidationError) as ctx:
            SimulationRequest(
                type=SimulationType.SUPER,
                settings=self.settings,
                selection="(operator_count <= ",
                combo=valid_combo,
            )
        self.assertIn("Malformed selection expression", str(ctx.exception))

        # Malformed combo
        with self.assertRaises(ValidationError) as ctx:
            SimulationRequest(
                type=SimulationType.SUPER,
                settings=self.settings,
                selection=valid_sel,
                combo="stats.returns + ",
            )
        self.assertIn("Malformed combo expression", str(ctx.exception))

    def test_super_rejects_regular_companion_expression(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            SimulationRequest(
                type=SimulationType.SUPER,
                settings=self.settings,
                selection="(operator_count <= 800)",
                combo="1",
                regular="rank(close)",
            )
        self.assertIn("does not accept a 'regular' expression", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
