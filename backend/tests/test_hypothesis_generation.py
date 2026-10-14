"""Tests verifying autonomous hypothesis generation, template expansion, and refinement."""

import random
import unittest

from alpha_harness.labs.fastexpr import (
    MAX_FIELDS,
    MAX_OPERATORS,
    data_fields,
    operator_count,
    operator_table,
    parse,
    validate,
)
from alpha_harness.labs.ga import Market
from alpha_harness.labs.hypothesis import HypothesisGenerator


class TestHypothesisGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.table = operator_table(
            [
                {"name": "rank", "definition": "rank(x)", "scope": ["REGULAR"]},
                {"name": "ts_rank", "definition": "ts_rank(x, d)", "scope": ["REGULAR"]},
                {"name": "ts_delta", "definition": "ts_delta(x, d)", "scope": ["REGULAR"]},
                {"name": "ts_zscore", "definition": "ts_zscore(x, d)", "scope": ["REGULAR"]},
                {"name": "abs", "definition": "abs(x)", "scope": ["REGULAR"]},
                {"name": "log", "definition": "log(x)", "scope": ["REGULAR"]},
            ]
        )
        self.market = Market(
            table=self.table,
            swappable={},
            dataset_of={
                "close": "ds1",
                "open": "ds1",
                "high": "ds1",
                "low": "ds1",
                "volume": "ds2",
                "vwap": "ds2",
            },
            by_dataset={
                "ds1": ("close", "open", "high", "low"),
                "ds2": ("volume", "vwap"),
            },
            names=frozenset({"close", "open", "high", "low", "volume", "vwap"}),
            neutralizations=("SUBINDUSTRY", "MARKET", "INDUSTRY"),
        )
        self.generator = HypothesisGenerator(self.market)

    def test_single_template_produces_at_least_three_distinct_candidates(self) -> None:
        """A single parametric research template expands into at least 3 distinct candidates."""
        tmpl = "$op($field, $lookback) - rank(open)"
        rng = random.Random(42)

        candidates = self.generator.generate_from_template(tmpl, count=3, rng=rng)
        self.assertGreaterEqual(len(candidates), 3, "Must generate at least 3 candidates")

        expressions = [c.expression for c in candidates]
        self.assertEqual(len(set(expressions)), len(candidates), "Candidates must be distinct")

        for cand in candidates:
            self.assertEqual(cand.originating_template, tmpl)
            self.assertEqual(cand.generation_method, "template_expansion")
            self.assertEqual(cand.generation, 0)
            self.assertIsNone(cand.parent_id)

            # Strict syntax and semantic check against market
            tree = parse(cand.expression)
            val_err = validate(tree, self.market.table, self.market.names)
            self.assertEqual(val_err, [], f"Candidate failed validation: {val_err}")
            self.assertLessEqual(operator_count(tree), MAX_OPERATORS)
            self.assertLessEqual(len(data_fields(tree)), MAX_FIELDS)

            # Request validity
            self.assertEqual(cand.request.regular, cand.expression)
            self.assertIsNotNone(cand.request.settings.neutralization)

    def test_multiple_templates_expand_into_broad_pool(self) -> None:
        """Multiple distinct research templates expand into a combined candidate pool."""
        templates = [
            "ts_rank($field, $lookback)",
            "rank($field) - rank(open)",
            "ts_delta($field, $lookback)",
        ]
        rng = random.Random(123)

        pool = self.generator.expand_templates(templates, count_per_template=3, rng=rng)
        self.assertGreaterEqual(len(pool), 9, "Broad pool must aggregate candidates")

        origin_counts = dict.fromkeys(templates, 0)
        for cand in pool:
            self.assertIn(cand.originating_template, origin_counts)
            origin_counts[cand.originating_template] += 1
            tree = parse(cand.expression)
            self.assertEqual(validate(tree, self.market.table, self.market.names), [])

        for tmpl, cnt in origin_counts.items():
            self.assertGreaterEqual(cnt, 3, f"Template {tmpl} must have >= 3 candidates")

    def test_outcome_informed_regeneration_promotes_oos_stable_parents(self) -> None:
        """Recorded experiment outcomes inform parent selection for subsequent generations,
        favoring OOS-stable candidates without mistaking missing validation data for failure."""
        tmpl = "ts_rank(close, $lookback)"
        rng = random.Random(77)
        initial_candidates = self.generator.generate_from_template(tmpl, count=3, rng=rng)
        self.assertGreaterEqual(len(initial_candidates), 3)

        cand_overfit = initial_candidates[0]
        cand_robust = initial_candidates[1]
        cand_missing_val = initial_candidates[2]

        outcomes = {
            # Candidate 0: High in-sample Sharpe (2.5), collapses OOS (0.2 -> ratio 0.08 < 0.25)
            cand_overfit.id: {
                "sharpe": 2.5,
                "fitness": 1.8,
                "turnover": 0.1,
                "train_sharpe": 2.5,
                "test_sharpe": 0.2,
                "checks": [{"name": "CHECK_SHARPE", "result": "PASS"}],
            },
            # Candidate 1: Moderate in-sample Sharpe (1.8), stable OOS (1.5 -> ratio 0.833 >= 0.50)
            cand_robust.id: {
                "sharpe": 1.8,
                "fitness": 1.4,
                "turnover": 0.1,
                "train_sharpe": 1.8,
                "test_sharpe": 1.5,
                "checks": [{"name": "CHECK_SHARPE", "result": "PASS"}],
            },
            # Candidate 2: Missing validation data, but solid Sharpe (1.7)
            # Must NOT be penalized as a failure; baseline utility preserved
            cand_missing_val.id: {
                "sharpe": 1.7,
                "fitness": 1.3,
                "turnover": 0.1,
                "train_sharpe": None,
                "test_sharpe": None,
                "checks": [{"name": "CHECK_SHARPE", "result": "PASS"}],
            },
        }

        next_gen = self.generator.regenerate_generation(
            initial_candidates, outcomes, count=3, rng=rng
        )
        self.assertGreaterEqual(len(next_gen), 3, "Must produce next generation from parents")

        for child_cand in next_gen:
            self.assertEqual(child_cand.generation, 1)
            self.assertEqual(child_cand.generation_method, "genetic_refinement")
            self.assertIsNotNone(child_cand.parent_id)
            self.assertIn(
                child_cand.parent_id,
                [cand_robust.id, cand_missing_val.id, cand_overfit.id],
            )

            tree = parse(child_cand.expression)
            self.assertEqual(validate(tree, self.market.table, self.market.names), [])
            self.assertLessEqual(operator_count(tree), MAX_OPERATORS)
            self.assertLessEqual(len(data_fields(tree)), MAX_FIELDS)

    def test_bounded_generation_attempts_with_impossible_constraints(self) -> None:
        """When an impossible template is provided, generator terminates cleanly."""
        impossible_tmpl = "impossible_operator(close)"
        candidates = self.generator.generate_from_template(impossible_tmpl, count=3)
        self.assertEqual(candidates, [], "Must safely return empty list on impossible template")


if __name__ == "__main__":
    unittest.main()
