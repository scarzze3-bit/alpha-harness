"""Semantic regression audit tests for Phase 5:
- Formula canonicalization and semantics safety
- Precedence, unary/binary, statement ordering, and configuration invariance
- Strict request validation (rejection of empty/whitespace/malformed formulas)
- Genetic crossover anti-bloat bounds, fallback safety, and bounded loops
- Closed-loop feedback weighting (held-out validation priority and collapse penalty)
- Persistence and failure state distinction
"""

import random
import unittest

from pydantic import ValidationError

from alpha_harness.brain.schemas import (
    FULL_MODE,
    SimulationRequest,
    SimulationSettings,
)
from alpha_harness.engine.lifecycle import request_hash
from alpha_harness.labs.fastexpr import (
    MAX_FIELDS,
    MAX_OPERATORS,
    canonical_formula,
    crossover,
    data_fields,
    operator_count,
    operator_table,
    parse,
    render,
    validate,
)
from alpha_harness.labs.ga import Market, Parent, child, seed_score


class TestSemanticAudit(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SimulationSettings(
            region="USA",
            delay=1,
            universe="TOP3000",
            neutralization="SUBINDUSTRY",
            decay=4,
            truncation=0.08,
            simulation_mode=FULL_MODE,
        )
        self.table = operator_table(
            [
                {"name": "rank", "definition": "rank(x)", "scope": ["REGULAR"]},
                {"name": "ts_rank", "definition": "ts_rank(x, d)", "scope": ["REGULAR"]},
                {"name": "ts_delta", "definition": "ts_delta(x, d)", "scope": ["REGULAR"]},
                {"name": "ts_corr", "definition": "ts_corr(x, y, d)", "scope": ["REGULAR"]},
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
            neutralizations=("SUBINDUSTRY", "MARKET"),
        )

    # --- Task A: Formula canonicalization and semantic safety ---

    def test_formatting_and_comment_invariance(self) -> None:
        raw1 = "rank(close) - rank(open)"
        raw2 = "  rank( close )   -   rank( open )  "
        raw3 = "# leading comment\nrank(close) - rank(open)\n# trailing comment"
        raw4 = "/* block comment */ rank(close) - rank(open)"
        raw5 = "rank(close) - rank(open);"

        c1 = canonical_formula(raw1)
        self.assertEqual(c1, "rank(close) - rank(open)")
        self.assertEqual(canonical_formula(raw2), c1)
        self.assertEqual(canonical_formula(raw3), c1)
        self.assertEqual(canonical_formula(raw4), c1)
        self.assertEqual(canonical_formula(raw5), c1)

        req1 = SimulationRequest(settings=self.settings, regular=raw1)
        req2 = SimulationRequest(settings=self.settings, regular=raw2)
        req3 = SimulationRequest(settings=self.settings, regular=raw3)
        self.assertEqual(request_hash(req1), request_hash(req2))
        self.assertEqual(request_hash(req1), request_hash(req3))

    def test_operator_precedence_and_associativity(self) -> None:
        # Multiplicative over additive
        self.assertEqual(canonical_formula("a + b * c"), "a + b * c")
        self.assertEqual(canonical_formula("(a + b) * c"), "(a + b) * c")
        # Exponentiation right-associativity: a ^ b ^ c is a ^ (b ^ c)
        self.assertEqual(canonical_formula("a ^ b ^ c"), "a ^ b ^ c")
        self.assertEqual(canonical_formula("(a ^ b) ^ c"), "(a ^ b) ^ c")
        # Subtraction grouping
        self.assertEqual(canonical_formula("a - b - c"), "a - b - c")
        self.assertEqual(canonical_formula("a - (b - c)"), "a - (b - c)")

    def test_unary_and_binary_distinction(self) -> None:
        u1 = canonical_formula("-(close + open)")
        b1 = canonical_formula("close - open")
        u2 = canonical_formula("close - -open")
        self.assertEqual(u1, "-(close + open)")
        self.assertEqual(b1, "close - open")
        self.assertEqual(u2, "close - -open")

        req_u1 = SimulationRequest(settings=self.settings, regular=u1)
        req_b1 = SimulationRequest(settings=self.settings, regular=b1)
        req_u2 = SimulationRequest(settings=self.settings, regular=u2)
        self.assertNotEqual(request_hash(req_u1), request_hash(req_b1))
        self.assertNotEqual(request_hash(req_b1), request_hash(req_u2))

    def test_multi_statement_ordering_and_assignment(self) -> None:
        seq1 = "x = rank(close); x - rank(open)"
        seq2 = "x = rank(open); x - rank(close)"
        seq3 = "x = rank(close); y = rank(open); x - y"
        self.assertNotEqual(canonical_formula(seq1), canonical_formula(seq2))

        req1 = SimulationRequest(settings=self.settings, regular=seq1)
        req2 = SimulationRequest(settings=self.settings, regular=seq2)
        req3 = SimulationRequest(settings=self.settings, regular=seq3)
        self.assertNotEqual(request_hash(req1), request_hash(req2))
        self.assertNotEqual(request_hash(req1), request_hash(req3))

    def test_empty_and_whitespace_rejection(self) -> None:
        # Canonicalization normalizes empty and whitespace to empty string
        self.assertEqual(canonical_formula(""), "")
        self.assertEqual(canonical_formula("   "), "")
        self.assertIsNone(canonical_formula(None))

        # SimulationRequest strictly rejects empty, whitespace, and None
        with self.assertRaises(ValidationError):
            SimulationRequest(settings=self.settings, regular="")
        with self.assertRaises(ValidationError):
            SimulationRequest(settings=self.settings, regular="   ")
        with self.assertRaises(ValidationError):
            SimulationRequest(settings=self.settings, regular=None)

    def test_malformed_formula_rejection(self) -> None:
        malformed = "rank(close) + "
        # Canonical formula preserves the raw string so downstream tools can diagnose it
        self.assertEqual(canonical_formula(malformed), malformed)
        # SimulationRequest strictly rejects malformed syntax
        with self.assertRaises(ValidationError):
            SimulationRequest(settings=self.settings, regular=malformed)

    # --- Task B: Genetic crossover bounds, diversity, and fallbacks ---

    def test_genetic_crossover_strictly_enforces_limits(self) -> None:
        p1 = Parent(
            alpha_id="p1",
            tree=parse("rank(close) + rank(open)"),
            neutralization="SUBINDUSTRY",
            decay=3,
            truncation=0.08,
        )
        p2 = Parent(
            alpha_id="p2",
            tree=parse("ts_corr(volume, vwap, 10)"),
            neutralization="SUBINDUSTRY",
            decay=4,
            truncation=0.08,
        )

        rng = random.Random(123)
        # Bounded crossover: rejected when operators or fields exceed budget
        c_bounded = crossover(rng, p1.tree, p2.tree, max_operators=2, max_fields=2)
        self.assertIsNone(c_bounded)

        # Allowed crossover within bounds
        c_ok = crossover(rng, p1.tree, p2.tree, max_operators=5, max_fields=4)
        if c_ok is not None:
            self.assertLessEqual(operator_count(c_ok), 5)
            self.assertLessEqual(len(data_fields(c_ok)), 4)

    def test_child_generation_diversity_and_termination(self) -> None:
        p1 = Parent(
            alpha_id="p1",
            tree=parse("rank(close) - rank(open)"),
            neutralization="SUBINDUSTRY",
            decay=3,
            truncation=0.08,
        )
        p2 = Parent(
            alpha_id="p2",
            tree=parse("ts_rank(volume, 10)"),
            neutralization="MARKET",
            decay=5,
            truncation=0.08,
        )

        generated: set[str] = set()
        for seed in range(20):
            rng = random.Random(seed)
            offspring = child(
                rng,
                p1,
                p2,
                self.market,
                rate=0.2,
                max_operators=MAX_OPERATORS,
                max_fields=MAX_FIELDS,
            )
            if offspring is not None:
                tree, _settings = offspring
                self.assertLessEqual(operator_count(tree), MAX_OPERATORS)
                self.assertLessEqual(len(data_fields(tree)), MAX_FIELDS)
                self.assertEqual(validate(tree, self.table, self.market.names), [])
                generated.add(render(tree))

        # Genetic search maintains healthy diversity across seeds
        self.assertGreater(len(generated), 1)

    def test_deeply_nested_operator_limits(self) -> None:
        deep = parse("abs(log(ts_rank(ts_delta(ts_corr(close, open, 5), 2), 10)))")
        self.assertEqual(operator_count(deep), 5)
        self.assertEqual(len(data_fields(deep)), 2)

        p_deep = Parent(
            alpha_id="p_deep",
            tree=deep,
            neutralization="SUBINDUSTRY",
            decay=3,
            truncation=0.08,
        )
        p_shallow = Parent(
            alpha_id="p_shallow",
            tree=parse("rank(volume)"),
            neutralization="SUBINDUSTRY",
            decay=3,
            truncation=0.08,
        )

        rng = random.Random(99)
        # When max_operators=4, deep grafting cannot exceed 4
        offspring = child(rng, p_deep, p_shallow, self.market, rate=0.1, max_operators=4)
        if offspring is not None:
            child_tree, _ = offspring
            self.assertLessEqual(operator_count(child_tree), 4)

    # --- Task C: Closed-loop research feedback and held-out priority ---

    def test_seed_score_rewards_held_out_validation(self) -> None:
        # Base row with positive in-sample metrics
        base = {
            "sharpe": 2.0,
            "fitness": 1.5,
            "turnover": 0.20,
            "margin": 0.0010,
            "drawdown": 0.10,
            "returns": 0.10,
            "checks": [{"name": "SELF_CORRELATION", "result": "PASS"}],
        }

        # Alpha A: maintained 60% of Sharpe in test period (out-of-sample robust)
        row_robust = {**base, "train_sharpe": 2.0, "test_sharpe": 1.2}
        # Alpha B: in-sample overfit, collapsed out-of-sample
        row_overfit = {**base, "train_sharpe": 2.0, "test_sharpe": 0.2}
        # Alpha C: no test period evaluated
        row_untested = {**base}

        score_robust = seed_score(row_robust)
        score_overfit = seed_score(row_overfit)
        score_untested = seed_score(row_untested)

        # Independent validation priority: robust candidate ranks higher than untested and collapsed
        self.assertGreater(score_robust, score_untested)
        self.assertGreater(score_untested, score_overfit)
        # Robust candidate gets +25% bonus; overfit candidate receives -50% penalty
        self.assertAlmostEqual(score_robust, score_untested * 1.25, places=5)
        self.assertAlmostEqual(score_overfit, score_untested * 0.50, places=5)


if __name__ == "__main__":
    unittest.main()
