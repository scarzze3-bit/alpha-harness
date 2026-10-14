"""Tests verifying genetic algorithm search boundaries, complexity limits, crossover,
mutation fallback, and multi-generation diversity."""

import random
import unittest

from alpha_harness.labs.fastexpr import (
    MAX_FIELDS,
    MAX_OPERATORS,
    crossover,
    data_fields,
    operator_count,
    operator_table,
    parse,
    render,
    validate,
)
from alpha_harness.labs.ga import Market, Parent, child


class TestGeneticBoundaries(unittest.TestCase):
    def setUp(self) -> None:
        self.table = operator_table(
            [
                {"name": "rank", "definition": "rank(x)", "scope": ["REGULAR"]},
                {"name": "ts_rank", "definition": "ts_rank(x, d)", "scope": ["REGULAR"]},
                {"name": "ts_delta", "definition": "ts_delta(x, d)", "scope": ["REGULAR"]},
                {"name": "ts_corr", "definition": "ts_corr(x, y, d)", "scope": ["REGULAR"]},
                {"name": "abs", "definition": "abs(x)", "scope": ["REGULAR"]},
                {"name": "log", "definition": "log(x)", "scope": ["REGULAR"]},
                {"name": "signed_power", "definition": "signed_power(x, y)", "scope": ["REGULAR"]},
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
                "returns": "ds1",
            },
            by_dataset={
                "ds1": ("close", "open", "high", "low", "returns"),
                "ds2": ("volume", "vwap"),
            },
            names=frozenset({"close", "open", "high", "low", "volume", "vwap", "returns"}),
            neutralizations=("SUBINDUSTRY", "MARKET", "INDUSTRY"),
        )
        self.settings = {"neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.08}

    def test_parents_at_and_near_complexity_limits(self) -> None:
        """Parents at 8-op / 3-field limit produce children that strictly observe limits."""
        # Parent A: exactly 8 operators, 3 fields
        expr_a = "rank(ts_delta(close, 5)) - rank(open) + rank(close) - rank(volume)"
        tree_a = parse(expr_a)
        ops_a = operator_count(tree_a)
        fields_a = len(data_fields(tree_a))
        self.assertEqual(ops_a, 8)
        self.assertEqual(fields_a, 3)

        # Parent B: 7 operators, 3 fields
        expr_b = "rank(close) - rank(open) + rank(volume) - rank(close)"
        tree_b = parse(expr_b)
        ops_b = operator_count(tree_b)
        fields_b = len(data_fields(tree_b))
        self.assertEqual(ops_b, 7)
        self.assertEqual(fields_b, 3)

        parent_a = Parent.of("a", expr_a, self.settings)
        parent_b = Parent.of("b", expr_b, self.settings)
        self.assertIsNotNone(parent_a)
        self.assertIsNotNone(parent_b)

        rng = random.Random(42)
        for trial in range(25):
            res = child(rng, parent_a, parent_b, self.market, rate=0.2)
            if res is not None:
                c_tree, _ = res
                c_ops = operator_count(c_tree)
                c_fields = len(data_fields(c_tree))
                self.assertLessEqual(
                    c_ops,
                    MAX_OPERATORS,
                    f"Child ops ({c_ops}) > {MAX_OPERATORS} in trial {trial}",
                )
                self.assertLessEqual(
                    c_fields,
                    MAX_FIELDS,
                    f"Child fields ({c_fields}) > {MAX_FIELDS} in trial {trial}",
                )
                # Validation check against market: empty list means valid
                val_err = validate(c_tree, self.market.table, self.market.names)
                self.assertEqual(val_err, [], f"Generated child failed validation: {val_err}")

    def test_parents_with_different_field_counts(self) -> None:
        """Breeding parents with 1 vs 3 fields produces children that never exceed MAX_FIELDS."""
        expr_1f = "rank(close) + abs(close)"
        expr_3f = "rank(close) - rank(open) + rank(volume)"

        p1 = Parent.of("p1", expr_1f, self.settings)
        p3 = Parent.of("p3", expr_3f, self.settings)
        self.assertIsNotNone(p1)
        self.assertIsNotNone(p3)

        rng = random.Random(101)
        children_seen = set()
        for _ in range(20):
            res = child(rng, p1, p3, self.market, rate=0.2)
            if res is not None:
                c_tree, _ = res
                fields = data_fields(c_tree)
                self.assertLessEqual(len(fields), MAX_FIELDS)
                self.assertGreaterEqual(len(fields), 1)
                children_seen.add(render(c_tree))

        self.assertGreater(len(children_seen), 1, "Must produce diverse children")

    def test_crossover_rejection_and_mutation_fallback(self) -> None:
        """When every crossover graft exceeds limits, crossover rejects with None and child()
        falls back to mutation."""
        expr_a = "rank(ts_delta(close, 5))"  # 2 ops
        expr_b = "rank(ts_delta(open, 10))"  # 2 ops
        tree_a = parse(expr_a)
        tree_b = parse(expr_b)

        # Force crossover with max_operators=1 -> impossible because minimum call has 1 or 2 ops
        rng = random.Random(7)
        crossover_res = crossover(rng, tree_a, tree_b, max_operators=1)
        self.assertIsNone(crossover_res, "Crossover must reject when graft exceeds limits")

        # In child(), crossover rejection falls back to left.tree and mutates it
        p_a = Parent.of("a", expr_a, self.settings)
        p_b = Parent.of("b", expr_b, self.settings)
        self.assertIsNotNone(p_a)
        self.assertIsNotNone(p_b)

        # child() with max_operators=2
        res = child(rng, p_a, p_b, self.market, rate=0.5, max_operators=2)
        if res is not None:
            c_tree, _ = res
            self.assertLessEqual(operator_count(c_tree), 2)
            self.assertEqual(validate(c_tree, self.market.table, self.market.names), [])

    def test_small_parents_grow_up_to_max_operators(self) -> None:
        """Small parents (e.g. 1 operator each) are not artificially capped at 1 operator;
        they can evolve up to configured MAX_OPERATORS."""
        p_small1 = Parent.of("s1", "rank(close)", self.settings)
        p_small2 = Parent.of("s2", "rank(open)", self.settings)
        self.assertIsNotNone(p_small1)
        self.assertIsNotNone(p_small2)

        rng = random.Random(42)
        max_ops_observed = 1
        for _ in range(30):
            res = child(rng, p_small1, p_small2, self.market, rate=0.5)
            if res is not None:
                c_tree, _ = res
                ops = operator_count(c_tree)
                self.assertLessEqual(ops, MAX_OPERATORS)
                if ops > max_ops_observed:
                    max_ops_observed = ops

        self.assertLessEqual(max_ops_observed, MAX_OPERATORS)

    def test_bounded_retries_and_unavoidable_clone_returns_none(self) -> None:
        """When complexity limit cannot be satisfied (e.g. max_operators=0), child() retries
        up to 10 times and cleanly returns None without an infinite loop."""
        p1 = Parent.of("id1", "rank(close)", self.settings)
        p2 = Parent.of("id2", "rank(open)", self.settings)
        self.assertIsNotNone(p1)
        self.assertIsNotNone(p2)

        rng = random.Random(999)
        # With max_operators=0, every candidate has operator_count >= 1 > 0, so all 10 tries fail
        res = child(rng, p1, p2, self.market, rate=0.2, max_operators=0)
        self.assertIsNone(res, "Must return None after 10 retries fail constraint")

    def test_deterministic_multi_generation_breeding(self) -> None:
        """Simulate 5 generations of breeding from a small seed pool. Verify:
        1. All offspring strictly obey MAX_OPERATORS (8) and MAX_FIELDS (3).
        2. All offspring pass semantic operator validation.
        3. Population maintains diversity without duplicate collapse or explosion."""
        seed_exprs = [
            "rank(close) - rank(open)",
            "ts_delta(high, 5)",
            "rank(volume) - rank(vwap)",
            "ts_rank(close, 10)",
            "abs(open - close)",
            "rank(low) + ts_delta(volume, 2)",
        ]
        population: list[Parent] = [
            p
            for i, expr in enumerate(seed_exprs)
            if (p := Parent.of(f"seed_{i}", expr, self.settings)) is not None
        ]
        self.assertEqual(len(population), 6)

        rng = random.Random(1337)
        total_children = 0
        all_child_expressions = set()

        for gen in range(5):
            new_generation: list[Parent] = []
            for _ in range(8):
                p_a, p_b = rng.sample(population, 2)
                res = child(rng, p_a, p_b, self.market, rate=0.25)
                if res is not None:
                    c_tree, c_settings = res
                    c_ops = operator_count(c_tree)
                    c_fields = len(data_fields(c_tree))

                    # Strict invariant checks
                    self.assertLessEqual(
                        c_ops,
                        MAX_OPERATORS,
                        f"Gen {gen} child operator count {c_ops} > {MAX_OPERATORS}",
                    )
                    self.assertLessEqual(
                        c_fields,
                        MAX_FIELDS,
                        f"Gen {gen} child field count {c_fields} > {MAX_FIELDS}",
                    )
                    val_err = validate(c_tree, self.market.table, self.market.names)
                    self.assertEqual(val_err, [], f"Gen {gen} child failed validation: {val_err}")

                    c_expr = render(c_tree)
                    all_child_expressions.add(c_expr)
                    total_children += 1

                    new_p = Parent(
                        alpha_id=f"gen{gen}_{total_children}",
                        tree=c_tree,
                        neutralization=c_settings["neutralization"],
                        decay=c_settings["decay"],
                        truncation=c_settings["truncation"],
                    )
                    new_generation.append(new_p)

            # Advance population with elitism and new children
            population = population[:2] + new_generation

        self.assertGreaterEqual(total_children, 25, "Must produce stream of offspring")
        self.assertGreaterEqual(
            len(all_child_expressions),
            15,
            "Must maintain rich formula diversity across generations",
        )


if __name__ == "__main__":
    unittest.main()
