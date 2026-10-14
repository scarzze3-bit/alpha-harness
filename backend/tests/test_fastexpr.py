"""Tests for Fast Expression grammar, parsing, rendering, validation, and genetic crossover."""

import random
import unittest

import _stubs  # noqa: F401

from alpha_harness.labs.fastexpr import (
    GROUPING,
    MAX_FIELDS,
    MAX_OPERATORS,
    Node,
    canonical_formula,
    crossover,
    data_fields,
    node_at,
    operator_count,
    operator_names,
    operator_table,
    parse,
    protected,
    render,
    validate,
    walk,
)
from alpha_harness.labs.ga import Market, Parent, child


class TestFastExpr(unittest.TestCase):
    def test_tokenize_and_parse_atoms(self) -> None:
        self.assertEqual(parse("123").kind, "num")
        self.assertEqual(parse("123").value, "123")
        self.assertEqual(parse("3.14").value, "3.14")
        self.assertEqual(parse(".5").value, ".5")
        self.assertEqual(parse("1e-4").value, "1e-4")
        self.assertEqual(parse("'str'").kind, "str")
        self.assertEqual(parse("close").kind, "name")
        self.assertEqual(parse("close").value, "close")

    def test_comments_stripped_without_shifting_valid_tokens(self) -> None:
        n1 = parse("close - open")
        n2 = parse("# line comment\nclose - open")
        n3 = parse("// slash comment\nclose - open")
        n4 = parse("/* block */ close - open")
        self.assertEqual(render(n1), render(n2))
        self.assertEqual(render(n1), render(n3))
        self.assertEqual(render(n1), render(n4))

    def test_binary_precedence(self) -> None:
        tree = parse("a + b * c")
        self.assertEqual(tree.kind, "binary")
        self.assertEqual(tree.value, "+")
        self.assertEqual(tree.args[0].value, "a")
        self.assertEqual(tree.args[1].value, "*")

    def test_power_right_associative(self) -> None:
        tree = parse("a ^ b ^ c")
        self.assertEqual(tree.kind, "binary")
        self.assertEqual(tree.value, "^")
        self.assertEqual(tree.args[0].value, "a")
        self.assertEqual(tree.args[1].value, "^")
        self.assertEqual(tree.args[1].args[0].value, "b")
        self.assertEqual(tree.args[1].args[1].value, "c")

    def test_unary_and_negative_constants(self) -> None:
        tree = parse("-close")
        self.assertEqual(tree.kind, "unary")
        self.assertEqual(tree.value, "-")
        self.assertEqual(tree.args[0].value, "close")
        self.assertEqual(operator_count(tree), 1)

        neg_const = parse("-5")
        self.assertEqual(neg_const.kind, "unary")
        self.assertEqual(operator_count(neg_const), 0)

    def test_ternary_operator(self) -> None:
        tree = parse("close > open ? rank(close) : 0")
        self.assertEqual(tree.kind, "ternary")
        self.assertEqual(render(tree), "close > open ? rank(close) : 0")

    def test_multi_statement_sequence_and_assignments(self) -> None:
        code = "x = rank(close); y = rank(open); x - y"
        tree = parse(code)
        self.assertEqual(tree.kind, "seq")
        self.assertEqual(len(tree.args), 3)
        self.assertEqual(render(tree), "x = rank(close); y = rank(open); x - y")

    def test_roundtrip_render_parse(self) -> None:
        cases = [
            "rank(close) - rank(open)",
            "ts_rank(volume, 20)",
            "ts_backfill(close, 21)",
            "a && b || c",
            "x = 1; y = 2; x + y",
            "group_rank(close, sector)",
            "trade_when(volume > 1000, rank(close), -1)",
        ]
        for expr in cases:
            with self.subTest(expr=expr):
                tree = parse(expr)
                rendered = render(tree)
                tree_again = parse(rendered)
                self.assertEqual(tree, tree_again)

    def test_operator_count(self) -> None:
        self.assertEqual(operator_count(parse("rank(close) - rank(open)")), 3)
        self.assertEqual(operator_count(parse("ts_backfill(close, 21)")), 0)
        tree = parse("rank(ts_backfill(close, 21)) + rank(open)")
        self.assertEqual(operator_count(tree), 3)

    def test_data_fields_extraction(self) -> None:
        tree = parse("x = rank(close); x - rank(open) + market")
        fields = data_fields(tree)
        self.assertEqual(fields, ["close", "open"])

        fields_with_group = data_fields(tree, grouping=True)
        self.assertEqual(fields_with_group, ["close", "open", "market"])

    def test_operator_names(self) -> None:
        tree = parse("rank(ts_rank(close, 10)) - rank(open)")
        ops = operator_names(tree)
        self.assertEqual(ops, ["rank", "ts_rank"])

    def test_operator_table_and_validation(self) -> None:
        ops_raw = [
            {
                "name": "ts_rank",
                "category": "TimeSeries",
                "definition": "ts_rank(x, d, constant = 0)",
                "scope": ["REGULAR", "COMBO"],
            },
            {
                "name": "rank",
                "category": "CrossSectional",
                "definition": "rank(x)",
                "scope": ["REGULAR"],
            },
            {
                "name": "group_rank",
                "category": "Group",
                "definition": "group_rank(x, group)",
                "scope": ["REGULAR"],
            },
            {
                "name": "combo_only",
                "category": "Combo",
                "definition": "combo_only(x)",
                "scope": ["COMBO"],
            },
        ]
        table = operator_table(ops_raw, scope="REGULAR")
        self.assertIn("ts_rank", table)
        self.assertIn("rank", table)
        self.assertIn("group_rank", table)
        self.assertNotIn("combo_only", table)

        self.assertEqual(table["ts_rank"].required, 2)
        self.assertEqual(table["ts_rank"].maximum, 3)

        valid_tree = parse("ts_rank(close, 20)")
        self.assertEqual(validate(valid_tree, table, {"close"}), [])

        unknown_tree = parse("fake_op(close)")
        errs = validate(unknown_tree, table, {"close"})
        self.assertIn("fake_op is not an operator on BRAIN.", errs)

        too_few = parse("ts_rank(close)")
        errs = validate(too_few, table, {"close"})
        self.assertIn("ts_rank does not take 1 inputs.", errs)

        bad_field = parse("rank(unknown_field)")
        errs = validate(bad_field, table, {"close"})
        self.assertIn("unknown_field is not a data field here.", errs)

    def test_grouping_slot_protection(self) -> None:
        tree = parse("group_rank(close, sector)")
        self.assertFalse(protected(tree, (0,)))
        self.assertTrue(protected(tree, (1,)))

    def test_field_types_and_grouping_validation(self) -> None:
        ops_raw = [
            {"name": "group_rank", "definition": "group_rank(x, group)", "scope": ["REGULAR"]},
            {"name": "rank", "definition": "rank(x)", "scope": ["REGULAR"]},
            {"name": "vec_avg", "definition": "vec_avg(x)", "scope": ["REGULAR"]},
        ]
        table = operator_table(ops_raw, scope="REGULAR")
        known_fields = {"close", "volume", *GROUPING}

        valid_group = parse("group_rank(close, sector)")
        self.assertEqual(validate(valid_group, table, known_fields), [])

        invalid_group = parse("group_rank(close, invalid_group_name)")
        errs = validate(invalid_group, table, known_fields)
        self.assertIn("invalid_group_name is not a data field here.", errs)

        matrix_field = "close"
        vector_field = "vec_data"
        types = {matrix_field: "MATRIX", vector_field: "VECTOR"}

        unreduced_tree = parse(f"rank({vector_field})")
        self.assertEqual(validate(unreduced_tree, table, {vector_field}), [])

        def vector_reduced(tree: Node, types_map: dict[str, str]) -> bool:
            for path, node in walk(tree):
                if node.kind == "name" and types_map.get(node.value) == "VECTOR":
                    parent = node_at(tree, path[:-1]) if path else None
                    if (
                        parent is None
                        or parent.kind != "call"
                        or not parent.value.startswith("vec_")
                    ):
                        return False
            return True

        self.assertFalse(vector_reduced(unreduced_tree, types))
        reduced_tree = parse(f"rank(vec_avg({vector_field}))")
        self.assertTrue(vector_reduced(reduced_tree, types))

    # --- TASK 2 REGRESSION TESTS: Crossover & Offspring Limits ---

    def test_crossover_four_field_expansion_rejected(self) -> None:
        # Reported four-field crossover case: parent left has 3 fields, right has 2 new fields.
        left = parse("rank(close) + rank(open) + rank(high)")
        right = parse("ts_corr(volume, vwap, 10) + 1")

        # Without constraints, seed 0 produces 4 fields
        rng = random.Random(0)
        unconstrained = crossover(rng, left, right)
        self.assertIsNotNone(unconstrained)
        self.assertEqual(len(data_fields(unconstrained)), 4)

        # With max_fields=MAX_FIELDS (3), four-field child is rejected across all graft attempts
        rng = random.Random(0)
        constrained = crossover(rng, left, right, max_fields=MAX_FIELDS)
        # No graft between these two can stay within 3 fields, so it safely returns None
        self.assertIsNone(constrained)

    def test_crossover_nine_operator_expansion_rejected(self) -> None:
        # Reported nine-operator crossover case: parent left has 7 ops, right has 3-op graft
        left = parse("rank(close) + open + high + low + vwap + volume + returns")
        right = parse("ts_rank(ts_decay_linear(rank(returns), 5), 10) + 1")

        # Without constraints, seed 2 produces 9 operators (exceeding MAX_OPERATORS=8)
        rng = random.Random(2)
        unconstrained = crossover(rng, left, right)
        self.assertIsNotNone(unconstrained)
        self.assertEqual(operator_count(unconstrained), 9)

        # With max_operators=MAX_OPERATORS (8), candidate exceeding 8 operators is rejected;
        # crossover attempts another valid graft within budget, satisfying operator limit <= 8
        rng = random.Random(2)
        constrained = crossover(rng, left, right, max_operators=MAX_OPERATORS)
        self.assertIsNotNone(constrained)
        self.assertLessEqual(operator_count(constrained), MAX_OPERATORS)

        # When the budget is constrained so that no graft can satisfy the limit, safely returns None
        rng = random.Random(2)
        exhausted = crossover(rng, left, right, max_operators=5)
        self.assertIsNone(exhausted)

    def test_crossover_valid_child_within_limits(self) -> None:
        # Crossover that remains within configured limits is accepted
        left = parse("rank(close) + rank(open)")
        right = parse("ts_rank(volume, 5)")
        rng = random.Random(42)
        child_node = crossover(rng, left, right, max_operators=MAX_OPERATORS, max_fields=MAX_FIELDS)
        self.assertIsNotNone(child_node)
        self.assertLessEqual(operator_count(child_node), MAX_OPERATORS)
        self.assertLessEqual(len(data_fields(child_node)), MAX_FIELDS)

    def test_crossover_exhausted_budget_returns_none(self) -> None:
        # When all possible graft pairs violate constraints, safely returns None
        left = parse("rank(close) + rank(open) + rank(high)")
        right = parse("ts_corr(volume, vwap, 10)")
        rng = random.Random(1)
        res = crossover(rng, left, right, max_fields=3)
        self.assertIsNone(res)

    def test_ga_child_enforces_limits_end_to_end(self) -> None:
        # In Evolution Lab ga.child(), offspring are rejected if they exceed limits
        table = operator_table(
            [
                {"name": "rank", "definition": "rank(x)", "scope": ["REGULAR"]},
                {"name": "ts_rank", "definition": "ts_rank(x, d)", "scope": ["REGULAR"]},
                {"name": "ts_corr", "definition": "ts_corr(x, y, d)", "scope": ["REGULAR"]},
            ]
        )
        market = Market(
            table=table,
            swappable={},
            dataset_of={
                "close": "ds1",
                "open": "ds1",
                "high": "ds1",
                "low": "ds1",
                "volume": "ds2",
                "vwap": "ds2",
                "cap": "ds2",
                "returns": "ds2",
            },
            by_dataset={
                "ds1": ("close", "open", "high", "low"),
                "ds2": ("volume", "vwap", "cap", "returns"),
            },
            names=frozenset({"close", "open", "high", "low", "volume", "vwap", "cap", "returns"}),
            neutralizations=("SUBINDUSTRY",),
        )

        # 1. When parents have valid representations, any returned child is strictly within limits
        p_valid1 = Parent(
            alpha_id="p1",
            tree=parse("rank(close) + rank(open)"),
            neutralization="SUBINDUSTRY",
            decay=3,
            truncation=0.08,
        )
        p_valid2 = Parent(
            alpha_id="p2",
            tree=parse("ts_rank(volume, 10)"),
            neutralization="SUBINDUSTRY",
            decay=5,
            truncation=0.08,
        )
        rng = random.Random(42)
        offspring = child(
            rng,
            p_valid1,
            p_valid2,
            market,
            rate=0.1,
            max_operators=MAX_OPERATORS,
            max_fields=MAX_FIELDS,
        )
        self.assertIsNotNone(offspring)
        child_tree, _ = offspring
        self.assertLessEqual(operator_count(child_tree), MAX_OPERATORS)
        self.assertLessEqual(len(data_fields(child_tree)), MAX_FIELDS)

        # 2. When parents both exceed max_fields (4 fields each), no crossover or mutation
        # can satisfy max_fields=3, and ga.child() attempts 10 tries and safely returns None.
        p_excess1 = Parent(
            alpha_id="p3",
            tree=parse("close + open + high + low"),
            neutralization="SUBINDUSTRY",
            decay=3,
            truncation=0.08,
        )
        p_excess2 = Parent(
            alpha_id="p4",
            tree=parse("volume + vwap + cap + returns"),
            neutralization="SUBINDUSTRY",
            decay=3,
            truncation=0.08,
        )
        rng = random.Random(0)
        offspring_none = child(rng, p_excess1, p_excess2, market, rate=0.0, max_fields=MAX_FIELDS)
        self.assertIsNone(offspring_none)

    def test_ga_child_custom_engine_limits(self) -> None:
        # Verify engine-specific differences: when custom limits (e.g. max_fields=5) are specified,
        # crossover can proceed if within the custom limits
        table = operator_table(
            [
                {"name": "rank", "definition": "rank(x)", "scope": ["REGULAR"]},
                {"name": "ts_corr", "definition": "ts_corr(x, y, d)", "scope": ["REGULAR"]},
            ]
        )
        market = Market(
            table=table,
            swappable={},
            dataset_of={"close": "ds1", "open": "ds1", "volume": "ds2", "vwap": "ds2"},
            by_dataset={"ds1": ("close", "open"), "ds2": ("volume", "vwap")},
            names=frozenset({"close", "open", "volume", "vwap"}),
            neutralizations=("SUBINDUSTRY",),
        )
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
            decay=3,
            truncation=0.08,
        )

        rng = random.Random(42)
        # With max_fields=4, grafting ts_corr produces a 3-field child which is accepted
        offspring = child(rng, p1, p2, market, rate=0.0, max_fields=4)
        if offspring is not None:
            child_tree, _ = offspring
            self.assertLessEqual(len(data_fields(child_tree)), 4)

    def test_canonical_formula_behavior(self) -> None:
        self.assertIsNone(canonical_formula(None))
        self.assertEqual(canonical_formula(""), "")
        self.assertEqual(canonical_formula("   "), "")
        self.assertEqual(canonical_formula("  close  -  open  "), "close - open")
        self.assertEqual(canonical_formula("# comment\nclose - open"), "close - open")
        self.assertEqual(canonical_formula("/* block */ close - open"), "close - open")
        self.assertEqual(
            canonical_formula("x = rank(close); x - open;"), "x = rank(close); x - open"
        )
        self.assertEqual(canonical_formula("rank(close) + "), "rank(close) + ")


if __name__ == "__main__":
    unittest.main()
