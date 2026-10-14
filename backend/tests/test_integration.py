"""Offline integration test covering formula generation, validation, request hashing,
and experiment deduplication without live BRAIN connectivity.
"""

import random
import unittest

from alpha_harness.engine.lifecycle import request_hash
from alpha_harness.labs.fastexpr import (
    MAX_FIELDS,
    MAX_OPERATORS,
    data_fields,
    operator_count,
    operator_table,
    parse,
    render,
    validate,
)
from alpha_harness.labs.ga import Market, Parent, child
from alpha_harness.labs.truncation import truncation_for


class OfflineMockRequest:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_wire(self) -> dict:
        return dict(self._data)


class TestOfflineIntegration(unittest.TestCase):
    def setUp(self) -> None:
        # Configure market metadata
        self.table = operator_table(
            [
                {"name": "rank", "definition": "rank(x)", "scope": ["REGULAR"]},
                {"name": "ts_rank", "definition": "ts_rank(x, d)", "scope": ["REGULAR"]},
                {"name": "ts_delta", "definition": "ts_delta(x, d)", "scope": ["REGULAR"]},
            ]
        )
        self.market = Market(
            table=self.table,
            swappable={},
            dataset_of={"close": "ds1", "open": "ds1", "volume": "ds2"},
            by_dataset={"ds1": ("close", "open"), "ds2": ("volume",)},
            names=frozenset({"close", "open", "volume"}),
            neutralizations=("SUBINDUSTRY", "MARKET"),
        )

    def test_pipeline_generate_validate_hash_dedup(self) -> None:
        # 1. Step: Parent seed formulation
        p1_expr = "rank(close) - rank(open)"
        p2_expr = "ts_rank(volume, 10)"

        p1_tree = parse(p1_expr)
        p2_tree = parse(p2_expr)

        # Validate seeds pass catalog validation
        self.assertEqual(validate(p1_tree, self.table, self.market.names), [])
        self.assertEqual(validate(p2_tree, self.table, self.market.names), [])

        trunc = truncation_for("USA", 1, "TOP3000")
        self.assertEqual(trunc, 0.08)

        p1 = Parent(
            alpha_id="p1",
            tree=p1_tree,
            neutralization="SUBINDUSTRY",
            decay=3,
            truncation=trunc,
        )
        p2 = Parent(
            alpha_id="p2",
            tree=p2_tree,
            neutralization="SUBINDUSTRY",
            decay=5,
            truncation=trunc,
        )

        # 2. Step: Genetic Breeding with strict constraints
        rng = random.Random(42)
        offspring = child(
            rng, p1, p2, self.market, rate=0.1, max_operators=MAX_OPERATORS, max_fields=MAX_FIELDS
        )
        self.assertIsNotNone(offspring)
        child_tree, child_settings = offspring

        # 3. Step: Verify offspring satisfies all constraints
        self.assertLessEqual(operator_count(child_tree), MAX_OPERATORS)
        self.assertLessEqual(len(data_fields(child_tree)), MAX_FIELDS)
        self.assertEqual(validate(child_tree, self.table, self.market.names), [])

        # 4. Step: Simulation request creation & hashing
        child_expr_str = render(child_tree)
        wire_data = {
            "regular": child_expr_str,
            "settings": {
                "region": "USA",
                "delay": 1,
                "universe": "TOP3000",
                "neutralization": child_settings["neutralization"],
                "decay": child_settings["decay"],
                "truncation": child_settings["truncation"],
                "simulationMode": "FULL",
            },
        }
        req = OfflineMockRequest(wire_data)
        hash1 = request_hash(req)

        # 5. Step: Deduplication invariance check with comments and formatting
        messy_wire_data = {
            "regular": f"\n  # evolved offspring\n  {child_expr_str}  \n",
            "settings": {
                "neutralization": child_settings["neutralization"],
                "decay": child_settings["decay"],
                "truncation": child_settings["truncation"],
                "region": "USA",
                "delay": 1,
                "universe": "TOP3000",
            },
        }
        messy_req = OfflineMockRequest(messy_wire_data)
        hash2 = request_hash(messy_req)

        # Canonicalization ensures exact deduplication hash match
        self.assertEqual(hash1, hash2)

        # 6. Step: Deduplication simulation queue check (mock database lookup)
        simulated_cache = {hash1: "alpha_id_12345"}
        self.assertIn(hash2, simulated_cache)
        self.assertEqual(simulated_cache[hash2], "alpha_id_12345")


if __name__ == "__main__":
    unittest.main()
