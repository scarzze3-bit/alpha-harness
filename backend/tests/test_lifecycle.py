"""Tests for lifecycle simulation request hashing and deduplication."""

import unittest

from alpha_harness.engine.lifecycle import request_hash
from alpha_harness.labs.fastexpr import canonical_formula


class MockWireRequest:
    """Mock simulation request implementing the to_wire() protocol."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def to_wire(self) -> dict:
        return dict(self._data)


class TestLifecycleDedup(unittest.TestCase):
    def test_key_order_does_not_change_canonical_payload(self) -> None:
        # Dictionary key ordering in settings or root payload must not change the hash
        r1 = MockWireRequest({"regular": "close - open", "settings": {"region": "USA", "delay": 1}})
        r2 = MockWireRequest({"settings": {"delay": 1, "region": "USA"}, "regular": "close - open"})
        self.assertEqual(request_hash(r1), request_hash(r2))

    def test_full_mode_normalization(self) -> None:
        # Default FULL simulationMode is deleted from settings before hashing so runs match
        r1 = MockWireRequest(
            {
                "regular": "close - open",
                "settings": {"region": "USA", "delay": 1, "simulationMode": "FULL"},
            }
        )
        r2 = MockWireRequest(
            {
                "regular": "close - open",
                "settings": {"region": "USA", "delay": 1},
            }
        )
        self.assertEqual(request_hash(r1), request_hash(r2))

    def test_equivalent_whitespace_and_comments_yield_identical_hashes(self) -> None:
        # Canonicalization ensures formatting variations hash to the exact same identity
        base_req = MockWireRequest(
            {"regular": "close - open", "settings": {"region": "USA", "delay": 1}}
        )
        space_req = MockWireRequest(
            {"regular": "  close   -   open  ", "settings": {"region": "USA", "delay": 1}}
        )
        comment_req = MockWireRequest(
            {
                "regular": "# calculate spread\nclose - open",
                "settings": {"region": "USA", "delay": 1},
            }
        )
        block_comment_req = MockWireRequest(
            {
                "regular": "/* block comment */ close - open",
                "settings": {"region": "USA", "delay": 1},
            }
        )

        base_hash = request_hash(base_req)
        self.assertEqual(request_hash(space_req), base_hash)
        self.assertEqual(request_hash(comment_req), base_hash)
        self.assertEqual(request_hash(block_comment_req), base_hash)

    def test_empty_and_whitespace_formula_deduplication(self) -> None:
        r_empty = MockWireRequest({"regular": "", "settings": {"region": "USA", "delay": 1}})
        r_spaces = MockWireRequest({"regular": "   ", "settings": {"region": "USA", "delay": 1}})
        self.assertEqual(request_hash(r_empty), request_hash(r_spaces))

    def test_different_formulas_yield_different_hashes(self) -> None:
        r1 = MockWireRequest({"regular": "close - open", "settings": {"region": "USA", "delay": 1}})
        r2 = MockWireRequest({"regular": "open - close", "settings": {"region": "USA", "delay": 1}})
        r3 = MockWireRequest(
            {"regular": "ts_rank(close, 10)", "settings": {"region": "USA", "delay": 1}}
        )
        self.assertNotEqual(request_hash(r1), request_hash(r2))
        self.assertNotEqual(request_hash(r1), request_hash(r3))

    def test_behavior_affecting_settings_yield_different_hashes(self) -> None:
        base = {
            "regular": "close - open",
            "settings": {
                "region": "USA",
                "delay": 1,
                "universe": "TOP3000",
                "neutralization": "SUBINDUSTRY",
                "decay": 3,
                "truncation": 0.08,
            },
        }
        r_base = MockWireRequest(base)

        # Region
        r_region = MockWireRequest(
            {
                "regular": "close - open",
                "settings": {**base["settings"], "region": "EUR"},
            }
        )
        self.assertNotEqual(request_hash(r_base), request_hash(r_region))

        # Delay
        r_delay = MockWireRequest(
            {
                "regular": "close - open",
                "settings": {**base["settings"], "delay": 0},
            }
        )
        self.assertNotEqual(request_hash(r_base), request_hash(r_delay))

        # Universe
        r_univ = MockWireRequest(
            {
                "regular": "close - open",
                "settings": {**base["settings"], "universe": "TOP1000"},
            }
        )
        self.assertNotEqual(request_hash(r_base), request_hash(r_univ))

        # Neutralization
        r_neut = MockWireRequest(
            {
                "regular": "close - open",
                "settings": {**base["settings"], "neutralization": "MARKET"},
            }
        )
        self.assertNotEqual(request_hash(r_base), request_hash(r_neut))

        # Decay
        r_decay = MockWireRequest(
            {
                "regular": "close - open",
                "settings": {**base["settings"], "decay": 10},
            }
        )
        self.assertNotEqual(request_hash(r_base), request_hash(r_decay))

        # Truncation
        r_trunc = MockWireRequest(
            {
                "regular": "close - open",
                "settings": {**base["settings"], "truncation": 0.05},
            }
        )
        self.assertNotEqual(request_hash(r_base), request_hash(r_trunc))

    def test_assignment_and_multi_statement_distinction(self) -> None:
        # Formula statements must preserve operator logic and assignment structure
        r_seq1 = MockWireRequest(
            {
                "regular": "x = rank(close); x - rank(open)",
                "settings": {"region": "USA", "delay": 1},
            }
        )
        r_seq2 = MockWireRequest(
            {
                "regular": "x = rank(close); x + rank(open)",
                "settings": {"region": "USA", "delay": 1},
            }
        )
        r_seq3 = MockWireRequest(
            {
                "regular": "x = rank(close); y = rank(open); x - y",
                "settings": {"region": "USA", "delay": 1},
            }
        )
        r_single = MockWireRequest(
            {
                "regular": "rank(close) - rank(open)",
                "settings": {"region": "USA", "delay": 1},
            }
        )

        self.assertNotEqual(request_hash(r_seq1), request_hash(r_seq2))
        self.assertNotEqual(request_hash(r_seq1), request_hash(r_seq3))
        self.assertNotEqual(request_hash(r_seq1), request_hash(r_single))

    def test_invalid_formula_error_policy(self) -> None:
        # Malformed formulas must NOT be silently normalized or converted; they remain raw
        invalid_raw = "rank(close) + "
        self.assertEqual(canonical_formula(invalid_raw), invalid_raw)

        # Hashing an invalid formula uses the raw string deterministically
        r_invalid1 = MockWireRequest(
            {"regular": invalid_raw, "settings": {"region": "USA", "delay": 1}}
        )
        r_invalid2 = MockWireRequest(
            {"regular": invalid_raw, "settings": {"region": "USA", "delay": 1}}
        )
        self.assertEqual(request_hash(r_invalid1), request_hash(r_invalid2))

        # None or non-string passes through untouched
        self.assertIsNone(canonical_formula(None))


if __name__ == "__main__":
    unittest.main()
