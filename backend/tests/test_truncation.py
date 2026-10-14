"""Tests for market truncation rules."""

import unittest

from alpha_harness.labs.truncation import (
    BROAD,
    MID,
    MULTI_COUNTRY,
    NARROW,
    TIGHT,
    breadth,
    truncation_for,
)


class TestTruncation(unittest.TestCase):
    def test_benchmark_markets(self) -> None:
        cases = {
            ("USA", 1, "TOP3000"): 0.08,
            ("USA", 1, "TOP1000"): 0.06,
            ("USA", 1, "TOP200"): 0.05,
            ("USA", 1, "TOPSP500"): 0.05,
            ("USA", 0, "TOP3000"): 0.05,
            ("CHN", 1, "TOP2000U"): 0.08,
            ("JPN", 1, "TOP1600"): 0.06,
            ("EUR", 1, "TOPCS1600"): 0.05,
            ("GLB", 1, "MINVOL1M"): 0.05,
            ("GBR", 1, "TOP700"): 0.05,
            ("ALL", 1, "LARGE"): 0.05,
        }
        for (region, delay, universe), want in cases.items():
            with self.subTest(region=region, delay=delay, universe=universe):
                got = truncation_for(region, delay, universe)
                self.assertEqual(got, want, f"{region} D{delay} {universe} mismatch")

    def test_delay_zero_clamped(self) -> None:
        # Delay 0 markets must be clamped to at most TIGHT (0.05) regardless of universe breadth
        self.assertEqual(truncation_for("USA", 0, "TOP3000"), TIGHT)
        self.assertEqual(truncation_for("CHN", 0, "TOP2000U"), TIGHT)

    def test_multi_country_clamped(self) -> None:
        # Multi-country regions must be clamped to 0.05
        for region in MULTI_COUNTRY:
            with self.subTest(region=region):
                self.assertEqual(truncation_for(region, 1, "TOP3000"), TIGHT)

    def test_breadth_extraction(self) -> None:
        self.assertEqual(breadth("TOP3000"), 3000)
        self.assertEqual(breadth("TOPSP500"), 500)
        self.assertEqual(breadth("TOP2000U"), 2000)
        self.assertEqual(breadth("top1000"), 1000)
        self.assertIsNone(breadth("MINVOL1M"))
        self.assertIsNone(breadth("LARGE"))

    def test_case_insensitivity(self) -> None:
        # Truncation logic normalizes region and universe casing
        self.assertEqual(truncation_for("usa", 1, "top3000"), 0.08)
        self.assertEqual(truncation_for("glb", 1, "top3000"), TIGHT)

    def test_breadth_threshold_boundaries(self) -> None:
        # >= 2000 is BROAD (0.08), >= 1000 is MID (0.06), < 1000 is NARROW (0.05)
        self.assertEqual(truncation_for("USA", 1, "TOP2000"), BROAD)
        self.assertEqual(truncation_for("USA", 1, "TOP1999"), MID)
        self.assertEqual(truncation_for("USA", 1, "TOP1000"), MID)
        self.assertEqual(truncation_for("USA", 1, "TOP999"), NARROW)
        self.assertEqual(truncation_for("USA", 1, "TOP0"), NARROW)


if __name__ == "__main__":
    unittest.main()
