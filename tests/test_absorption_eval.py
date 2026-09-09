#!/usr/bin/env python3
"""CPU checks for SH-only absorption bookkeeping."""

from __future__ import annotations

import unittest

from train_zoom_gen import _fractional_crop, _view_means, parse_eval_steps


class AbsorptionEvalTests(unittest.TestCase):
    def test_parse_eval_steps_sorts_and_deduplicates(self) -> None:
        self.assertEqual(parse_eval_steps(""), [])
        self.assertEqual(parse_eval_steps("500,0,50,50,100,250"), [0, 50, 100, 250, 500])

    def test_parse_eval_steps_rejects_negatives(self) -> None:
        with self.assertRaises(ValueError):
            parse_eval_steps("-1,0")

    def test_fractional_crop_stays_in_bounds(self) -> None:
        self.assertEqual(_fractional_crop(2048, 2048, (0.04, 0.20, 0.44, 0.59)), (82, 410, 901, 1208))
        x0, y0, x1, y1 = _fractional_crop(8, 8, (0.0, 0.0, 1.0, 1.0))
        self.assertEqual((x0, y0, x1, y1), (0, 0, 8, 8))

    def test_view_means(self) -> None:
        stats = _view_means(
            {
                "a": {"l1_to_gt": 0.2, "hf_l1_to_gt": 0.4},
                "b": {"l1_to_gt": 0.4, "hf_l1_to_gt": 0.6},
            }
        )
        self.assertEqual(stats["count"], 2)
        self.assertAlmostEqual(stats["l1_to_gt_mean"], 0.3)
        self.assertAlmostEqual(stats["hf_l1_to_gt_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
