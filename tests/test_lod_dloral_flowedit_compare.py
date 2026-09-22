"""CPU checks for the locked local DLoRAL / FlowEdit comparison."""

from __future__ import annotations

import unittest

from lod.dloral_flowedit_compare import (
    COMPARE_AZIMUTHS_DEG,
    COMPARE_CENTER,
    COMPARE_CHECKPOINT_STEPS,
    COMPARE_MAX_POINTS,
    COMPARE_MIX_RATIO,
    COMPARE_STEPS,
    CROP_BOXES,
    FLOWEDIT_TARGET_PROMPT,
    GROUPS,
    HELDOUT_AZIMUTHS_DEG,
    compare_poses,
    compare_protocol,
    flowedit_seed,
)


class LocalCompareSpecTests(unittest.TestCase):
    def test_center_and_views_are_locked(self) -> None:
        self.assertEqual(COMPARE_CENTER, (-128.0, 0.0, 0.0))
        self.assertEqual(COMPARE_AZIMUTHS_DEG, (0.0, 60.0, 120.0, 180.0, 240.0, 300.0))
        self.assertEqual(len(compare_poses()), 6)
        self.assertEqual(
            [row["azimuth_deg"] for row in compare_poses()],
            list(COMPARE_AZIMUTHS_DEG),
        )
        self.assertTrue(set(HELDOUT_AZIMUTHS_DEG).isdisjoint(COMPARE_AZIMUTHS_DEG))

    def test_three_arms_and_shared_seed_contract(self) -> None:
        self.assertEqual([group["id"] for group in GROUPS], ["A", "B", "C"])
        self.assertEqual(GROUPS[0]["output_kind"], "dloral")
        self.assertEqual(GROUPS[1]["output_kind"], "flowedit_l0")
        self.assertEqual(GROUPS[2]["input_kind"], "dloral")
        self.assertTrue(GROUPS[0]["shared_dloral"])
        self.assertTrue(GROUPS[2]["shared_dloral"])
        self.assertEqual(
            [flowedit_seed(value) for value in COMPARE_AZIMUTHS_DEG],
            [flowedit_seed(value) for value in COMPARE_AZIMUTHS_DEG],
        )

    def test_budget_and_prompt_are_fixed(self) -> None:
        self.assertEqual(COMPARE_STEPS, 500)
        self.assertEqual(COMPARE_MAX_POINTS, 50_000)
        self.assertEqual(COMPARE_MIX_RATIO, 0.2)
        self.assertEqual(COMPARE_CHECKPOINT_STEPS, (0, 50, 100, 250, 500))
        self.assertNotIn("white angular structure", FLOWEDIT_TARGET_PROMPT.lower())
        for box in CROP_BOXES.values():
            self.assertEqual(len(box), 4)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in box))

    def test_protocol_blocks_automatic_followup(self) -> None:
        protocol = compare_protocol()
        self.assertFalse(protocol["evaluation"]["automatic_followup"])
        self.assertFalse(protocol["evaluation"]["eight_x"] == "allowed")
        self.assertFalse(protocol["evaluation"]["fifty_four_view"] == "allowed")
        self.assertTrue(protocol["supervision"]["flowedit_resets_rng_before_each_image"])
        self.assertTrue(protocol["supervision"]["training_rng_isolated"])


if __name__ == "__main__":
    unittest.main()
