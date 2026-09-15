import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submodules" / "MoGe"))

from submodules.MoGe.moge.utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift


class MoGeSparseSupportTests(unittest.TestCase):
    def point_map(self):
        uv = normalized_view_plane_uv(128, 128)
        return torch.cat((uv * 10.0, torch.full((128, 128, 1), 8.0)), dim=-1).unsqueeze(0)

    def test_dense_known_camera_keeps_the_original_shift_solution(self):
        points = self.point_map()
        focal, shift = recover_focal_shift(points, torch.ones(1, 128, 128, dtype=torch.bool),
                                           focal=torch.ones(1))
        self.assertTrue(torch.allclose(focal, torch.ones(1)))
        self.assertTrue(torch.allclose(shift, torch.tensor([2.0]), atol=1e-4))

    def test_valid_pixels_missed_by_nearest_sampling_are_not_discarded(self):
        points = self.point_map()
        mask = torch.zeros(1, 128, 128, dtype=torch.bool)
        mask[:, 1::2, 1::2] = True
        focal, shift = recover_focal_shift(points, mask, focal=torch.ones(1))
        self.assertTrue(torch.allclose(focal, torch.ones(1)))
        self.assertTrue(torch.allclose(shift, torch.tensor([2.0]), atol=1e-4))

    def test_empty_support_is_explicitly_undefined_not_a_fabricated_shift(self):
        points = self.point_map().expand(2, -1, -1, -1)
        mask = torch.ones(2, 128, 128, dtype=torch.bool)
        mask[1] = False
        focal, shift = recover_focal_shift(points, mask, focal=torch.ones(2))
        self.assertTrue(torch.allclose(shift[:1], torch.tensor([2.0]), atol=1e-4))
        self.assertTrue(torch.isnan(shift[1]))
        self.assertTrue(torch.equal(focal, torch.ones(2)))
