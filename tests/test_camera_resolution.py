"""Resized training RGB must remain aligned with its mask and depth."""
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

from scene.dataset_readers import CameraInfo
from utils.camera_utils import loadCam


@unittest.skipUnless(torch.cuda.is_available(), "Skyfall Camera matrices require CUDA")
class CameraResolutionTests(unittest.TestCase):
    def test_masked_training_targets_after_downsampling(self):
        mask = np.zeros((24, 32), dtype=np.float32)
        mask[:, 16:] = 1.0
        depth = np.full((24, 32), 12.0, dtype=np.float32)
        depth[:, :16] = 3.0
        info = CameraInfo(
            uid=0, R=np.eye(3), T=np.zeros(3), FovX=0.6, FovY=0.5,
            cx=0.0, cy=0.0, image=Image.new("RGB", (32, 24), (255, 255, 255)),
            image_path="unused.png", image_name="view", depth=depth, mask=mask,
            width=32, height=24,
        )
        camera = loadCam(SimpleNamespace(resolution=2, data_device="cpu"), 0, info, 1)
        masked_rgb = camera.original_mask * camera.original_image
        masked_depth = camera.original_mask * camera.original_depth
        expected_rgb = torch.zeros(3, 12, 16)
        expected_rgb[:, :, 8:] = 1.0
        expected_depth = torch.zeros(1, 12, 16)
        expected_depth[:, :, 8:] = 12.0
        torch.testing.assert_close(masked_rgb, expected_rgb)
        torch.testing.assert_close(masked_depth, expected_depth)


if __name__ == "__main__":
    unittest.main()
