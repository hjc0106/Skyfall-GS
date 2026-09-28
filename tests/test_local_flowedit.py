"""Behavioral tests for the localized FlowEdit module and geometry invalidation.

Only editor-free paths are exercised (crop planning, feathered compositing,
zero-mask short-circuit, containment defenses, target-side geometry
invalidation); no FLUX load, no GPU, no network.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from unittest.mock import patch
import numpy as np
from PIL import Image

from refinement.local_flowedit import (
    CROP_ALIGN,
    LocalFlowEditConfig,
    LocalFlowEditError,
    compose_local_edit,
    dilate_support,
    extract_crop,
    feather_alpha,
    invalidate_target_side_geometry,
    plan_local_crop,
    refine_local_flowedit_jobs,
)


def _gradient_image(height, width, seed=7):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return base


def _rect_support(shape, top, bottom, left, right):
    support = np.zeros(shape, dtype=bool)
    support[top:bottom, left:right] = True
    return support


class ConfigTests(unittest.TestCase):

    def test_invalid_window_rejected(self):
        with self.assertRaises(LocalFlowEditError):
            LocalFlowEditConfig.from_dict({"n_min": 10, "n_max": 4})

    def test_negative_padding_rejected(self):
        with self.assertRaises(LocalFlowEditError):
            LocalFlowEditConfig.from_dict({"context_padding": -1})

    def test_unknown_model_type_rejected(self):
        with self.assertRaises(LocalFlowEditError):
            LocalFlowEditConfig.from_dict({"model_type": "XL"})



class CropPlanTests(unittest.TestCase):
    def test_plan_contains_support_with_context_and_aligns_to_16(self):
        support = _rect_support((80, 100), 20, 40, 30, 50)
        plan = plan_local_crop(support, context_padding=10)
        left, top, right, bottom = plan["box"]
        width, height = plan["padded_size"]
        self.assertEqual((width, height), (48, 64))
        self.assertEqual(width % CROP_ALIGN, 0)
        self.assertEqual(height % CROP_ALIGN, 0)
        # The aligned box keeps the requested padding around the support.
        self.assertLessEqual(left, 30 - 10)
        self.assertGreaterEqual(right, 50 + 10)
        self.assertLessEqual(top, 20 - 10)
        self.assertGreaterEqual(bottom, 40 + 10)
        self.assertEqual(plan["pad"], [0, 0, 0, 0])
        self.assertIsNone(plan["resample"])

    def test_extracted_crop_contains_support_context(self):
        support = _rect_support((80, 100), 20, 40, 30, 50)
        plan = plan_local_crop(support, context_padding=10)
        image = _gradient_image(80, 100)
        crop = extract_crop(image, plan)
        left, top, right, bottom = plan["box"]
        self.assertEqual(crop.shape, (plan["padded_size"][1], plan["padded_size"][0], 3))
        self.assertTrue(np.array_equal(crop[20 - top:40 - top, 30 - left:50 - left],
                                       image[20:40, 30:50]))

    def test_border_support_pads_outwards_and_records_padding(self):
        height, width = 50, 70  # neither dimension is a multiple of 16
        support = _rect_support((height, width), 40, 50, 60, 70)
        plan = plan_local_crop(support, context_padding=8)
        crop_w, crop_h = plan["padded_size"]
        self.assertEqual(crop_w % CROP_ALIGN, 0)
        self.assertEqual(crop_h % CROP_ALIGN, 0)
        original = _gradient_image(height, width)
        crop = extract_crop(original, plan)
        self.assertEqual(crop.shape, (crop_h, crop_w, 3))
        left, top, right, bottom = plan["box"]
        self.assertTrue(np.array_equal(crop[:bottom-top, :right-left], original[top:bottom, left:right]))

    def test_over_large_crop_is_resampled_with_recorded_scale(self):
        support = _rect_support((2000, 2000), 0, 1500, 0, 1500)
        plan = plan_local_crop(support, context_padding=64, max_crop_pixels=1 << 20)
        resample = plan["resample"]
        self.assertIsNotNone(resample)
        self.assertLess(resample["scale"], 1.0)
        new_w, new_h = plan["padded_size"]
        self.assertLessEqual(new_w * new_h, 1 << 20)
        self.assertEqual(new_w % CROP_ALIGN, 0)
        self.assertEqual(new_h % CROP_ALIGN, 0)
        self.assertEqual(resample["source_crop"],
                         [(plan["box"][2] - plan["box"][0]),
                          (plan["box"][3] - plan["box"][1])])

    def test_empty_support_is_rejected(self):
        support = np.zeros((32, 32), dtype=bool)
        with self.assertRaises(LocalFlowEditError):
            plan_local_crop(support, context_padding=8)

    def test_non_boolean_support_rejected(self):
        with self.assertRaises(LocalFlowEditError):
            plan_local_crop(np.zeros((32, 32), dtype=np.uint8), context_padding=8)


class CompositingTests(unittest.TestCase):
    def test_radius_zero_composite_is_byte_identical_outside_support(self):
        height, width = 80, 100
        image = _gradient_image(height, width)
        support = _rect_support((height, width), 20, 40, 30, 50)
        plan = plan_local_crop(support, context_padding=10)
        left, top, right, bottom = plan["box"]
        refined_crop = np.full((bottom - top, right - left, 3), (0, 255, 0), np.uint8)
        alpha = feather_alpha(support, feather_radius=0)
        self.assertEqual(int((alpha > 0).sum()), int(support.sum()))
        composite = compose_local_edit(image, refined_crop, plan, alpha)
        outside = ~support
        self.assertTrue(np.array_equal(composite[outside], image[outside]))
        self.assertTrue((composite[support] == (0, 255, 0)).all())

    def test_feathered_alpha_is_exactly_zero_outside_mask_support(self):
        support = _rect_support((80, 100), 20, 40, 30, 50)
        alpha = feather_alpha(support, feather_radius=3)
        self.assertTrue(np.all(alpha[~support] == 0.0))
        self.assertGreater(alpha[support].min(), 0.0)

    def test_feathered_composite_never_leaks_outside_support(self):
        height, width = 80, 100
        image = _gradient_image(height, width)
        support = _rect_support((height, width), 20, 40, 30, 50)
        plan = plan_local_crop(support, context_padding=10)
        left, top, right, bottom = plan["box"]
        refined_crop = np.full((bottom - top, right - left, 3), (0, 255, 0), np.uint8)
        alpha = feather_alpha(support, feather_radius=4)
        composite = compose_local_edit(image, refined_crop, plan, alpha)
        diff = (composite.astype(int) - image.astype(int)).any(axis=2)
        self.assertFalse(diff[~support].any(), "edit leaked outside the mask support")
        self.assertTrue(diff[support].any(), "mask interior was not edited")

    def test_alpha_shape_mismatch_rejected(self):
        image = _gradient_image(32, 32)
        plan = plan_local_crop(_rect_support((32, 32), 8, 16, 8, 16), 4)
        with self.assertRaises(LocalFlowEditError):
            compose_local_edit(image, np.zeros((8, 8, 3), np.uint8), plan,
                               np.zeros((31, 31)))

    def test_refined_crop_size_mismatch_rejected_without_resample(self):
        image = _gradient_image(80, 100)
        support = _rect_support((80, 100), 20, 40, 30, 50)
        plan = plan_local_crop(support, context_padding=10)
        alpha = feather_alpha(support, 0)
        with self.assertRaises(LocalFlowEditError):
            compose_local_edit(image, np.zeros((7, 7, 3), np.uint8), plan, alpha)


class ZeroMaskSkipTests(unittest.TestCase):
    def test_zero_mask_job_skips_editor_with_byte_identical_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "in.png"
            mask_path = root / "mask.png"
            output_path = root / "out.png"
            metadata_path = root / "meta.json"
            Image.fromarray(_gradient_image(48, 64)).save(image_path)
            Image.new("L", (64, 48), 0).save(mask_path)
            job = {"image_path": str(image_path), "mask_path": str(mask_path),
                   "output_path": str(output_path), "metadata_path": str(metadata_path)}
            # No weights_path anywhere: if the editor were invoked this raises.
            results = refine_local_flowedit_jobs([job], {"n_min": 0, "n_max": 5})
            self.assertEqual(results[0]["status"], "skipped")
            self.assertFalse(results[0]["metadata"]["editor_invoked"])
            self.assertEqual(
                hashlib.sha256(output_path.read_bytes()).hexdigest(),
                hashlib.sha256(image_path.read_bytes()).hexdigest(),
            )
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(metadata["status"], "skipped")
            self.assertEqual(metadata["edit_support"]["mask_pixels"], 0)

    def test_nonempty_batch_writes_real_png_and_returns_result(self):
        class Editor:
            pipe = None

            def run(self, images, **kwargs):
                return [Image.new("RGB", (images[0].shape[1], images[0].shape[0]), (240, 20, 60))]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = _gradient_image(48, 64)
            support = _rect_support((48, 64), 10, 30, 20, 40)
            Image.fromarray(original).save(root / "input.png")
            Image.fromarray(support.astype(np.uint8) * 255).save(root / "mask.png")
            job = {"image_path": root / "input.png", "mask_path": root / "mask.png",
                   "output_path": root / "new/output.png", "metadata_path": root / "new/result.json"}
            with patch("refinement.local_flowedit._load_editor", return_value=(Editor(), "source", "target")):
                results = refine_local_flowedit_jobs([job], {"feather_radius": 0, "context_padding": 0})
            self.assertEqual(results[0]["status"], "edited")
            with Image.open(job["output_path"]) as result:
                self.assertEqual(result.format, "PNG")
                pixels = np.array(result)
            self.assertTrue(np.array_equal(pixels[~support], original[~support]))
            self.assertTrue((pixels[support] == (240, 20, 60)).all())

    def test_mask_image_size_mismatch_rejected_before_any_editor_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "in.png"
            Image.fromarray(_gradient_image(48, 64)).save(image_path)
            bad_mask = root / "bad.png"
            Image.new("L", (32, 32), 255).save(bad_mask)
            job = {"image_path": str(image_path), "mask_path": str(bad_mask),
                   "output_path": str(root / "out.png"),
                   "metadata_path": str(root / "meta.json")}
            with self.assertRaises(LocalFlowEditError):
                refine_local_flowedit_jobs([job], {"n_min": 0, "n_max": 5})

    def test_missing_paths_rejected(self):
        with self.assertRaises(LocalFlowEditError):
            refine_local_flowedit_jobs([{"mask_path": "a", "output_path": "b",
                                         "metadata_path": "c"}], {})


class GeometryInvalidationTests(unittest.TestCase):
    def test_target_side_validity_zeroed_exactly_in_dilated_support(self):
        flow = np.full((8, 10, 2), -2.5, np.float32)
        flow[..., 0] = 1.5
        valid = np.ones((8, 10), bool)
        support = _rect_support((8, 10), 2, 5, 3, 6)
        result = invalidate_target_side_geometry(flow, valid, support,
                                                 dilation_radius=1)
        new_flow, new_valid = result["pixel_flow"], result["valid_mask"]
        self.assertEqual(result["support_pixels"], int(support.sum()))
        dilated = dilate_support(support, 1)
        self.assertTrue(np.isnan(new_flow[dilated]).all())
        self.assertEqual(int(new_valid[dilated].sum()), 0)
        # Everything outside the dilated support is untouched.
        self.assertTrue(np.array_equal(new_valid[~dilated], valid[~dilated]))
        self.assertTrue(np.array_equal(new_flow[~dilated], flow[~dilated]))
        self.assertEqual(result["invalidated_pixels"], int((valid & dilated).sum()))
        self.assertEqual(result["dilation_radius"], 1)

    def test_invalidation_survives_dloral_feature_downsample(self):
        """Compose the helper with the real downsample_image_flow consumer.

        A one-pixel edit inside an 8x8 feature cell leaves that cell valid at
        radius 0 (min_valid_fraction=0.5); the caller's conservative radius of
        one full cell width (8 px) must make the edited cell invalid in the
        actual consumed external-flow payload.
        """
        import torch

        from refinement.dloral_flows import (
            VAE_DOWNSAMPLE, downsample_image_flow,
        )

        height, width = 64, 64
        flow = np.full((height, width, 2), -2.5, np.float32)
        flow[..., 0] = 1.5
        valid = np.ones((height, width), bool)
        support = np.zeros((height, width), bool)
        support[3, 3] = True  # one pixel strictly inside the first 8x8 cell

        radius0 = invalidate_target_side_geometry(flow, valid, support,
                                                  dilation_radius=0)
        _, feat_valid0 = downsample_image_flow(
            torch.from_numpy(radius0["pixel_flow"]),
            downsample=VAE_DOWNSAMPLE, min_valid_fraction=0.5,
            max_flow_std=4.0, valid=torch.from_numpy(radius0["valid_mask"]),
        )
        self.assertTrue(bool(feat_valid0[0, 0]),
                        "radius 0 leaves the edited feature cell valid; the "
                        "conservative radius is required")

        conservative = invalidate_target_side_geometry(
            flow, valid, support, dilation_radius=VAE_DOWNSAMPLE)
        _, feat_valid = downsample_image_flow(
            torch.from_numpy(conservative["pixel_flow"]),
            downsample=VAE_DOWNSAMPLE, min_valid_fraction=0.5,
            max_flow_std=4.0, valid=torch.from_numpy(conservative["valid_mask"]),
        )
        self.assertFalse(bool(feat_valid[0, 0]),
                         "the edited feature cell still consumed old geometry")
        self.assertEqual(
            conservative["invalidated_pixels"],
            int((valid & dilate_support(support, VAE_DOWNSAMPLE)).sum()))

    def test_zero_dilation_invalidation_is_exactly_the_edit_support(self):
        flow = np.full((8, 10, 2), 0.0, np.float32)
        valid = np.ones((8, 10), bool)
        support = _rect_support((8, 10), 0, 4, 0, 5)
        result = invalidate_target_side_geometry(flow, valid, support,
                                                 dilation_radius=0)
        self.assertTrue(np.isnan(result["pixel_flow"][support]).all())
        self.assertFalse(result["valid_mask"][support].any())
        self.assertTrue(result["valid_mask"][~support].all())
        self.assertEqual(result["invalidated_pixels"], int(support.sum()))

    def test_support_shape_mismatch_rejected(self):
        with self.assertRaises(LocalFlowEditError):
            invalidate_target_side_geometry(
                np.zeros((8, 10, 2), np.float32), np.ones((8, 10), bool),
                np.ones((5, 5), bool))

    def test_flow_shape_mismatch_rejected(self):
        with self.assertRaises(LocalFlowEditError):
            invalidate_target_side_geometry(
                np.zeros((7, 10, 2), np.float32), np.ones((8, 10), bool),
                np.ones((8, 10), bool))


    def test_preexisting_nan_flow_stays_nan_after_invalidation(self):
        flow = np.full((4, 4, 2), 0.0, np.float32)
        flow[0, 0] = np.nan  # pre-invalid pixel far from the edit
        valid = np.ones((4, 4), bool)
        valid[0, 0] = False
        support = _rect_support((4, 4), 2, 4, 2, 4)
        result = invalidate_target_side_geometry(flow, valid, support, 0)
        self.assertTrue(np.isnan(result["pixel_flow"][0, 0]).all())
        self.assertFalse(result["valid_mask"][0, 0])


if __name__ == "__main__":
    unittest.main()
