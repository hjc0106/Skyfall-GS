"""Stage2 zoom teachers must actually come from FlowEdit."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from refinement.flowedit_stage2 import build_flowedit_views_manifest
from refinement.types import CameraSnapshot


class FlowEditSupervisionTests(unittest.TestCase):
    def test_dloral_only_output_cannot_be_labeled_as_flowedit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "render_refine").mkdir()
            Image.new("RGB", (16, 16), (64, 96, 128)).save(root / "render_refine/00000.png")
            camera = CameraSnapshot(
                image_name="idu_view", uid=1000, colmap_id=1000,
                image_width=16, image_height=16, fov_x=1.0, fov_y=1.0,
                cx=0.0, cy=0.0,
                R=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                T=(0.0, 0.0, 0.0),
            )
            prepared = {
                "schema_version": 1,
                "kind": "gaussianzoom_stage2_prepared",
                "episode_dir": directory,
                "views": [{"index": 0, "camera": camera.to_dict()}],
            }
            with self.assertRaises(ValueError):
                build_flowedit_views_manifest(
                    prepared, checkpoint=str(root / "checkpoint.pth"),
                    episode_idx=0, samples_per_view=1,
                )


def _write_manifest(root: Path, checkpoint: str, *, overrides=None):
    """Write a minimal valid ``skyfall_flowedit_views`` manifest and return its path."""
    import json

    image = root / "view.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(image)
    camera = CameraSnapshot(
        image_name="idu_view_00000.png", uid=1000, colmap_id=1000,
        image_width=16, image_height=16, fov_x=1.047, fov_y=1.047,
        cx=0.0, cy=0.0,
        R=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        T=(0.0, 0.0, 1.0),
    )
    payload = {
        "schema_version": 1,
        "kind": "skyfall_flowedit_views",
        "checkpoint": checkpoint,
        "episode_idx": 0,
        "views": [{
            "id": "idu_e00_v00000_s00",
            "camera": camera.to_dict(),
            "image_path": str(image),
            "source_stage": "stage2",
            "appearance_uid": 6,
        }],
    }
    if overrides:
        for key, value in overrides.items():
            if key == "_view":
                payload["views"][0].update(value)
            else:
                payload[key] = value
    manifest_path = root / "flowedit_views.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path, payload


class LoadFlowEditViewsManifestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.checkpoint = self.root / "checkpoint.pth"
        self.checkpoint.write_bytes(b"fake")
        self.manifest_path, self.payload = _write_manifest(self.root, str(self.checkpoint))

    def tearDown(self):
        self._tmp.cleanup()

    def _load(self, path=None, **kwargs):
        from refinement.flowedit_stage2 import load_flowedit_views_manifest

        return load_flowedit_views_manifest(path or self.manifest_path, **kwargs)


    def test_wrong_kind_is_rejected(self):
        import json

        self.payload["kind"] = "gaussianzoom_stage2_views"
        self.manifest_path.write_text(json.dumps(self.payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            self._load()

    def test_unsupported_schema_version_is_rejected(self):
        import json

        self.payload["schema_version"] = 2
        self.manifest_path.write_text(json.dumps(self.payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            self._load()

    def test_zoomed_or_unknown_teacher_stage_is_rejected(self):
        import json

        self.payload["views"][0]["source_stage"] = "zoom_l2"
        self.manifest_path.write_text(json.dumps(self.payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            self._load()

    def test_real_photos_cannot_be_mislabeled_as_flowedit_teachers(self):
        import json

        self.payload["views"][0]["source_stage"] = "real_replay"
        self.manifest_path.write_text(json.dumps(self.payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            self._load()

    def test_foreign_appearance_uid_is_rejected(self):
        import json

        self.payload["views"][0]["appearance_uid"] = 3
        self.manifest_path.write_text(json.dumps(self.payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            self._load()

    def test_checkpoint_mismatch_is_rejected(self):
        other = self.root / "other.pth"
        other.write_bytes(b"fake")
        with self.assertRaises(ValueError):
            self._load(expected_checkpoint=str(other))

    def test_missing_checkpoint_file_is_rejected(self):
        import json

        self.payload["checkpoint"] = str(self.root / "gone.pth")
        self.manifest_path.write_text(json.dumps(self.payload), encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            self._load()

    def test_image_dimension_mismatch_is_rejected(self):
        import json

        Image.new("RGB", (32, 16), (0, 0, 0)).save(self.root / "view.png")
        self.manifest_path.write_text(json.dumps(self.payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            self._load()

    def test_non_uniform_raster_across_views_is_rejected(self):
        import json

        image = self.root / "view2.png"
        Image.new("RGB", (32, 32), (0, 0, 0)).save(image)
        second = dict(self.payload["views"][0])
        second["id"] = "idu_e00_v00001_s00"
        second["image_path"] = str(image)
        second["camera"] = dict(second["camera"])
        second["camera"]["image_width"] = 32
        second["camera"]["image_height"] = 32
        self.payload["views"].append(second)
        self.manifest_path.write_text(json.dumps(self.payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            self._load()

    def test_duplicate_view_id_is_rejected(self):
        import json

        self.payload["views"].append(dict(self.payload["views"][0]))
        self.manifest_path.write_text(json.dumps(self.payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            self._load()

    def test_missing_image_file_is_rejected(self):
        import json

        self.payload["views"][0]["image_path"] = str(self.root / "gone.png")
        self.manifest_path.write_text(json.dumps(self.payload), encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            self._load()

    def test_relative_image_path_resolves_against_manifest(self):
        import json

        self.payload["views"][0]["image_path"] = "view.png"
        self.manifest_path.write_text(json.dumps(self.payload), encoding="utf-8")
        loaded = self._load()
        self.assertEqual(loaded["views"][0]["image_path"], str(self.root / "view.png"))


if __name__ == "__main__":
    unittest.main()
