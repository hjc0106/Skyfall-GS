"""Only the conditioning prompt is required to contain text."""
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import tempfile
import unittest

from refinement.qwen3_vlm import CACHE_SCHEMA_VERSION, Qwen3PromptProvider, _model_context, _parse_prompt_response
from refinement.types import CameraSnapshot, PromptDescription
from refinement.vlm_prompt import PromptCache, PromptManager, SHARED_PROMPT_CONTEXT_FIELDS


class QwenPromptSchemaTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "shared_region_description": "Buildings, roads and trees.",
            "current_scale_description": "",
            "source_prompt": "",
            "target_prompt": "Restore roof textures while preserving the visible building geometry.",
            "visible_features": ["Flat roofs"],
            "preserve_structure": ["Building silhouettes"],
            "uncertain_information": [],
        }

    def test_valid_conditioning_survives_empty_auxiliary_descriptions(self):
        self.assertEqual(_parse_prompt_response(json.dumps(self.payload)), self.payload)

    def test_empty_conditioning_prompt_is_rejected(self):
        self.payload["target_prompt"] = "  "
        with self.assertRaises(ValueError):
            _parse_prompt_response(json.dumps(self.payload))

    def test_truncated_response_is_rejected(self):
        with self.assertRaises(ValueError):
            _parse_prompt_response(json.dumps(self.payload)[:-1])

    def test_direct_worker_starts_without_site_initialization(self):
        worker = Path(__file__).resolve().parents[1] / "refinement" / "qwen3_vlm.py"
        environment = dict(os.environ)
        # Skip .pth startup imports so they cannot hide types.py shadowing,
        # while keeping the worker package's Pillow dependency available.
        environment["PYTHONPATH"] = os.pathsep.join(site.getsitepackages())
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-S", str(worker), "--serve"],
                input="", text=True, capture_output=True, cwd=directory,
                env=environment, timeout=20,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")


class _RecordingQwenProvider(Qwen3PromptProvider):
    """Real Qwen cache identity, stubbed inference.

    ``cache_schema_version=1`` reproduces a cache written before the shared
    context contract changed.
    """

    def __init__(self, model_path, cache_schema_version=CACHE_SCHEMA_VERSION, **kwargs):
        super().__init__(model_path, **kwargs)
        self._cache_schema_version = cache_schema_version
        self.requests = []

    def cache_config(self):
        return {**dict(super().cache_config()), "schema_version": self._cache_schema_version}

    def describe(self, wide_image, zoom_image, *, zoom_factor, level_index, context):
        self.requests.append(dict(context))
        return PromptDescription(
            shared_region_description="Wide view: brick buildings with flat roofs above a paved street.",
            current_scale_description=f"Zoom level {zoom_factor:g}x.",
            visible_features=("flat roofs", "paved street"),
            preserve_structure=("building silhouettes",),
            uncertain_information=("balcony details",),
            source_prompt="Street scene.",
            target_prompt="Restore roof and facade textures while preserving visible geometry.",
            provider=self.name,
            config={**self.cache_config(), "raw_response": '{"provider": "qwen3_vl", "config": {"model_files": ['},
        )


class QwenSharedPromptBoundaryTests(unittest.TestCase):
    """A cached shared description is an audit record, not model prompt text.

    Regression: the cached shared record (provider/config/model file inventory/
    raw VLM response) was forwarded verbatim into the Qwen prompt, so the model
    echoed the metadata and its JSON answer was truncated at the token budget.
    """

    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        root = Path(self.workspace.name)
        self.model_path = root / "model"
        self.model_path.mkdir()
        (self.model_path / "config.json").write_text("{}", encoding="utf-8")
        self.cache_dir = root / "prompt_cache"
        checkpoint = root / "control.pth"
        checkpoint.write_bytes(b"checkpoint")
        self.request_kwargs = dict(
            checkpoint=str(checkpoint),
            camera=CameraSnapshot(
                image_name="view_000", uid=0, colmap_id=0, image_width=64, image_height=64,
                fov_x=1.0, fov_y=1.0, cx=32.0, cy=32.0,
                R=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), T=(0.0, 0.0, 0.0),
            ),
            roi={"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
            wide_image=None,
            zoom_image=None,
        )

    def _describe(self, provider, **overrides):
        kwargs = {**self.request_kwargs, "zoom_factor": 4.0, "level_index": 0, **overrides}
        return PromptManager(provider, PromptCache(self.cache_dir)).get_prompt(**kwargs)

    def _shared_records(self):
        return sorted(self.cache_dir.glob("shared_*.json"))

    def _seed_legacy_contaminated_cache(self):
        """Write the pre-fix cache: shared record carries provider/config/raw_response."""
        legacy = _RecordingQwenProvider(self.model_path, cache_schema_version=1)
        self._describe(legacy)
        records = self._shared_records()
        self.assertEqual(len(records), 1)
        record = json.loads(records[0].read_text(encoding="utf-8"))
        self.assertEqual(record["provider"], "qwen3_vl")
        self.assertIn("model_files", record["config"])
        self.assertIn("raw_response", record["config"])
        return records[0], record

    def _assert_no_metadata(self, model_text):
        for leaked in ("model_files", "raw_response", "model_path", "instruction_sha256",
                       "shared_cache_key", "level_cache_key", '"provider"', "schema_version"):
            self.assertNotIn(leaked, model_text)

    def test_legacy_contaminated_cache_is_not_reused_and_prompt_stays_semantic(self):
        legacy_path, legacy_record = self._seed_legacy_contaminated_cache()

        provider = _RecordingQwenProvider(self.model_path)
        # Same checkpoint/camera/roi/level as the legacy run: with the old cache
        # identity this would have been a silent cache hit on contaminated data.
        prompt_at_seed_level = self._describe(provider, level_index=0)
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(provider.requests[0]["shared_prompt"], {})
        self.assertEqual(
            prompt_at_seed_level.target_prompt,
            "Restore roof and facade textures while preserving visible geometry.",
        )

        # The fresh shared record now feeds later levels with semantics only.
        prompt = self._describe(provider, level_index=1)
        self.assertEqual(len(provider.requests), 2)
        shared = provider.requests[1]["shared_prompt"]
        self.assertEqual(
            shared["shared_region_description"], legacy_record["shared_region_description"]
        )
        self.assertEqual(shared["visible_features"], legacy_record["visible_features"])
        self.assertEqual(shared["preserve_structure"], legacy_record["preserve_structure"])
        self.assertEqual(
            shared["uncertain_information"], legacy_record["uncertain_information"]
        )
        self.assertTrue(set(shared) <= set(SHARED_PROMPT_CONTEXT_FIELDS), sorted(shared))
        self._assert_no_metadata(json.dumps(shared))

        # Semantic content still reaches the returned description.
        self.assertEqual(prompt.shared_region_description, legacy_record["shared_region_description"])
        self.assertEqual(prompt.visible_features, tuple(legacy_record["visible_features"]))

        # Provenance stays available: the legacy record is untouched and the new
        # shared record keeps provider/config/raw_response audit metadata.
        self.assertEqual(json.loads(legacy_path.read_text(encoding="utf-8")), legacy_record)
        records = self._shared_records()
        self.assertEqual(len(records), 2)
        fresh_paths = [path for path in records if path != legacy_path]
        self.assertEqual(len(fresh_paths), 1)
        fresh = json.loads(fresh_paths[0].read_text(encoding="utf-8"))
        self.assertEqual(fresh["provider"], "qwen3_vl")
        self.assertIn("model_files", fresh["config"])
        self.assertIn("raw_response", fresh["config"])
        self.assertIn("raw_response", prompt.config)
        self.assertIn("model_files", prompt.config)

    def test_model_context_neither_grows_nor_dumps_metadata(self):
        self._seed_legacy_contaminated_cache()
        provider = _RecordingQwenProvider(self.model_path)
        self._describe(provider, level_index=0)  # seeds the new-identity shared record
        self._describe(provider, level_index=1)
        self._describe(provider, level_index=2)

        shared_contexts = [request["shared_prompt"] for request in provider.requests]
        self.assertEqual(len(shared_contexts), 3)
        self.assertEqual(shared_contexts[0], {})  # stale identity: nothing reused
        self.assertEqual(shared_contexts[1], shared_contexts[2])
        self.assertTrue(shared_contexts[1])
        self._assert_no_metadata(json.dumps(shared_contexts[1]))
        largest_stored = max(
            len(path.read_text(encoding="utf-8")) for path in self._shared_records()
        )
        self.assertLess(len(json.dumps(shared_contexts[1])), largest_stored)

    def test_worker_model_context_projects_stale_request_shared_context(self):
        """A request.json written before the fix must still not reach the model."""
        _, contaminated = self._seed_legacy_contaminated_cache()
        context = _model_context({
            "zoom_factor": 4.0, "level_index": 0, "shared": contaminated,
        })
        self.assertEqual(set(context), {"zoom_factor", "level_index", "shared"})
        self.assertTrue(set(context["shared"]) <= set(SHARED_PROMPT_CONTEXT_FIELDS))
        self._assert_no_metadata(json.dumps(context))
        self.assertEqual(context["shared"]["visible_features"], contaminated["visible_features"])
        self.assertEqual(
            context["shared"]["shared_region_description"],
            contaminated["shared_region_description"],
        )

    def test_metadata_echoed_response_is_still_rejected(self):
        """The strict parser keeps failing truncated answers; no repair fallback."""
        echoed = (
            '{"shared_region_description": "Buildings.", "config": {"model_files": [["config.json", 1, 2]], '
            '"raw_response": "..."}, "visible_features": ["Flat roofs"], "preserve_structure": [], '
            '"uncertain_information": [], "source_prompt": "Street.", "target_prompt": "Restore roofs'
        )
        with self.assertRaises(ValueError):
            _parse_prompt_response(echoed)


if __name__ == "__main__":
    unittest.main()
