"""Completed SR targets survive a same-run restart without accepting changed inputs.

Geometry depth contract: every sample carries a versioned
``geometry.geometry_identity``; legacy or mismatched targets are refused before
reuse, and Stage 2 ``--real_supervision`` inheritance may not bypass that.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
import zlib

from PIL import Image

from refinement.scene_zoom import (
    SceneZoomConfig, ViewTask, _ManifestState, _dloral_model_config,
    _view_entry, _zoom_snapshot, build_view_tiles, geometry_identity,
    ingest_real_supervision, restore_completed_supervision,
)
from refinement.types import CameraSnapshot


class SceneZoomResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        checkpoint = self.root / 'chkpnt80000.pth'
        checkpoint.write_bytes(b'checkpoint identity')
        # The geometry renders of this run dispatch to the backend recorded
        # in the Stage 1 cfg_args next to the checkpoint.
        (self.root / 'cfg_args').write_text(
            str(argparse.Namespace(rasterizer_backend='diff_gauss'))
        )
        self.cfg = SceneZoomConfig(
            start_checkpoint=str(checkpoint), output_dir=str(self.root), resume=True,
            vlm_model_path=str(self.root / 'vlm'), vlm_python=str(self.root / 'vlm-python'),
        )
        camera = CameraSnapshot(
            image_name='view', uid=0, colmap_id=0, image_width=512, image_height=512,
            fov_x=0.6, fov_y=0.6, cx=0.0, cy=0.0,
            R=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), T=(0., 0., 0.),
        )
        self.view = ViewTask('view', camera, str(self.root / 'base.png'), 'stage2', None)
        self.plans = [(self.view, build_view_tiles(self.view, (2., 4.)))]
        self.tile = self.plans[0][1][2.][0]
        self.image = self.root / 'cached.png'
        Image.new('RGB', (512, 512)).save(self.image)
        stat = checkpoint.stat()
        self.identity = geometry_identity('diff_gauss')
        request = {
            'checkpoint': str(checkpoint),
            'checkpoint_stat': {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns},
            'camera': _zoom_snapshot(camera, self.tile.roi, 2., *self.tile.crop_size,
                                     name=self.tile.alias, uid=self.tile.uid).to_dict(),
            'model_config': _dloral_model_config(self.cfg, 2., 'geometry'),
            'prompt_config': {
                'vlm_model_path': self.cfg.vlm_model_path, 'vlm_python': self.cfg.vlm_python,
                'vlm_device': self.cfg.vlm_device, 'vlm_max_new_tokens': self.cfg.vlm_max_new_tokens,
                'vlm_max_image_size': self.cfg.vlm_max_image_size,
            },
            'cache_context': {'geometry_identity': dict(self.identity)},
        }
        self.image.with_suffix('.json').write_text(json.dumps({'request': request}))
        sample = {
            'view_id': 'view', 'sample_id': self.tile.sample_id, 'roi': self.tile.roi_dict,
            'crop_box': list(self.tile.crop_box), 'image_path': str(self.image),
            'geometry': {'mode': 'geometry', 'geometry_identity': dict(self.identity)},
            'backend': {'seed': zlib.crc32(self.tile.sample_id.encode()) % (2**31 - 1)},
        }
        self.previous = {
            'kind': 'skyfall_scene_zoom', 'schema_version': 1,
            'base_checkpoint': str(checkpoint), 'views': [_view_entry(self.view)],
            'levels': [{'zoom_factor': 2., 'samples': [sample]}],
        }

    def test_completed_target_is_reused_but_missing_output_is_pending(self):
        state = _ManifestState(self.cfg)
        self.assertEqual(restore_completed_supervision(state, self.previous, self.cfg, self.plans), 1)
        self.assertTrue(state.sample_exists(2., self.tile.sample_id))
        self.image.unlink()
        fresh = _ManifestState(self.cfg)
        self.assertEqual(restore_completed_supervision(fresh, self.previous, self.cfg, self.plans), 0)
        self.assertFalse(fresh.sample_exists(2., self.tile.sample_id))

    def test_changed_checkpoint_cannot_reuse_old_supervision(self):
        path = Path(self.cfg.start_checkpoint)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        with self.assertRaises(ValueError):
            restore_completed_supervision(_ManifestState(self.cfg), self.previous, self.cfg, self.plans)

    def test_changed_generation_seed_cannot_reuse_old_supervision(self):
        self.cfg.seed = 1
        with self.assertRaises(ValueError):
            restore_completed_supervision(_ManifestState(self.cfg), self.previous, self.cfg, self.plans)

    def test_legacy_sample_without_geometry_identity_is_refused(self):
        del self.previous['levels'][0]['samples'][0]['geometry']['geometry_identity']
        with self.assertRaises(ValueError):
            restore_completed_supervision(_ManifestState(self.cfg), self.previous, self.cfg, self.plans)

    def test_mismatched_renderer_backend_is_refused(self):
        self.previous['levels'][0]['samples'][0]['geometry']['geometry_identity'] = (
            geometry_identity('rade')
        )
        with self.assertRaises(ValueError):
            restore_completed_supervision(_ManifestState(self.cfg), self.previous, self.cfg, self.plans)

    def test_cached_request_without_geometry_identity_is_refused(self):
        request = json.loads(self.image.with_suffix('.json').read_text())
        del request['request']['cache_context']['geometry_identity']
        self.image.with_suffix('.json').write_text(json.dumps(request))
        with self.assertRaises(ValueError):
            restore_completed_supervision(_ManifestState(self.cfg), self.previous, self.cfg, self.plans)

    def test_carried_sample_with_valid_foreign_identity_is_not_rerendered(self):
        carried = dict(self.previous['levels'][0]['samples'][0])
        carried['sample_id'] = 'carried__z2__r0c0'
        carried['view_id'] = 'carried_view'  # not planned by this run
        carried['geometry'] = {
            'mode': 'geometry', 'geometry_identity': geometry_identity('rade'),
        }
        self.previous['levels'][0]['samples'].append(carried)
        state = _ManifestState(self.cfg)
        self.assertEqual(restore_completed_supervision(state, self.previous, self.cfg, self.plans), 1)
        self.assertFalse(state.sample_exists(2., 'carried__z2__r0c0'))

    def test_carried_legacy_sample_is_refused_even_off_plan(self):
        carried = dict(self.previous['levels'][0]['samples'][0])
        carried['sample_id'] = 'carried__z2__r0c0'
        carried['view_id'] = 'carried_view'  # not planned by this run
        carried['geometry'] = {'mode': 'geometry'}  # legacy: no contract
        self.previous['levels'][0]['samples'].append(carried)
        with self.assertRaises(ValueError):
            restore_completed_supervision(_ManifestState(self.cfg), self.previous, self.cfg, self.plans)


class Stage2RealSupervisionIngestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        checkpoint = self.root / 'chkpnt80000.pth'
        checkpoint.write_bytes(b'checkpoint identity')
        (self.root / 'cfg_args').write_text(
            str(argparse.Namespace(rasterizer_backend='diff_gauss'))
        )
        self.cfg = SceneZoomConfig(
            start_checkpoint=str(checkpoint), output_dir=str(self.root),
            vlm_model_path=str(self.root / 'vlm'), vlm_python=str(self.root / 'vlm-python'),
        )
        self.identity = geometry_identity('diff_gauss')

    def _payload(self, identity=None, omit_identity=False, base_checkpoint=True):
        sample = {
            'view_id': 'real_view', 'sample_id': 'real_view__z2__r0c0',
            'geometry': {
                'mode': 'target_only',
                **({} if omit_identity else {'geometry_identity': dict(identity or self.identity)}),
            },
        }
        return {
            'kind': 'skyfall_scene_zoom', 'schema_version': 1,
            'base_checkpoint': self.cfg.start_checkpoint if base_checkpoint else None,
            'views': [], 'levels': [{'zoom_factor': 2., 'samples': [sample]}],
        }

    def test_matching_geometry_contract_is_carried(self):
        state = _ManifestState(self.cfg)
        ingest_real_supervision(state, self._payload())
        self.assertTrue(state.sample_exists(2., 'real_view__z2__r0c0'))

    def test_legacy_sample_without_identity_is_refused(self):
        with self.assertRaises(ValueError):
            ingest_real_supervision(_ManifestState(self.cfg), self._payload(omit_identity=True))

    def test_sample_mismatching_its_own_checkpoint_backend_is_refused(self):
        with self.assertRaises(ValueError):
            ingest_real_supervision(
                _ManifestState(self.cfg), self._payload(identity=geometry_identity('rade'))
            )

    def test_payload_without_base_checkpoint_still_requires_valid_identity(self):
        state = _ManifestState(self.cfg)
        payload = self._payload(base_checkpoint=False)
        ingest_real_supervision(state, payload)
        self.assertTrue(state.sample_exists(2., 'real_view__z2__r0c0'))
        legacy = self._payload(base_checkpoint=False, omit_identity=True)
        with self.assertRaises(ValueError):
            ingest_real_supervision(_ManifestState(self.cfg), legacy)


if __name__ == '__main__':
    unittest.main()
