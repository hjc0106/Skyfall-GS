"""Recursive teacher reuse must retain real anchors and the exact parent identity."""
import json
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from refinement.scene_zoom import (
    PROGRESSIVE_RGB_CONTRACT,
    SceneZoomConfig, ViewTask, _ManifestState, _dloral_model_config,
    _file_identity, _parent_bundle_identity, _tensor_to_pil, _view_entry, _zoom_snapshot,
    build_view_tiles, geometry_identity, ingest_previous_supervision,
    restore_completed_supervision,
)
from refinement.types import CameraSnapshot


class ProgressiveIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / 'base.pth'
        self.parent = self.root / 'l1.lod.pt'
        self.sidecar = Path(str(self.parent) + '.appearance.pt')
        for path in (self.base, self.parent, self.sidecar):
            path.write_bytes(path.name.encode())
        self.photo = self.root / 'real.png'
        Image.new('RGB', (512, 512), (35, 67, 91)).save(self.photo)
        camera = CameraSnapshot('real', 0, 0, 512, 512, .6, .6, 0., 0.,
                                ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), (0., 0., 0.))
        self.view = ViewTask('real_view', camera, str(self.photo), 'stage1', 0)
        self.cfg = SceneZoomConfig(str(self.base), str(self.root), progressive=True,
                                   parent_lod_checkpoint=str(self.parent), zoom_factors=(4.,))

    def test_parent_appearance_change_invalidates_completed_teacher(self):
        tiles = build_view_tiles(self.view, (4.,), level_offset=1)
        tile = tiles[4.][0]
        lr = self.root / 'lr.png'
        lr_mask = self.root / 'lr_mask.png'
        target = self.root / 'target.png'
        Image.new('RGB', (128, 128)).save(lr)
        Image.new('L', (128, 128), 255).save(lr_mask)
        Image.new('RGB', (512, 512)).save(target)
        parent = _parent_bundle_identity(self.cfg)
        identity = geometry_identity('rade', parent)
        request = {
            'checkpoint': str(self.base),
            'checkpoint_stat': {'size': self.base.stat().st_size, 'mtime_ns': self.base.stat().st_mtime_ns},
            'camera': _zoom_snapshot(self.view.snapshot, tile.roi, 4., 256, 256,
                                     name=tile.alias, uid=tile.uid).to_dict(),
            'model_config': _dloral_model_config(self.cfg, 4., 'geometry'),
            'prompt_config': {'vlm_model_path': None, 'vlm_python': None,
                              'vlm_device': 'cuda:0', 'vlm_max_new_tokens': 768,
                              'vlm_max_image_size': 1024},
            'cache_context': {'geometry_identity': identity, 'render_rgb_contract': PROGRESSIVE_RGB_CONTRACT},
        }
        target.with_suffix('.json').write_text(json.dumps({'request': request}))
        sample = {
            'sample_id': tile.sample_id, 'view_id': self.view.view_id,
            'image_path': str(target), 'lr_image_path': str(lr), 'lr_mask_path': str(lr_mask),
            'roi': tile.roi_dict, 'crop_box': list(tile.crop_box),
            'geometry': {'mode': 'geometry', 'geometry_identity': identity},
            'backend': {'seed': zlib.crc32(tile.sample_id.encode()) % (2**31 - 1)},
            'progressive': {'parent_identity': parent, 'render_rgb_contract': PROGRESSIVE_RGB_CONTRACT,
                            'real_base_identity': _file_identity(self.photo),
                            'lr_image_identity': _file_identity(lr), 'lr_mask_identity': _file_identity(lr_mask)},
        }
        manifest = {'kind': 'skyfall_scene_zoom', 'schema_version': 1, 'base_checkpoint': str(self.base),
                    'views': [_view_entry(self.view)], 'levels': [{'zoom_factor': 4., 'samples': [sample]}]}
        plan = [(self.view, tiles)]
        state = _ManifestState(self.cfg)
        self.assertEqual(restore_completed_supervision(state, manifest, self.cfg, plan), 1)
        self.sidecar.write_bytes(b'different learned appearance')
        with self.assertRaises(ValueError):
            restore_completed_supervision(_ManifestState(self.cfg), manifest, self.cfg, plan)

    def test_previous_level_accepts_relocated_identical_real_photo_not_changed_photo(self):
        old_photo = self.root / 'old_real.png'
        old_photo.write_bytes(self.photo.read_bytes())
        old_view = dict(_view_entry(self.view), image_path=str(old_photo))
        sample = {'sample_id': 'old', 'view_id': self.view.view_id, 'image_path': str(old_photo),
                  'geometry': {'geometry_identity': geometry_identity('rade')},
                  'tile_row': 0, 'tile_col': 0}
        manifest = {'kind': 'skyfall_scene_zoom', 'schema_version': 1, 'base_checkpoint': str(self.base),
                    'views': [old_view], 'levels': [{'zoom_factor': 2., 'samples': [sample]}]}
        state = _ManifestState(self.cfg)
        ingest_previous_supervision(state, manifest, manifest_path=str(self.root / 'old.json'),
                                    collected_entries={self.view.view_id: _view_entry(self.view)})
        self.assertEqual(state.samples[2.]['old']['image_path'], str(old_photo))
        self.assertEqual(state.views[self.view.view_id]['image_path'], str(self.photo))
        Image.new('RGB', (512, 512), (220, 0, 0)).save(old_photo)
        with self.assertRaises(ValueError):
            ingest_previous_supervision(_ManifestState(self.cfg), manifest,
                                        manifest_path=str(self.root / 'old.json'),
                                        collected_entries={self.view.view_id: _view_entry(self.view)})


class RenderPixelUnitsTests(unittest.TestCase):
    def test_float_hdr_highlight_does_not_rescale_the_whole_frame(self):
        rendered = torch.full((3, 4, 4), 0.5)
        rendered[:, 0, 0] = 1.05
        image = np.array(_tensor_to_pil(rendered))
        self.assertTrue((image[1, 1] == 128).all())
        self.assertTrue((image[0, 0] == 255).all())

    def test_dark_uint8_pixels_keep_byte_units(self):
        image = np.array(_tensor_to_pil(torch.ones((3, 4, 4), dtype=torch.uint8)))
        self.assertTrue((image == 1).all())


if __name__ == '__main__':
    unittest.main()
