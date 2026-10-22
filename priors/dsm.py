"""Read unreferenced DSM GeoTIFFs using explicit, provenance-tagged sidecars."""
from pathlib import Path
import numpy as np
import tifffile


def read_dsm(path, sidecar, epsg=32617, nodata=None):
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        heights = page.asarray().astype(np.float64)
        tags = {tag.name: tag.value for tag in page.tags.values()
                if tag.name in ('ModelPixelScaleTag', 'ModelTiepointTag', 'GeoKeyDirectoryTag', 'GDAL_NODATA')}
    meta = np.loadtxt(sidecar).ravel()
    if len(meta) != 4:
        raise ValueError('DSM sidecar must contain xoff, yoff, size, resolution')
    xoff, yoff, size, resolution = map(float, meta)
    if resolution <= 0 or size != int(size) or heights.shape != (int(size), int(size)):
        raise ValueError('DSM dimensions/resolution do not match sidecar')
    valid = np.isfinite(heights)
    if nodata is not None:
        valid &= heights != nodata
    info = {
        'path': str(Path(path).resolve()), 'sidecar': str(Path(sidecar).resolve()),
        'epsg': epsg, 'crs_source': 'explicit_user_UTM_17N_interpretation_of_sidecar_values',
        'vertical_datum': 'unknown', 'vertical_shift': None,
        'xoff': xoff, 'yoff': yoff, 'size': int(size), 'resolution_m': resolution,
        'geotiff_tags': tags, 'nodata_override': nodata,
        'valid_cells': int(valid.sum()), 'finite_height_range_m': [float(heights[valid].min()), float(heights[valid].max())],
    }
    return heights, valid, info


def sample_dsm(heights, valid, info, spacing_m=2.0, max_points=20000):
    if spacing_m <= 0 or max_points <= 0:
        raise ValueError('Sampling spacing and point limit must be positive')
    step = max(1, int(np.ceil(spacing_m / info['resolution_m'])))
    rows, cols = np.meshgrid(np.arange(0, heights.shape[0], step), np.arange(0, heights.shape[1], step), indexing='ij')
    keep = valid[rows, cols]
    rows, cols = rows[keep], cols[keep]
    if len(rows) > max_points:
        indices = np.linspace(0, len(rows) - 1, max_points, dtype=int)
        rows, cols = rows[indices], cols[indices]
    r = info['resolution_m']
    return np.c_[info['xoff'] + (cols + .5) * r,
                 info['yoff'] + (heights.shape[0] - rows - .5) * r,
                 heights[rows, cols]]
