"""Georeferencing helpers migrated from the ArtiFixer DSM preparation script."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def ecef_to_latlon(x: float, y: float, z: float) -> tuple[float, float, float]:
    a = 6378137.0
    e = 8.1819190842622e-2
    asq = a**2
    esq = e**2
    b = np.sqrt(asq * (1.0 - esq))
    bsq = b**2
    ep = np.sqrt((asq - bsq) / bsq)
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2((z + (ep**2) * b * (np.sin(th) ** 3)), (p - esq * a * (np.cos(th) ** 3)))
    n = a / np.sqrt(1.0 - esq * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return float(lat * 180.0 / np.pi), float(lon * 180.0 / np.pi), float(alt)

def latlon_to_utm(lat: float, lon: float) -> tuple[float, float, int]:
    zone = int((lon + 180.0) / 6.0) + 1
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    a = 6378137.0
    f = 1.0 / 298.257223563
    k0 = 0.9996
    e2 = f * (2.0 - f)
    ep2 = e2 / (1.0 - e2)
    n = a / np.sqrt(1.0 - e2 * np.sin(lat_rad) ** 2)
    t = np.tan(lat_rad) ** 2
    c = ep2 * np.cos(lat_rad) ** 2
    aa = (lon_rad - lon0) * np.cos(lat_rad)
    m = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat_rad
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * np.sin(2 * lat_rad)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * np.sin(4 * lat_rad)
        - (35 * e2**3 / 3072) * np.sin(6 * lat_rad)
    )
    east = k0 * n * (aa + (1 - t + c) * aa**3 / 6 + (5 - 18 * t + t**2 + 72 * c - 58 * ep2) * aa**5 / 120) + 500000.0
    north = k0 * (
        m
        + n
        * np.tan(lat_rad)
        * (aa**2 / 2 + (5 - t + 9 * c + 4 * c**2) * aa**4 / 24 + (61 - 58 * t + t**2 + 600 * c - 330 * ep2) * aa**6 / 720)
    )
    if lat < 0:
        north += 10_000_000.0
    return float(east), float(north), zone

def load_enu_origin(path: Path) -> dict:
    loc = json.loads(path.read_text())
    lat, lon, alt = ecef_to_latlon(loc["X_offset"], loc["Y_offset"], loc["Z_offset"])
    east, north, zone = latlon_to_utm(lat, lon)
    return {
        "scene_loc": loc,
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "utm_easting": east,
        "utm_northing": north,
        "utm_zone": zone,
    }

def load_dsm(tif_path: Path, txt_path: Path) -> dict:
    west, south, size, gsd = np.loadtxt(txt_path)
    size = int(size)
    dsm = np.array(Image.open(tif_path), dtype=np.float64)
    if dsm.shape != (size, size):
        raise ValueError(f"DSM shape {dsm.shape} != {size}x{size} from {txt_path}")
    return {
        "height": dsm,
        "west": float(west),
        "south": float(south),
        "east": float(west + size * gsd),
        "north": float(south + size * gsd),
        "size": size,
        "gsd": float(gsd),
    }

def colorize(values: np.ndarray, cmap_name: str, vmin: float, vmax: float, nan_rgb=(0.08, 0.08, 0.08)) -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = cmap(norm(np.nan_to_num(values, nan=vmin)))
    rgb = rgba[..., :3].copy()
    rgb[~np.isfinite(values)] = np.asarray(nan_rgb, dtype=np.float64)
    return rgb

def write_xyz_rgb_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    xyz_f32 = xyz.astype("<f4")
    n = xyz_f32.shape[0]
    vertex = np.empty(
        n,
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"] = xyz_f32[:, 0]
    vertex["y"] = xyz_f32[:, 1]
    vertex["z"] = xyz_f32[:, 2]
    vertex["red"] = rgb_u8[:, 0]
    vertex["green"] = rgb_u8[:, 1]
    vertex["blue"] = rgb_u8[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertex.tobytes())
