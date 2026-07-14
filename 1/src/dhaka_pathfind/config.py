"""City-specific configuration. Only bbox and paths are Dhaka-specific."""

from __future__ import annotations

from pathlib import Path

# Project root = parent of src/ when running from repo; fall back to cwd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# (south, west, north, east) — WGS84, Dhaka metropolitan area
DHAKA_BBOX: tuple[float, float, float, float] = (23.65, 90.30, 23.95, 90.52)

# OSMnx / GeoPandas CRS for projected operations when needed
DEFAULT_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:32646"  # UTM zone 46N — reasonable for Dhaka

# Primary routable graph: driving (+service) — vehicle multipliers applied in cost model
DEFAULT_NETWORK_TYPE = "drive_service"

GRAPHML_FILENAME = "dhaka_graph.graphml"
EDGES_PARQUET_FILENAME = "dhaka_edges.parquet"
LANDMARKS_FILENAME = "landmarks.yaml"


def graphml_path() -> Path:
    return DATA_DIR / GRAPHML_FILENAME


def edges_parquet_path() -> Path:
    return DATA_DIR / EDGES_PARQUET_FILENAME


def landmarks_path() -> Path:
    return DATA_DIR / LANDMARKS_FILENAME


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def ensure_outputs_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "results").mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "maps").mkdir(parents=True, exist_ok=True)
