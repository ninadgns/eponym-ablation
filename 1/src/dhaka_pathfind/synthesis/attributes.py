"""Vectorized synthetic edge (and optional node) attributes for Dhaka simulation."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from dhaka_pathfind.config import DATA_DIR, EDGES_PARQUET_FILENAME, edges_parquet_path, ensure_data_dir

DEFAULT_SYNTH_SEED = 42


def _extract_edge_frame(graph: nx.MultiDiGraph) -> pd.DataFrame:
    """One pass over edges to build an index-aligned table base."""
    u, v, k, length = [], [], [], []
    for a, b, key, data in graph.edges(keys=True, data=True):
        u.append(a)
        v.append(b)
        k.append(key)
        length.append(float(data.get("length", 0.0) or 0.0))
    return pd.DataFrame({"u": u, "v": v, "key": k, "length_m": length})


def build_synthetic_edge_table(
    graph: nx.MultiDiGraph,
    seed: int = DEFAULT_SYNTH_SEED,
) -> pd.DataFrame:
    """
    Generate synthetic attributes in vectorized form (RNG columns are vectorized).

    Columns are in [0, 1] where noted; ``rickshaw_allowed`` is 0/1.
    """
    base = _extract_edge_frame(graph)
    n = len(base)
    if n == 0:
        return base

    rng = np.random.default_rng(seed)
    # Road-intrinsic
    base["lanes"] = rng.uniform(1.0, 6.0, size=n).astype(np.float64)
    base["surface_quality"] = rng.uniform(0.0, 1.0, size=n)
    base["base_safety"] = rng.uniform(0.0, 1.0, size=n)
    base["accident_risk"] = rng.uniform(0.0, 1.0, size=n)
    base["lighting"] = rng.uniform(0.0, 1.0, size=n)
    base["crime_proxy"] = rng.uniform(0.0, 1.0, size=n)
    base["water_logging"] = rng.uniform(0.0, 1.0, size=n)
    base["incident_rate"] = rng.uniform(0.0, 1.0, size=n)
    base["rickshaw_allowed"] = (rng.uniform(0.0, 1.0, size=n) > 0.12).astype(np.int8)
    # Dynamic priors (used with time/weather in cost model)
    base["traffic_congestion_prior"] = rng.uniform(0.0, 1.0, size=n)

    # Node-level intersection risk via vectorized groupby (two-way merge on u and v)
    u_risk = base.groupby("u")["accident_risk"].mean()
    v_risk = base.groupby("v")["accident_risk"].mean()
    base["u_intersection_risk"] = base["u"].map(u_risk).fillna(0.0).astype(np.float64)
    base["v_intersection_risk"] = base["v"].map(v_risk).fillna(0.0).astype(np.float64)

    return base


def save_edge_table(df: pd.DataFrame, path: Path | None = None) -> None:
    ensure_data_dir()
    p = path or edges_parquet_path()
    df.to_parquet(p, index=False)


def load_edge_table(path: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or edges_parquet_path())


def attach_synthetic_to_graph(
    graph: nx.MultiDiGraph,
    df: pd.DataFrame,
    prefix: str = "synth_",
) -> None:
    """Mutate graph edge data in place with synthetic columns keyed by ``prefix``."""
    idx = df.set_index(["u", "v", "key"])
    synth_cols = [c for c in df.columns if c not in ("u", "v", "key")]
    for u, v, key in graph.edges(keys=True):
        try:
            row = idx.loc[(u, v, key)]
        except KeyError:
            continue
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        for c in synth_cols:
            graph.edges[u, v, key][prefix + c] = float(row[c]) if c != "rickshaw_allowed" else int(row[c])


def ensure_synthetic_edges(
    graph: nx.MultiDiGraph,
    seed: int = DEFAULT_SYNTH_SEED,
    parquet_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load Parquet if present and shape matches; else build, save, attach.
    """
    p = parquet_path or edges_parquet_path()
    n_edges = graph.number_of_edges()
    if p.exists():
        df = load_edge_table(p)
        if len(df) == n_edges:
            attach_synthetic_to_graph(graph, df)
            return df
    df = build_synthetic_edge_table(graph, seed=seed)
    save_edge_table(df, p)
    attach_synthetic_to_graph(graph, df)
    return df
