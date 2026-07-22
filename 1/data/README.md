# Data directory

- `dhaka_graph.graphml.gz` — **the OSM road graph the paper's §4 numbers come from (tracked).**
  55,009 nodes, 137,529 directed edges; Overpass base timestamp `2026-04-21T12:54:14Z`. Loaded
  transparently by `ox.load_graphml`, so nothing has to unpack it.
- `dhaka_graph.graphml` — uncompressed working copy (generated, gitignored). `load_or_download()`
  prefers this if present and falls back to the `.gz`.
- `dhaka_graph.graphml.meta.json` — bbox, network type, node/edge counts, OSM snapshot timestamp
  and the OSMnx version that simplified it (tracked).
- `dhaka_edges.parquet` — vectorized synthetic edge attributes (generated, gitignored). Seeded off
  `DEFAULT_SYNTH_SEED`, so it is a deterministic function of the graph; `ensure_synthetic_edges()`
  rebuilds it.
- `landmarks.yaml` — named places → coordinates (tracked).

## Why the graph is committed

OpenStreetMap is edited continuously. Re-downloading this bbox tomorrow gives a different graph,
and every node count in §4 of the paper — 14,735 UCS expansions, 6,845 for A\*, the 2.15× ratio —
is conditional on one snapshot. Shipping the snapshot is what makes those numbers checkable after
OSM has moved on; a download script would not.

To deliberately fetch a *current* extract instead, call `load_or_download(force_download=True)`.
That overwrites the uncompressed copy and rewrites the metadata, and the paper's node counts will
no longer match.
