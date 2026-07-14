"""Write example Folium map — used by ``run.sh all``."""

from __future__ import annotations

from dhaka_pathfind.config import OUTPUTS_DIR
from dhaka_pathfind.cost.context import CostPreset, TravellerContext
from dhaka_pathfind.graph.load import load_landmarks, load_or_download, nearest_node
from dhaka_pathfind.search.algorithms import ucs
from dhaka_pathfind.synthesis.attributes import ensure_synthetic_edges
from dhaka_pathfind.viz.folium_routes import save_route_map


def main() -> None:
    g = load_or_download()
    ensure_synthetic_edges(g)
    lm = load_landmarks()
    ctx = TravellerContext()
    s = nearest_node(g, lm["Shahbag"]["lat"], lm["Shahbag"]["lon"])
    t = nearest_node(g, lm["Motijheel"]["lat"], lm["Motijheel"]["lon"])
    r = ucs(g, s, t, ctx, CostPreset.BALANCED)
    out = save_route_map(g, r.path, OUTPUTS_DIR / "maps" / "example_ucs.html", "UCS Shahbag–Motijheel")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
