# Presentation outline

1. **Motivation** — “Shortest” ≠ best in Dhaka: water, safety, vehicle bans, rush hour.
2. **Data** — OSMnx bbox, cached `graphml`; landmarks YAML.
3. **Synthesis** — Vectorized edge features (lanes, crime proxy, lighting, …).
4. **Cost model** — Intrinsic × dynamic × traveller factors; presets (`balanced`, `speed`, `safety`, `comfort`).
5. **Algorithms** — UCS, Dijkstra (pair), bidirectional UCS, A\*, weighted A\*, greedy best-first; all use `edge_cost`.
6. **Heuristics** — `admissible` (conservative \(m^\*\) × great-circle), `realism`, `fast`.
7. **Evaluation** — Batch CSV, nodes expanded, runtime, heuristic gap plots.
8. **Demo** — Folium route + Streamlit compare-all-six.
9. **Limitations** — Synthetic dynamics; admissibility assumes embedding consistency; not real-time traffic.
