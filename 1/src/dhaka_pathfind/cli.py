"""Typer CLI for ad-hoc queries and batch."""

from __future__ import annotations

from pathlib import Path

import typer

from dhaka_pathfind.analysis.batch import run_batch
from dhaka_pathfind.analysis.plots import generate_all, latest_csv
from dhaka_pathfind.cost.context import (
    AgeGroup,
    CostPreset,
    Gender,
    Social,
    TimeOfDay,
    TravellerContext,
    Vehicle,
    Weather,
)
from dhaka_pathfind.graph.load import load_or_download, resolve_named_or_coords
from dhaka_pathfind.search.algorithms import (
    astar,
    bidirectional_ucs,
    dijkstra,
    greedy_best_first,
    ucs,
    weighted_astar,
)
from dhaka_pathfind.synthesis.attributes import ensure_synthetic_edges
from dhaka_pathfind.viz.folium_routes import save_route_map

app = typer.Typer(no_args_is_help=True)


def _dispatch(algo: str, graph, s, t, ctx, preset, heur: str):
    a = algo.lower().replace("-", "_")
    if a == "ucs":
        return ucs(graph, s, t, ctx, preset)
    if a == "dijkstra":
        return dijkstra(graph, s, t, ctx, preset)
    if a == "bidirectional_ucs":
        return bidirectional_ucs(graph, s, t, ctx, preset)
    if a == "astar":
        return astar(graph, s, t, ctx, preset, heuristic_name=heur)
    if a == "weighted_astar":
        return weighted_astar(graph, s, t, ctx, preset, heuristic_name=heur)
    if a == "greedy_best_first" or a == "greedy":
        return greedy_best_first(graph, s, t, ctx, preset, heuristic_name=heur)
    raise typer.BadParameter(f"unknown algorithm {algo}")


@app.command()
def query(
    from_name: str = typer.Option(..., "--from", help="Landmark name (see data/landmarks.yaml)"),
    to_name: str = typer.Option(..., "--to", help="Landmark name"),
    algorithm: str = typer.Option("ucs", "--algorithm", "-a"),
    preset: CostPreset = typer.Option(CostPreset.BALANCED, "--preset", "-p"),
    heuristic: str = typer.Option("admissible", "--heuristic", "-H"),
    gender: Gender = Gender.MALE,
    social: Social = Social.ALONE,
    age: AgeGroup = AgeGroup.ADULT,
    vehicle: Vehicle = Vehicle.CAR,
    time_of_day: TimeOfDay = TimeOfDay.MIDDAY,
    weather: Weather = Weather.CLEAR,
    map_out: Path | None = typer.Option(None, "--map", help="Write Folium HTML path"),
):
    """Run one search between named landmarks."""
    graph = load_or_download()
    ensure_synthetic_edges(graph)
    ctx = TravellerContext(
        gender=gender,
        social=social,
        age=age,
        vehicle=vehicle,
        time_of_day=time_of_day,
        weather=weather,
    )
    s = resolve_named_or_coords(graph, from_name, None)
    t = resolve_named_or_coords(graph, to_name, None)
    res = _dispatch(algorithm, graph, s, t, ctx, preset, heuristic)
    typer.echo(f"algorithm={algorithm} cost={res.path_cost:.4f} expanded={res.metrics.nodes_expanded}")
    if map_out and res.path:
        save_route_map(graph, res.path, map_out)


@app.command("batch")
def batch_cmd(
    n: int = typer.Option(10, "--n", "-n", help="landmark pairs"),
    seed: int = 42,
    out: Path | None = None,
):
    run_batch(pairs_count=n, seed=seed, out_csv=out)


@app.command()
def report(
    csv_path: Path | None = typer.Option(None, "--csv"),
    latest: bool = typer.Option(True, "--latest/--no-latest"),
):
    p = latest_csv() if latest and csv_path is None else csv_path
    if p is None:
        raise typer.BadParameter("provide --csv or use --latest")
    generate_all(p)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
