"""Real-coded Genetic Algorithm — from scratch (PROBLEM_STATEMENT.md §A.8).

Holland (1975); BLX-alpha is Eshelman & Schaffer (1993). Runs at the IDENTICAL evaluation budget
as PSO (3030), so the cross-family comparison is honest.

  * Selection:  tournament, size 3 (on the objective — lower is better)
  * Crossover:  BLX-alpha, alpha = 0.5
  * Mutation:   Gaussian, sigma = 0.05*T, rate 1/K per gene
  * Elitism:    carry the top 2 unchanged

**How the budget lands on exactly 3030.** Each generation draws a full replacement pool of
`pop_size` offspring and evaluates all of them, so a generation costs exactly `pop_size`
evaluations and the run costs pop_size * (1 + n_generations) = 3030 — the same as PSO's
n_particles * (1 + n_iters). The next population is then the 2 elites (already evaluated, so they
cost nothing to carry) plus the best pop_size - 2 offspring. The alternative — breeding only
pop_size - elitism offspring — would spend 2830 evaluations and quietly hand the GA a different
budget from PSO, which is the one thing §A.8 says must not happen.

**The encoding's weakness, and it is worth saying out loud in the report.** The chromosome is the
departure-time vector itself, and the objective is invariant under permutation of its genes (§A.4).
So gene position carries no meaning: crossing gene 3 of parent 1 with gene 3 of parent 2 is mixing
two coordinates that are only "the same coordinate" by an accident of labelling. BLX-alpha is a
positional operator applied to a non-positional encoding — it can take two good schedules and blend
them into one that is good nowhere, because the parents may be sorted differently. PSO's velocity
update has the identical exposure. Sorting the vector before evaluation (the simulator does this
internally) does not remove the problem; it only hides it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shuttle_timetable.instance import ShuttleConfig
from shuttle_timetable.simulator import objective


@dataclass(frozen=True)
class GAResult:
    best_x: np.ndarray
    best_j: float
    curve: np.ndarray  # best-so-far objective, one entry per evaluation
    n_evals: int


def genetic_algorithm(
    arrival_sets: list[np.ndarray],
    cfg: ShuttleConfig,
    rng: np.random.Generator,
    pop_size: int = 30,
    n_generations: int = 100,
    tournament_size: int = 3,
    blx_alpha: float = 0.5,
    mutation_sigma_frac: float = 0.05,
    elitism: int = 2,
) -> GAResult:
    k = cfg.K
    lo, hi = 0.0, float(cfg.T)
    sigma = mutation_sigma_frac * cfg.T
    mut_rate = 1.0 / k
    budget = pop_size * (1 + n_generations)

    curve = np.empty(budget, dtype=float)
    n_evals = 0
    best_so_far = np.inf

    def _evaluate(x: np.ndarray) -> float:
        """The single point where the budget is spent. Nothing else may call `objective`."""
        nonlocal n_evals, best_so_far
        j = objective(x, arrival_sets, cfg)
        best_so_far = min(best_so_far, j)
        curve[n_evals] = best_so_far
        n_evals += 1
        return j

    pop = rng.uniform(lo, hi, size=(pop_size, k))
    fit = np.array([_evaluate(ind) for ind in pop])  # spends pop_size evaluations

    for _ in range(n_generations):
        children = np.empty_like(pop)
        for c in range(0, pop_size, 2):
            p1 = pop[_tournament(fit, tournament_size, rng)]
            p2 = pop[_tournament(fit, tournament_size, rng)]
            ch1, ch2 = _blx_alpha(p1, p2, blx_alpha, rng)
            children[c] = ch1
            if c + 1 < pop_size:
                children[c + 1] = ch2

        mask = rng.random(children.shape) < mut_rate
        children += mask * rng.normal(0.0, sigma, children.shape)
        np.clip(children, lo, hi, out=children)  # box repair, same rule as PSO

        child_fit = np.array([_evaluate(ch) for ch in children])  # pop_size evaluations

        elite_idx = np.argsort(fit, kind="stable")[:elitism]
        keep = np.argsort(child_fit, kind="stable")[: pop_size - elitism]
        pop = np.vstack([pop[elite_idx], children[keep]])
        fit = np.concatenate([fit[elite_idx], child_fit[keep]])

    best = int(np.argmin(fit))
    return GAResult(
        best_x=pop[best].copy(),
        best_j=float(fit[best]),
        curve=curve,
        n_evals=n_evals,
    )


def _tournament(fit: np.ndarray, size: int, rng: np.random.Generator) -> int:
    """Pick `size` individuals at random (with replacement); the fittest — lowest J — wins."""
    contenders = rng.integers(0, fit.size, size)
    return int(contenders[np.argmin(fit[contenders])])


def _blx_alpha(
    p1: np.ndarray, p2: np.ndarray, alpha: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """BLX-alpha: sample each gene uniformly from the parents' interval, widened by alpha*d.

    The widening is what lets the GA explore outside the parents' bounding box; without it the
    population contracts monotonically and the search collapses.
    """
    c_min = np.minimum(p1, p2)
    c_max = np.maximum(p1, p2)
    d = c_max - c_min
    low = c_min - alpha * d
    high = c_max + alpha * d
    return rng.uniform(low, high), rng.uniform(low, high)
