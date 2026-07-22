"""Part B model: state space, transition kernel P, expected reward R, and a simulator.

PROBLEM_STATEMENT.md §B.2–B.5. Two things here are easy to get wrong and are load-bearing:

1. **The truncated Poisson is RENORMALISED, not clipped.** Clipping dumps the tail mass onto the
   endpoint and silently biases the arrival model.

2. **r is NOT a function of (s, a, s').** Realised spillback depends on realised arrivals, and
   once a queue truncates at Q_max the arrivals are unrecoverable from s'. This is the standard
   disturbance form of an MDP. Value Iteration consumes the *expected* reward R(s,a) tabulated
   here; Q-learning MUST consume the *realised* r from `step()`. Feeding Q-learning E[spill]
   would leak the arrival distribution into a supposedly model-free agent and destroy the entire
   comparison Part B exists to make.

`lam_scale` builds a *believed* model with deliberately wrong arrival rates — that is the whole
mechanism of experiment B.10.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse

HOLD, SWITCH = 0, 1
ACTIONS = (HOLD, SWITCH)

PEAK, OFF, NIGHT = 0, 1, 2
N_REGIMES = 3


@dataclass(frozen=True)
class SignalConfig:
    dt: float = 10.0  # tick length, seconds
    Q_max: int = 12  # queue truncation per approach, vehicles
    mu: int = 5  # saturation discharge per tick (~1800 veh/h)
    e_min: int = 2  # minimum green, ticks (20 s) — pedestrian safety
    E_max: int = 6  # maximum green, ticks (60 s) — anti-starvation
    A_max: int = 8  # arrival truncation per approach per tick

    lam_A: tuple[float, ...] = (2.0, 1.0, 0.3)  # main road, by regime
    lam_B: tuple[float, ...] = (1.0, 0.6, 0.15)  # side road, by regime

    c_switch: float = 3.0  # changeover cost (lost time), queue-vehicle units
    c_spill: float = 8.0  # cost per vehicle of spillback beyond Q_max
    gamma: float = 0.99  # discount; horizon 100 ticks ~ 17 min

    eta: float = 1.0 / 360.0  # regime-change prob per tick (mean dwell 1 h)


DEFAULT = SignalConfig()


def _truncated_poisson(lam: float, a_max: int) -> np.ndarray:
    """Poisson(lam) truncated to {0..a_max} and RENORMALISED. Sums to exactly 1."""
    k = np.arange(a_max + 1)
    pmf = np.array([math.exp(-lam) * lam**ki / math.factorial(ki) for ki in k])
    return pmf / pmf.sum()


class SignalMDP:
    """Finite discounted MDP for two-phase signal control.

    State s = (q_A, q_B, phi, e, p), flattened row-major.
    Action a in {HOLD, SWITCH}, availability depends on the phase timer e.
    """

    def __init__(self, cfg: SignalConfig = DEFAULT, lam_scale: float = 1.0):
        self.cfg = cfg
        self.lam_scale = lam_scale
        self.dims = (cfg.Q_max + 1, cfg.Q_max + 1, 2, cfg.E_max + 1, N_REGIMES)
        self.nS = int(np.prod(self.dims))
        self.nA = 2

        self._pmf_A = [
            _truncated_poisson(lam_scale * lam, cfg.A_max) for lam in cfg.lam_A
        ]
        self._pmf_B = [
            _truncated_poisson(lam_scale * lam, cfg.A_max) for lam in cfg.lam_B
        ]

        self._build()

    # ---- indexing -------------------------------------------------------------

    def index(self, qa, qb, phi, e, p) -> int:
        return int(np.ravel_multi_index((qa, qb, phi, e, p), self.dims))

    def unravel(self, s: int) -> tuple[int, int, int, int, int]:
        return tuple(int(v) for v in np.unravel_index(s, self.dims))  # type: ignore[return-value]

    def legal_actions(self, e: int) -> tuple[int, ...]:
        """Minimum green forbids switching; maximum green forces it.

        These are HARD ENGINEERING CONSTRAINTS and they DO change V* — they remove actions that
        would be better in the unconstrained problem but are illegal in the real one. Do not
        claim masking 'cannot change the optimum'; that argument is about no-op duplicates, and
        that is not what is happening here.
        """
        if e < self.cfg.e_min:
            return (HOLD,)
        if e >= self.cfg.E_max:
            return (SWITCH,)
        return (HOLD, SWITCH)

    # ---- dynamics -------------------------------------------------------------

    def _next_queue_dist(self, base: int, pmf: np.ndarray) -> tuple[np.ndarray, float]:
        """Distribution over next queue and E[spill], given post-discharge base level."""
        q_max = self.cfg.Q_max
        dist = np.zeros(q_max + 1, dtype=float)
        e_spill = 0.0
        for k, pk in enumerate(pmf):
            v = base + k
            dist[min(v, q_max)] += pk
            e_spill += pk * max(0, v - q_max)
        return dist, e_spill

    def _regime_next(self, p: int) -> list[tuple[int, float]]:
        """Cyclic peak -> off -> night -> peak, mean dwell 1/eta ticks."""
        eta = self.cfg.eta
        return [(p, 1.0 - eta), ((p + 1) % N_REGIMES, eta)]

    def _apply_action(self, qa, qb, phi, e, a):
        """Returns (d_A, d_B, phi', e', is_switch). SWITCH burns a clearance tick: no discharge."""
        cfg = self.cfg
        if a == SWITCH:
            return 0, 0, 1 - phi, 0, True
        q_green = qa if phi == 0 else qb
        d = min(q_green, cfg.mu)
        d_a = d if phi == 0 else 0
        d_b = d if phi == 1 else 0
        return d_a, d_b, phi, min(e + 1, cfg.E_max), False

    def _build(self) -> None:
        cfg = self.cfg
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []

        self.R = np.zeros((self.nS, self.nA), dtype=float)
        self.legal = np.zeros((self.nS, self.nA), dtype=bool)

        for s in range(self.nS):
            qa, qb, phi, e, p = self.unravel(s)
            for a in self.legal_actions(e):
                self.legal[s, a] = True
                d_a, d_b, phi_n, e_n, is_switch = self._apply_action(qa, qb, phi, e, a)

                dist_a, spill_a = self._next_queue_dist(qa - d_a, self._pmf_A[p])
                dist_b, spill_b = self._next_queue_dist(qb - d_b, self._pmf_B[p])

                # Delay is charged on the queue ENTERING the tick (Little's law).
                self.R[s, a] = -(
                    (qa + qb)
                    + cfg.c_switch * is_switch
                    + cfg.c_spill * (spill_a + spill_b)
                )

                outer = np.outer(dist_a, dist_b)
                nz_a, nz_b = np.nonzero(outer)
                probs = outer[nz_a, nz_b]

                for p_n, p_prob in self._regime_next(p):
                    s_next = np.ravel_multi_index(
                        (
                            nz_a,
                            nz_b,
                            np.full(nz_a.size, phi_n),
                            np.full(nz_a.size, e_n),
                            np.full(nz_a.size, p_n),
                        ),
                        self.dims,
                    )
                    rows.append(np.full(s_next.size, s * self.nA + a))
                    cols.append(s_next)
                    vals.append(probs * p_prob)

        self.P = sparse.csr_matrix(
            (
                np.concatenate(vals),
                (np.concatenate(rows), np.concatenate(cols)),
            ),
            shape=(self.nS * self.nA, self.nS),
        )

    # ---- simulator (Q-learning's only view of the world) ----------------------

    def step(self, s: int, a: int, rng: np.random.Generator) -> tuple[int, float]:
        """One tick. Returns (s', realised r). The realised spill is in r — that is the point."""
        cfg = self.cfg
        qa, qb, phi, e, p = self.unravel(s)
        if not self.legal[s, a]:
            raise ValueError(f"illegal action {a} in state {self.unravel(s)}")

        d_a, d_b, phi_n, e_n, is_switch = self._apply_action(qa, qb, phi, e, a)

        arr_a = int(rng.choice(cfg.A_max + 1, p=self._pmf_A[p]))
        arr_b = int(rng.choice(cfg.A_max + 1, p=self._pmf_B[p]))

        raw_a = qa - d_a + arr_a
        raw_b = qb - d_b + arr_b
        qa_n = min(raw_a, cfg.Q_max)
        qb_n = min(raw_b, cfg.Q_max)
        spill = max(0, raw_a - cfg.Q_max) + max(0, raw_b - cfg.Q_max)

        p_n = p if rng.random() > cfg.eta else (p + 1) % N_REGIMES

        r = -((qa + qb) + cfg.c_switch * is_switch + cfg.c_spill * spill)
        return self.index(qa_n, qb_n, phi_n, e_n, p_n), float(r)

    def reset(self, rng: np.random.Generator) -> int:
        """Exploring start: uniform over legal states."""
        return int(rng.integers(self.nS))
