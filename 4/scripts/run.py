"""Part B experiment driver (PROBLEM_STATEMENT.md §B.10).

Produces results/tables/*.csv and results/plots/*.png.

  1. VI correctness   Sweeps to convergence, empirical contraction rate vs gamma, and the
                      exact-linear-solve cross-check.
  2. Lookahead        Full baseline table by EXACT regret. Myopic-greedy and always-hold coincide,
                      provably (see signal_control/baselines.py) — both rows are kept and the reason said.
  3. Learning         Q-learning vs certainty-equivalence VI on the SAME samples, 5 seeds.
                      Report coverage, regret +/- sd, action-optimality.
  4. Wrong models     Sweep lam_scale over 0.25 .. 2.0, plan with the wrong model, plot regret.
                      Q-learning's regret is the horizontal line: THE CROSSING POINT IS THE ANSWER.
  5. Hyperparameters  alpha schedule, epsilon floor, Q init, gamma.
  6. Anticipation     Decompose Q(s,hold) - Q(s,switch) into immediate and discounted-future terms.

Every policy is scored by exact policy evaluation under the TRUE model — never by rollout averages.
Rollout statistics are reported alongside, because a traffic engineer reads delay and throughput,
not discounted return.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signal_control import baselines as bl
from signal_control.evaluation import exact_policy_value, regret, rollout_stats
from signal_control.mdp import (
    HOLD,
    NIGHT,
    PEAK,
    SWITCH,
    SignalConfig,
    SignalMDP,
)
from signal_control.q_learning import q_learning
from signal_control.value_iteration import certainty_equivalence, value_iteration

LAM_SCALES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
QL_SEEDS = tuple(range(5))

# The learning budget. Q-learning and certainty-equivalence see EXACTLY these samples — same
# seed, same transitions, same order — so any gap between them is what they do with the data,
# not how much of it they got. Short episodes buy exploring starts, and starts are what drive
# state-action coverage here.
QL_EPISODES = 4000
QL_EPISODE_LEN = 100
SWEEP_EPISODES = 1500  # the hyperparameter sweep is 13 configs x 3 seeds; keep it affordable
SWEEP_SEEDS = (0, 1, 2)

# The best configuration the B.10.5 sweep found. B.10.4 asks whether a wrong model beats learning
# from scratch, and that question is only worth answering against the learner's BEST showing —
# beating a badly-initialised Q-learner would prove nothing about models.
QL_TUNED = {"q_init": -560.0, "alpha_power": 1.0}

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
PLOTS = ROOT / "results" / "plots"


def true_model() -> SignalMDP:
    return SignalMDP(SignalConfig())


def believed_model(lam_scale: float) -> SignalMDP:
    """A planner's model with deliberately wrong arrival rates. The ONLY thing that differs."""
    return SignalMDP(SignalConfig(), lam_scale=lam_scale)


# --------------------------------------------------------------------------- job dispatch
#
# Q-learning is a Python loop over mdp.step(), which costs ~70 us a tick — the two categorical
# arrival draws dominate it. The learning experiments are ~10M ticks in total, so they are fanned
# out across processes. Results are keyed by (config, seed) and every run carries its own seed, so
# the answer cannot depend on the order the workers finish in.
#
# Both caches below are per-process and keyed by identity, which is the point: a worker that
# handles several jobs at the same gamma builds the model once and solves for V* once, instead of
# paying 5 seconds of Value Iteration per task.


@lru_cache(maxsize=4)
def _model(gamma: float) -> SignalMDP:
    return SignalMDP(SignalConfig(gamma=gamma))


@lru_cache(maxsize=4)
def _v_star(gamma: float) -> np.ndarray:
    return value_iteration(_model(gamma)).v


def _learning_job(spec: tuple) -> dict:
    """One Q-learning run (+ optional certainty-equivalence on its own samples). Picklable."""
    seed, gamma, n_episodes, kw, with_ce, eval_every = spec
    mdp = _model(gamma)
    v_star = _v_star(gamma)

    ql = q_learning(
        mdp,
        np.random.default_rng(seed),
        n_episodes=n_episodes,
        episode_len=QL_EPISODE_LEN,
        eval_every=eval_every,
        **kw,
    )
    out = {
        "seed": seed,
        "samples": ql.n_steps,
        "coverage_pct": 100.0 * ql.coverage,
        "regret_pct": regret(v_star, exact_policy_value(mdp, ql.policy)),
        "action_optimality_pct": _action_optimality(ql.policy, _greedy_star(gamma), mdp),
        "curve": ql.curve,
    }
    if with_ce:
        # The same transitions Q-learning just consumed, planned on instead of learned from.
        ce = certainty_equivalence(mdp, ql.transitions)
        out["ce_regret_pct"] = regret(v_star, exact_policy_value(mdp, ce.policy))
        out["ce_action_optimality_pct"] = _action_optimality(ce.policy, _greedy_star(gamma), mdp)
    return out


@lru_cache(maxsize=4)
def _greedy_star(gamma: float) -> np.ndarray:
    return value_iteration(_model(gamma)).policy


def _misspecification_job(scale: float) -> dict:
    """Plan on a model with the wrong arrival rates; score the result under the TRUE one."""
    mdp = _model(SignalConfig().gamma)
    pi = value_iteration(believed_model(scale)).policy
    v = exact_policy_value(mdp, pi)
    return {
        "lam_scale": scale,
        "regret_pct": regret(_v_star(mdp.cfg.gamma), v),
        "action_optimality_pct": _action_optimality(pi, _greedy_star(mdp.cfg.gamma), mdp),
    }


def _map(fn, specs: list, jobs: int) -> list:
    if jobs == 1 or len(specs) == 1:
        return [fn(s) for s in specs]
    with ProcessPoolExecutor(max_workers=min(jobs, len(specs))) as pool:
        return list(pool.map(fn, specs))


def _save(df: pd.DataFrame, name: str) -> None:
    path = TABLES / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"\n--- {name} ---")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"[saved] {path.relative_to(ROOT)}")


def _action_optimality(policy: np.ndarray, pi_star: np.ndarray, mdp: SignalMDP) -> float:
    """% of states where the policy picks V*'s action. Only states with a real CHOICE count —
    a state where the masks force the action is not evidence of anything."""
    free = np.array([len(mdp.legal_actions(mdp.unravel(s)[3])) > 1 for s in range(mdp.nS)])
    return 100.0 * float((policy[free] == pi_star[free]).mean())


# --------------------------------------------------------------------------- experiments


def exp1_vi_correctness(mdp: SignalMDP) -> object:
    print("\n[1/6] VI correctness")
    t0 = time.perf_counter()
    vi = value_iteration(mdp)
    secs = time.perf_counter() - t0

    v_pi = exact_policy_value(mdp, vi.policy)  # the cross-check: solve (I - gamma P_pi) V = R_pi
    cross = float(np.abs(vi.v - v_pi).max())
    tail = float(vi.contraction_rates[-20:].mean())

    _save(
        pd.DataFrame(
            [
                {
                    "states": mdp.nS,
                    "legal_pairs": int(mdp.legal.sum()),
                    "sweeps_to_converge": vi.n_sweeps,
                    "seconds": secs,
                    "empirical_contraction_rate": tail,
                    "gamma": mdp.cfg.gamma,
                    "contraction_abs_error": abs(tail - mdp.cfg.gamma),
                    "crosscheck_max_abs_diff": cross,
                    "V_star_mean": float(vi.v.mean()),
                }
            ]
        ),
        "B1_vi_correctness",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].semilogy(vi.contraction_rates, color="C0", lw=0.8)
    axes[0].axhline(mdp.cfg.gamma, color="C3", ls="--", label=f"$\\gamma$ = {mdp.cfg.gamma}")
    axes[0].set_xlabel("sweep $k$")
    axes[0].set_ylabel(r"$\|\Delta V_k\|_\infty / \|\Delta V_{k-1}\|_\infty$")
    axes[0].set_title(f"Empirical contraction rate $\\to$ {tail:.4f}")
    axes[0].legend()

    axes[1].hist(vi.v, bins=60, color="C0")
    axes[1].set_xlabel("$V^*(s)$")
    axes[1].set_ylabel("states")
    axes[1].set_title(f"$V^*$ over {mdp.nS} states (cross-check: {cross:.2e})")
    fig.tight_layout()
    fig.savefig(PLOTS / "B1_vi_correctness.png", dpi=150)
    plt.close(fig)
    return vi


def exp2_lookahead(mdp: SignalMDP, vi) -> None:
    print("\n[2/6] does this problem need lookahead? exact regret of every baseline")
    rng = np.random.default_rng(0)
    policies = {
        "Value Iteration (optimal)": vi.policy,
        "Longest-queue-first": bl.longest_queue_first(mdp),
        "Fixed-time (Webster-style)": bl.fixed_time(mdp),
        "Myopic greedy (gamma = 0)": bl.myopic_greedy(mdp),
        "Always hold": bl.always_hold(mdp),
        "Random (legal actions)": bl.random_legal(mdp, rng),
    }
    rows = []
    for name, pi in policies.items():
        v = exact_policy_value(mdp, pi)
        stats_ = rollout_stats(mdp, pi, np.random.default_rng(1), n_ticks=50_000)
        rows.append(
            {
                "policy": name,
                "V_pi_mean": float(v.mean()),
                "regret_pct": regret(vi.v, v),
                "action_optimality_pct": _action_optimality(pi, vi.policy, mdp),
                **{k: round(val, 3) for k, val in stats_.items()},
            }
        )
    _save(pd.DataFrame(rows), "B2_baselines")

    forced = np.array([mdp.unravel(s)[3] >= mdp.cfg.E_max for s in range(mdp.nS)])
    voluntary = int(((vi.policy == SWITCH) & ~forced).sum())
    print(
        f"\n  Myopic greedy and always-hold are the SAME policy: "
        f"{np.array_equal(policies['Myopic greedy (gamma = 0)'], policies['Always hold'])}. "
        f"R(s,HOLD) >= R(s,SWITCH) everywhere, so gamma=0 never switches voluntarily.\n"
        f"  V*'s policy switches voluntarily in {voluntary} states — every one of them paid for "
        f"out of discounted future value alone."
    )

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for col, (p, pname) in enumerate([(PEAK, "peak"), (1, "off-peak"), (NIGHT, "night")]):
        for row, e in enumerate((2, 5)):
            ax = axes[row, col]
            grid = np.array(
                [
                    [vi.policy[mdp.index(qa, qb, 0, e, p)] for qb in range(mdp.cfg.Q_max + 1)]
                    for qa in range(mdp.cfg.Q_max + 1)
                ]
            )
            ax.imshow(grid, origin="lower", cmap="coolwarm", vmin=0, vmax=1)
            ax.set_title(f"{pname}, e = {e}", fontsize=9)
            if row == 1:
                ax.set_xlabel("$q_B$ (side road)")
            if col == 0:
                ax.set_ylabel("$q_A$ (main road)")
    fig.suptitle(
        "B.11 Optimal action, green on A (blue = hold, red = switch). "
        "Above the diagonal the side queue is longer — yet the switching curve is not the diagonal."
    )
    fig.tight_layout()
    fig.savefig(PLOTS / "B2_policy_slices.png", dpi=150)
    plt.close(fig)


def exp3_learning(mdp: SignalMDP, vi, jobs: int) -> tuple[float, float]:
    print(
        f"\n[3/6] learning: Q-learning vs certainty-equivalence on the SAME samples "
        f"({QL_EPISODES * QL_EPISODE_LEN:,} steps, {len(QL_SEEDS)} seeds)"
    )
    g = mdp.cfg.gamma
    # The DEFAULT learner (with the learning curve) and CE on its own samples, plus the TUNED
    # learner at the identical budget. The tuned arm exists because B.10.4's conclusion could
    # otherwise be dismissed as "your Q-learning was handicapped" — the B.10.5 sweep shows the
    # default Q0 = 0 is not the best setting, and beating a straw learner proves nothing.
    default = _map(
        _learning_job,
        [(s, g, QL_EPISODES, {}, True, 200) for s in QL_SEEDS],
        jobs,
    )
    tuned = _map(
        _learning_job,
        [(s, g, QL_EPISODES, dict(QL_TUNED), False, None) for s in QL_SEEDS],
        jobs,
    )

    rows, curves = [], []
    for d, t in zip(default, tuned, strict=True):
        rows.append(
            {
                "seed": d["seed"],
                "samples": d["samples"],
                "coverage_pct": d["coverage_pct"],
                "QL_regret_pct": d["regret_pct"],
                "QL_tuned_regret_pct": t["regret_pct"],
                "CE_regret_pct": d["ce_regret_pct"],
                "QL_action_optimality_pct": d["action_optimality_pct"],
                "CE_action_optimality_pct": d["ce_action_optimality_pct"],
            }
        )
        curves.append(d["curve"])
        print(f"  seed {d['seed']}: {d['regret_pct']:.2f}% QL / "
              f"{t['regret_pct']:.2f}% QL-tuned / "
              f"{d['ce_regret_pct']:.2f}% CE, coverage {d['coverage_pct']:.1f}%")

    df = pd.DataFrame(rows)
    _save(df, "B3_learning_raw")
    _save(
        pd.DataFrame(
            [
                {
                    "method": "Q-learning (model-free, default hyperparameters)",
                    "mean_regret_pct": df.QL_regret_pct.mean(),
                    "sd_regret_pct": df.QL_regret_pct.std(ddof=1),
                    "mean_action_optimality_pct": df.QL_action_optimality_pct.mean(),
                    "mean_coverage_pct": df.coverage_pct.mean(),
                    "samples": int(df.samples.iloc[0]),
                },
                {
                    "method": "Q-learning (model-free, tuned by the B.10.5 sweep)",
                    "mean_regret_pct": df.QL_tuned_regret_pct.mean(),
                    "sd_regret_pct": df.QL_tuned_regret_pct.std(ddof=1),
                    "mean_action_optimality_pct": float("nan"),
                    "mean_coverage_pct": df.coverage_pct.mean(),
                    "samples": int(df.samples.iloc[0]),
                },
                {
                    "method": "Certainty-equivalence VI (same data)",
                    "mean_regret_pct": df.CE_regret_pct.mean(),
                    "sd_regret_pct": df.CE_regret_pct.std(ddof=1),
                    "mean_action_optimality_pct": df.CE_action_optimality_pct.mean(),
                    "mean_coverage_pct": df.coverage_pct.mean(),
                    "samples": int(df.samples.iloc[0]),
                },
            ]
        ),
        "B3_learning_summary",
    )

    c = np.vstack(curves)
    x = np.arange(1, c.shape[1] + 1) * 200 * QL_EPISODE_LEN
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, np.median(c, axis=0), color="C0", label="Q-learning (median of 5 seeds)")
    ax.fill_between(
        x, np.percentile(c, 25, axis=0), np.percentile(c, 75, axis=0),
        color="C0", alpha=0.2, linewidth=0,
    )
    ax.axhline(
        df.CE_regret_pct.mean(), color="C2", ls="--",
        label=f"Certainty-equivalence VI on the same data ({df.CE_regret_pct.mean():.1f}%)",
    )
    ax.axhline(
        regret(vi.v, exact_policy_value(mdp, bl.longest_queue_first(mdp))),
        color="C3", ls=":", label="Longest-queue-first baseline",
    )
    ax.set_xlabel("environment steps")
    ax.set_ylabel("exact regret of the greedy policy (%)")
    ax.set_title("B.10.3 Learning from experience — same data, two ways of using it")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "B3_learning_curves.png", dpi=150)
    plt.close(fig)
    return float(df.QL_regret_pct.mean()), float(df.QL_tuned_regret_pct.mean())


def exp4_wrong_models(mdp: SignalMDP, vi, ql_regret: float, ql_tuned: float, jobs: int) -> None:
    print("\n[4/6] how wrong can the model be before learning from scratch wins?")
    rows = _map(_misspecification_job, list(LAM_SCALES), jobs)
    for r in rows:
        r["beats_q_learning"] = bool(r["regret_pct"] < ql_regret)
        r["beats_tuned_q_learning"] = bool(r["regret_pct"] < ql_tuned)
    df = pd.DataFrame(rows)
    _save(df, "B4_model_misspecification")

    lost = df[~df.beats_tuned_q_learning].lam_scale.tolist()
    print(
        f"\n  Q-learning with no model at all scores {ql_regret:.2f}% (default) / "
        f"{ql_tuned:.2f}% (tuned). Planning with a WRONG model loses to the TUNED learner only "
        f"at lam_scale = {lost if lost else 'no multiplier tested'}."
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df.lam_scale, df.regret_pct, "o-", color="C0", label="Plan with the wrong model")
    ax.axhline(ql_regret, color="C1", ls="--", label=f"Q-learning, default ({ql_regret:.1f}%)")
    ax.axhline(
        ql_tuned, color="C3", ls="-.",
        label=f"Q-learning, tuned ({ql_tuned:.1f}%) — the fair bar",
    )
    ax.axvline(1.0, color="grey", ls=":", lw=1, label="true rates")
    ax.set_xlabel(r"assumed arrival rate / true arrival rate ($\lambda$ multiplier)")
    ax.set_ylabel("exact regret under the TRUE model (%)")
    ax.set_title("B.10.4 How wrong must a model be before learning from scratch beats it?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "B4_model_misspecification.png", dpi=150)
    plt.close(fig)


def exp5_hyperparameters(mdp: SignalMDP, vi, jobs: int) -> None:
    print(f"\n[5/6] Q-learning hyperparameter sweep ({SWEEP_EPISODES * QL_EPISODE_LEN:,} steps/run)")
    configs = [
        ("baseline (alpha=(1+n)^-0.7, eps_floor=0.05, Q0=0)", {}),
        ("alpha power 0.5", {"alpha_power": 0.5}),
        ("alpha power 1.0", {"alpha_power": 1.0}),
        ("constant alpha 0.1", {"alpha_const": 0.1}),
        ("constant alpha 0.5", {"alpha_const": 0.5}),
        ("eps floor 0.0", {"eps_floor": 0.0}),
        ("eps floor 0.2", {"eps_floor": 0.2}),
        ("eps floor 0.5", {"eps_floor": 0.5}),
        # Every reward in this MDP is <= 0, so V* < 0 everywhere and the BASELINE's Q0 = 0 is
        # already the optimistic initialisation. The contrast is against a neutral and a
        # pessimistic one, not against a second copy of the baseline.
        ("neutral Q0 = -560 (~ mean V*)", {"q_init": -560.0}),
        ("pessimistic Q0 = -2000", {"q_init": -2000.0}),
        ("no exploring starts", {"exploring_starts": False}),
        ("TUNED: Q0 = -560 and alpha power 1.0", dict(QL_TUNED)),
    ]
    # gamma is a property of the objective, not a knob on the learner: it needs its own MDP and
    # its own V*, so its regret is measured against a different optimum from every other row.
    gamma_configs = [
        (f"gamma = {g} (regret measured against that gamma's own V*)", g, {})
        for g in (0.5, 0.9, 0.999)
    ]
    all_configs = [(name, mdp.cfg.gamma, kw) for name, kw in configs] + gamma_configs

    specs = [
        (seed, g, SWEEP_EPISODES, kw, False, None)
        for _name, g, kw in all_configs
        for seed in SWEEP_SEEDS
    ]
    results = _map(_learning_job, specs, jobs)

    rows = []
    for i, (name, _g, _kw) in enumerate(all_configs):
        block = results[i * len(SWEEP_SEEDS) : (i + 1) * len(SWEEP_SEEDS)]
        regrets = [r["regret_pct"] for r in block]
        covers = [r["coverage_pct"] for r in block]
        rows.append(
            {
                "config": name,
                "mean_regret_pct": float(np.mean(regrets)),
                "sd_regret_pct": float(np.std(regrets, ddof=1)),
                "mean_coverage_pct": float(np.mean(covers)),
            }
        )
        print(f"  {name:52s} regret {rows[-1]['mean_regret_pct']:6.2f}% "
              f"+/- {rows[-1]['sd_regret_pct']:4.2f}, coverage {rows[-1]['mean_coverage_pct']:.0f}%")

    df = pd.DataFrame(rows)
    _save(df, "B5_hyperparameters")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(df))
    ax.barh(y, df.mean_regret_pct, xerr=df.sd_regret_pct, color="C0", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(df.config, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("exact regret (%), mean +/- sd over 3 seeds")
    ax.set_title("B.10.5 Q-learning hyperparameter sweep")
    fig.tight_layout()
    fig.savefig(PLOTS / "B5_hyperparameters.png", dpi=150)
    plt.close(fig)


def exp6_anticipation(mdp: SignalMDP, vi) -> None:
    print("\n[6/6] evidence of anticipation: what actually flips the decision?")
    gamma = mdp.cfg.gamma
    future = gamma * (mdp.P @ vi.v).reshape(mdp.nS, mdp.nA)  # the discounted-future term

    forced = np.array([mdp.unravel(s)[3] >= mdp.cfg.E_max for s in range(mdp.nS)])
    voluntary = np.flatnonzero((vi.policy == SWITCH) & ~forced)

    # Only states with a genuine CHOICE are meaningful here: where an action is masked out its
    # row of P is empty and its R is 0, which is not a reward, it is an absence.
    both = mdp.legal[:, HOLD] & mdp.legal[:, SWITCH]

    imm = mdp.R[:, HOLD] - mdp.R[:, SWITCH]  # immediate advantage of holding: >= 0 ALWAYS (proved)
    fut = future[:, HOLD] - future[:, SWITCH]  # discounted-future advantage of holding
    total = imm + fut  # = Q(s,hold) - Q(s,switch); < 0 exactly where switching wins

    # Lead with the counter-intuitive switches — the ones where the green queue is the LONGER of
    # the two, so every reactive rule would hold — then fill with the largest voluntary switches.
    counter = sorted(
        (s for s in voluntary if _is_counterintuitive(mdp, s)), key=lambda s: total[s]
    )
    rest = sorted((s for s in voluntary if s not in set(counter)), key=lambda s: total[s])
    chosen = (counter + rest)[:10]

    rows = []
    for s in chosen:
        qa, qb, phi, e, p = mdp.unravel(s)
        rows.append(
            {
                "state (qA,qB,phi,e,p)": f"({qa},{qb},{'A' if phi == 0 else 'B'},{e},"
                f"{['peak', 'off', 'night'][p]})",
                "green_queue_is_longer": _is_counterintuitive(mdp, s),
                "R(hold) - R(switch)": imm[s],
                "gamma*[EV(hold) - EV(switch)]": fut[s],
                "Q(hold) - Q(switch)": total[s],
                "optimal": "SWITCH" if total[s] < 0 else "HOLD",
            }
        )
    _save(pd.DataFrame(rows), "B6_anticipation")

    # The case §B.10.6 actually names: the optimal policy HOLDS the main road green while the side
    # queue is the longer one. Every reactive rule switches here; V* does not. This is the other
    # half of the anticipation story, and it is the half a traffic engineer will argue with.
    held = np.flatnonzero((vi.policy == HOLD) & both & _red_longer(mdp))
    hold_rows = []
    for s in sorted(held, key=lambda s: -total[s])[:10]:
        qa, qb, phi, e, p = mdp.unravel(s)
        q_green, q_red = (qa, qb) if phi == 0 else (qb, qa)
        hold_rows.append(
            {
                "state (qA,qB,phi,e,p)": f"({qa},{qb},{'A' if phi == 0 else 'B'},{e},"
                f"{['peak', 'off', 'night'][p]})",
                "green_queue": q_green,
                "red_queue (longer!)": q_red,
                "R(hold) - R(switch)": imm[s],
                "gamma*[EV(hold) - EV(switch)]": fut[s],
                "Q(hold) - Q(switch)": total[s],
                "longest_queue_first_would": "SWITCH",
                "optimal": "HOLD",
            }
        )
    _save(pd.DataFrame(hold_rows), "B6_counterintuitive_holds")

    print(
        f"\n  Immediate term favours HOLD in {int((imm[both] >= 0).sum())}/{int(both.sum())} "
        f"states with a real choice (min = {imm[both].min():.3f}) — it NEVER argues for "
        f"switching, as proved.\n"
        f"  Yet V* switches voluntarily in {voluntary.size} states. In every one, the "
        f"discounted-future term is what flips the sign.\n"
        f"  And in {held.size} states V* HOLDS while the RED queue is longer — the exact case "
        f"longest-queue-first gets wrong, and the reason it carries "
        f"{regret(vi.v, exact_policy_value(mdp, bl.longest_queue_first(mdp))):.1f}% regret."
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(imm[voluntary], fut[voluntary], s=8, alpha=0.5, color="C3", label="V* switches (voluntary)")
    hold_states = np.flatnonzero((vi.policy == HOLD) & both)
    ax.scatter(imm[hold_states], fut[hold_states], s=8, alpha=0.35, color="C0", label="V* holds")
    lim = np.array([min(imm[both].min(), 0) - 0.5, imm[both].max() + 0.5])
    ax.plot(lim, -lim, "k--", lw=1, label="$Q(\\mathrm{hold}) = Q(\\mathrm{switch})$")
    ax.set_xlabel("immediate advantage of holding,  $R(s,\\mathrm{hold}) - R(s,\\mathrm{switch}) \\geq 0$")
    ax.set_ylabel("discounted-future advantage of holding")
    ax.set_title("B.10.6 Every voluntary switch is bought with future value alone")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "B6_anticipation.png", dpi=150)
    plt.close(fig)


def _is_counterintuitive(mdp: SignalMDP, s: int) -> bool:
    """A switch away from the LONGER queue: the reactive rule would hold, the optimal one does not."""
    qa, qb, phi, _, _ = mdp.unravel(s)
    q_green, q_red = (qa, qb) if phi == 0 else (qb, qa)
    return q_green > q_red


def _red_longer(mdp: SignalMDP) -> np.ndarray:
    """Mask of states where the queue waiting on RED is strictly longer than the one on green —
    the states where longest-queue-first insists on switching."""
    out = np.zeros(mdp.nS, dtype=bool)
    for s in range(mdp.nS):
        qa, qb, phi, _, _ = mdp.unravel(s)
        q_green, q_red = (qa, qb) if phi == 0 else (qb, qa)
        out[s] = q_red > q_green
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Part B experiments (PROBLEM_STATEMENT.md §B.10)")
    ap.add_argument("--quick", action="store_true", help="smaller learning budget, for smoke tests")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()

    global QL_EPISODES, SWEEP_EPISODES
    if args.quick:
        QL_EPISODES, SWEEP_EPISODES = 400, 200

    TABLES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    mdp = true_model()
    print(
        f"Part B — |S| = {mdp.nS}, legal (s,a) pairs = {int(mdp.legal.sum())}, "
        f"gamma = {mdp.cfg.gamma} (horizon ~{1 / (1 - mdp.cfg.gamma):.0f} ticks ~ "
        f"{1 / (1 - mdp.cfg.gamma) * mdp.cfg.dt / 60:.0f} min)"
    )

    vi = exp1_vi_correctness(mdp)
    exp2_lookahead(mdp, vi)
    ql_regret, ql_tuned = exp3_learning(mdp, vi, args.jobs)
    exp4_wrong_models(mdp, vi, ql_regret, ql_tuned, args.jobs)
    exp5_hyperparameters(mdp, vi, args.jobs)
    exp6_anticipation(mdp, vi)

    print(f"\nDone. Tables -> {TABLES.relative_to(ROOT)}, plots -> {PLOTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
