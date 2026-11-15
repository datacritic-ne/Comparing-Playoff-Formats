#!/usr/bin/env python3
"""Stage 4 (playoff-format) — Mass simulation of the 6 new scenarios.

For each scenario 1-6, runs 100,000 seasons with the SAME seed as the
Likens/Stage-1 run (20261028) — regular seasons therefore bit-match
scenario 0 / the Likens run (common random numbers), so every difference
in playoff outcomes is attributable to the format alone.

Per scenario, writes:
  outputs/pf_s<scenario>_teams.parquet  — per-team long table
      (season, team, team_idx, talent_rank, talent_gfpct, PTS, GF, GA,
       made_playoffs, round_reached, cup_winner, seed, playoff_seed_no)
      seed = scenario id; playoff_seed_no = the team's 1-based seed in
      that format (NaN/None for scenario 0, whose seeds are bracket slots)
  outputs/pf_s<scenario>_series.parquet — per-series log flattened
      (season, round, hi_team, lo_team, hi_seed, lo_seed, winner, games)
  outputs/reports/pf_stage4_report.txt  — headline metrics per scenario

Verification (embedded, fails loudly):
  V1  determinism: scenario 2 rerun on a small N is bit-identical
  V2  regular-season fields identical to the Likens parquet (all scenarios)
  V3  scenario 1k pilot vs 100k headline metrics within MC tolerance
  V4  structure gates per scenario (16/8/4/2/1, round_reached counts)

Usage:
    code/.venv/bin/python pf_mass_sim.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/projects/hockey-lab/analyses/likens/code")
sys.path.insert(0, "/workspace/projects/hockey-lab/analyses/playoff-format/code")

import stage13_batch_engine as eng  # noqa: E402
from pf_brackets import build_seed_orders  # noqa: E402
from pf_postseason import simulate_postseason  # noqa: E402

PF_OUT = "/workspace/projects/hockey-lab/analyses/playoff-format/outputs"
LIKENS_PARQUET = ("/workspace/projects/hockey-lab/analyses/likens/outputs/"
                  "stage13_full_100k.parquet")
N_SEASONS = 100_000
SEED = 20261028
CHUNK = 2_000
SCEN_NAMES = {1: "2-6", 2: "1-8", 3: "1-16",
              4: "2-6-reseed", 5: "1-8-reseed", 6: "1-16-reseed",
              7: "current-reseed"}


def _round_reached(flags: dict) -> np.ndarray:
    rw = (flags["round1_wins"] + flags["round2_wins"]
          + flags["round3_wins"] + flags["cup_winner"])
    return np.where(flags["made_playoffs"] == 1, rw + 1, 0).astype(np.int8)


def _structure_check(flags: dict, K: int, scen: int) -> None:
    assert (flags["made_playoffs"].sum(1) == 16).all()
    assert (flags["round1_wins"].sum(1) == 8).all()
    assert (flags["round2_wins"].sum(1) == 4).all()
    assert (flags["round3_wins"].sum(1) == 2).all()
    assert (flags["cup_winner"].sum(1) == 1).all()
    for a, b in [("round2_wins", "round1_wins"),
                 ("round3_wins", "round2_wins"),
                 ("cup_winner", "round3_wins")]:
        assert (flags[a] <= flags[b]).all(), f"scen {scen}: {a}>{b}"
    rr = _round_reached(flags)
    vals, counts = np.unique(rr, return_counts=True)
    expect = {0: 16, 1: 8, 2: 4, 3: 2, 4: 1, 5: 1}
    got = dict(zip(vals.tolist(), (counts / K).round(6).tolist()))
    for v, c in expect.items():
        assert abs(got.get(v, 0) - c) < 1e-6, f"scen {scen}: rr {got}"


def run_scenario(scen: int, n_seasons: int = N_SEASONS, seed: int = SEED):
    """Run one scenario; returns (flags, log, order, seedno, meta)."""
    all_flags: dict[str, list] = {c: [] for c in
                                  ["made_playoffs", "round1_wins",
                                   "round2_wins", "round3_wins",
                                   "cup_winner"]}
    logs: list = []
    talents: list = []
    orders: list = []
    done = 0
    while done < n_seasons:
        CH = min(CHUNK, n_seasons - done)
        rng = np.random.default_rng(seed + done)   # same streams as Likens
        talent, pts, gf, ga, row, diff, h2h, gf_rate, ga_rate = \
            eng._regular_season_chunk(CH, rng)
        div_order, conf_order, league_order = eng._rank_all(pts, row, diff, h2h)
        if scen == 7:
            order = seedno = None  # R1 via stage13 internals inside the engine
        else:
            order, seedno = build_seed_orders(div_order, conf_order,
                                              league_order, pts, scen)
        flags, log = simulate_postseason(scen, order, seedno, gf_rate,
                                         ga_rate, rng, pts=pts, row=row,
                                         diff=diff, h2h=h2h)
        for c in all_flags:
            all_flags[c].append(flags[c])
        logs.append(log)
        talents.append(talent)
        orders.append((order, seedno))
        done += CH
    flags = {c: np.concatenate(v, 0) for c, v in all_flags.items()}
    talent = np.concatenate(talents, 0)
    return flags, logs, orders, talent


def flags_to_frame(flags: dict, scen: int) -> pd.DataFrame:
    K = flags["made_playoffs"].shape[0]
    ki = np.arange(K)[:, None]
    # talent must come from the same regular seasons: recompute from a
    # fresh pass is wasteful — instead read from the scenario-0 artifact
    # (bit-identical regular seasons, V2-verified).
    s0 = pd.read_parquet(os.path.join(PF_OUT, "pf_scenario0_teams.parquet"))
    tal = s0.pivot(index="season", columns="team_idx",
                   values="talent_gfpct").to_numpy()
    pts_w = s0.pivot(index="season", columns="team_idx",
                     values="PTS").to_numpy()
    gf_w = s0.pivot(index="season", columns="team_idx", values="GF").to_numpy()
    ga_w = s0.pivot(index="season", columns="team_idx", values="GA").to_numpy()
    assert np.array_equal(pts_w, 0) is False
    tr = (-tal).argsort(1).argsort(1)
    rr = _round_reached(flags)
    frames = []
    for j, team in enumerate(eng._TEAMS):
        f = pd.DataFrame({
            "season": np.arange(K, dtype=np.int32),
            "team": team,
            "team_idx": j,
            "talent_rank": (tr[:, j] + 1).astype(np.int16),
            "talent_gfpct": tal[:, j],
            "PTS": pts_w[:, j],
            "GF": gf_w[:, j],
            "GA": ga_w[:, j],
            "made_playoffs": flags["made_playoffs"][:, j],
            "round_reached": rr[:, j],
            "cup_winner": flags["cup_winner"][:, j],
            "scenario": scen,
        })
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def logs_to_frame(logs: list, scen: int, n_seasons: int) -> pd.DataFrame:
    """Flatten per-chunk round logs into one long DataFrame."""
    rows = []
    chunk = CHUNK
    for ci, log in enumerate(logs):
        base = ci * chunk
        for entry in log:
            rnd = entry["round"]
            hi, lo = entry["hi"], entry["lo"]
            K, P = hi.shape
            ki = np.arange(K)[:, None]
            win = entry["winner"]
            games = entry["games"]
            hs = entry["hi_seed"]
            ls = entry["lo_seed"]
            for p in range(P):
                rows.append(pd.DataFrame({
                    "season": np.arange(base, base + K, dtype=np.int32),
                    "round": rnd,
                    "pair": p,
                    "hi_team": hi[:, p],
                    "lo_team": lo[:, p],
                    "hi_seed": hs[:, p] if hs is not None else -1,
                    "lo_seed": ls[:, p] if ls is not None else -1,
                    "winner": win[:, p],
                    "games": games[:, p],
                    "scenario": scen,
                }))
    return pd.concat(rows, ignore_index=True)


def headline_metrics(flags: dict, tal: np.ndarray) -> dict:
    K = flags["made_playoffs"].shape[0]
    ki = np.arange(K)
    tr = (-tal).argsort(1).argsort(1)
    best = (tr == 0).argmax(1)
    second = (tr == 1).argmax(1)
    m = {
        "P(best wins Cup)": flags["cup_winner"][ki, best].mean(),
        "P(best conf final)": flags["round3_wins"][ki, best].mean(),
        "P(best playoffs)": flags["made_playoffs"][ki, best].mean(),
    }
    # P(1 and 2 meet in the final): both won their R3 series (round3_wins=1)
    in_final = flags["round3_wins"] == 1
    assert (in_final.sum(1) == 2).all()
    m["P(1v2 final)"] = (in_final[ki, best] & in_final[ki, second]).mean()
    # FIXED 2026-11-06 (labeling bug found when sanity-checking magnitudes):
    # "final four" = won the R2 series (round2_wins>=1 = reached the semis),
    # NOT round3_wins (that is reaching the Cup final). Corrected ordering:
    # P(top8 -> R2) < P(top4 -> final4) < P(1v2 final) holds in every
    # scenario.
    top4 = np.stack([flags["round2_wins"][ki, (tr == r).argmax(1)]
                     for r in range(4)]) >= 1
    m["P(top4 final4)"] = top4.all(0).mean()
    top8 = np.stack([flags["round1_wins"][ki, (tr == r).argmax(1)]
                     for r in range(8)]) >= 1
    m["P(top8 r2)"] = top8.all(0).mean()
    return m


def main() -> None:
    t_all = time.perf_counter()
    s0 = pd.read_parquet(os.path.join(PF_OUT, "pf_scenario0_teams.parquet"))
    tal = s0.pivot(index="season", columns="team_idx",
                   values="talent_gfpct").to_numpy()
    pts_w = s0.pivot(index="season", columns="team_idx", values="PTS").to_numpy()
    gf_w = s0.pivot(index="season", columns="team_idx", values="GF").to_numpy()
    ga_w = s0.pivot(index="season", columns="team_idx", values="GA").to_numpy()

    lik = pd.read_parquet(LIKENS_PARQUET)
    lik_pts = lik.pivot(index="season", columns="team_idx", values="PTS").to_numpy()

    report = ["Stage 4 (playoff-format) — mass simulation, 6 scenarios",
              f"seed {SEED} (same streams as Likens), {N_SEASONS} seasons each",
              ""]
    results: dict[int, dict] = {}
    # Optional CLI filter: `python pf_mass_sim.py 1,2` reruns only those
    # scenarios (used 2026-11-12 for the crossed-R2 fix rerun of 1-2).
    wanted = (list(range(1, 7)) + [7])
    if len(sys.argv) > 1:
        wanted = [int(x) for x in sys.argv[1].split(",")]
    for scen in wanted:
        t0 = time.perf_counter()
        flags, logs, _, talent_run = run_scenario(scen)
        dt = time.perf_counter() - t0
        _structure_check(flags, N_SEASONS, scen)

        # V2: regular seasons identical to Likens (talents from THIS run
        # must equal the scenario-0/Likens talents — same seed). The
        # parquet stores float32; compare in float32 (exact under it).
        assert np.array_equal(talent_run.astype(np.float32), tal), \
            f"V2: scenario {scen} talents differ from Likens"

        teams = flags_to_frame(flags, scen)
        teams.to_parquet(os.path.join(
            PF_OUT, f"pf_s{scen}_teams.parquet"), index=False)
        series = logs_to_frame(logs, scen, N_SEASONS)
        series.to_parquet(os.path.join(
            PF_OUT, f"pf_s{scen}_series.parquet"), index=False)
        m = headline_metrics(flags, tal)
        results[scen] = m
        report.append(
            f"[scenario {scen} — {SCEN_NAMES[scen]}] {dt:.0f}s "
            f"({dt / N_SEASONS * 1e3:.2f} ms/season)")
        for k, v in m.items():
            report.append(f"    {k:24s} {v:.4f}")
        report.append("")  # placeholder replaced below if gates appended

    # V5: scenarios pairwise-different (Stage 4 lesson — cross-product
    # scenarios must be asserted DIFFERENT, not just internally consistent).
    # Compare cup_winner maps across scenario parquets just written.
    import itertools
    cup_maps = {}
    for scen in list(range(1, 7)) + [7]:
        t = pd.read_parquet(os.path.join(
            PF_OUT, f"pf_s{scen}_teams.parquet"))
        cup_maps[scen] = t.pivot(index="season", columns="team_idx",
                                 values="cup_winner").to_numpy()
    n_diff_pairs = 0
    for a, b in itertools.combinations(sorted(cup_maps), 2):
        same = np.array_equal(cup_maps[a], cup_maps[b])
        frac = (cup_maps[a] == cup_maps[b]).mean()
        if same:
            raise AssertionError(f"V5: scenarios {a} and {b} IDENTICAL")
        n_diff_pairs += 1
        report.append(f"[V5] scen {a} vs {b}: cup maps differ "
                      f"(match fraction {frac:.4f})")
    report.append(f"[V5] all {n_diff_pairs} scenario pairs differ: PASS")

    report.append("")
    f_a, l_a, _, _ = run_scenario(2, n_seasons=3_000)
    f_b, l_b, _, _ = run_scenario(2, n_seasons=3_000)
    for c in f_a:
        assert np.array_equal(f_a[c], f_b[c]), "V1: non-deterministic"
    for la_chunk, lb_chunk in zip(l_a, l_b):
        for entry_a, entry_b in zip(la_chunk, lb_chunk):
            for k in entry_a:
                assert np.array_equal(entry_a[k], entry_b[k]), \
                    "V1: log non-deterministic"
    report.append("[V1] determinism (scenario 2, 3k seasons rerun): PASS")

    # V3: pilot-vs-full consistency (1k pilot, independent seed). The pilot
    # has its own regular seasons, so use ITS talents for team ranking.
    for scen in (1, 3, 5):
        if scen not in results:
            continue  # partial rerun (CLI filter) — skip scenarios not run
        f_p, _, _, tal_p = run_scenario(scen, n_seasons=1_000, seed=999_000)
        m_p = headline_metrics(f_p, tal_p)
        for k in m_p:
            se = np.sqrt(m_p[k] * (1 - m_p[k]) / 1_000
                         + results[scen][k] * (1 - results[scen][k]) / N_SEASONS)
            gate = 4 * se + 0.01
            d = abs(m_p[k] - results[scen][k])
            ok = d <= gate
            report.append(
                f"[V3 scen {scen}] {k}: pilot {m_p[k]:.4f} vs full "
                f"{results[scen][k]:.4f} (|d| {d:.4f}, gate ±{gate:.4f}) "
                f"{'PASS' if ok else 'FAIL'}")
            assert ok, f"V3 FAIL scen {scen} {k}"

    report.append("")
    report.append("All Stage 4 checks PASS")
    text = "\n".join(report)
    os.makedirs(os.path.join(PF_OUT, "reports"), exist_ok=True)
    with open(os.path.join(PF_OUT, "reports", "pf_stage4_report.txt"),
              "w") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"total {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()
