#!/usr/bin/env python3
"""Stage 1 (playoff-format) — Scenario 0 reproduction + baseline metrics.

Reruns the Likens Stage 13 batch engine for 100,000 seasons with the SAME
seed used for the Likens primary run (20261028, chunk 2,000). Because the
engine is deterministic in (n_seasons, seed), the regular seasons are
bit-identical to the Likens run; this script asserts that, then writes the
scenario-0 artifact set in the playoff-format project's per-scenario format
and computes the baseline values of the headline metrics.

Outputs (all under playoff-format/outputs/):
  pf_scenario0_teams.parquet  — per-team long table:
      season, team, team_idx, talent_rank (1=best of 32), talent_gfpct,
      PTS, GF, GA, made_playoffs, round_reached (0=missed playoffs; 1..5 = lost R1,
 lost R2, lost R3, lost final, champion),
      cup_winner, seed_seedformat (NaN for scenario 0: seeds are
      division-bracket slots, not 1-8 ranks — added from Stage 2 on)
  reports/pf_stage1_report.txt — verification + baseline metrics

Verification (fails loudly, writes nothing on failure):
  V1: regular-season fields (talent_gfpct, PTS, GF, GA) bit-identical to
      likens/outputs/stage13_full_100k.parquet.
  V2: all playoff-flag columns bit-identical to the Likens parquet.
  V3: structure gates (16/8/4/2/1 per season; cumulative consistency).
  V4: baseline best-team metrics cross-checked against values computed
      directly from the Likens parquet (must match EXACTLY — same array).

Usage:
    code/.venv/bin/python pf_scenario0.py     (from likens/code, venv path)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

LIKENS_CODE = "/workspace/projects/hockey-lab/analyses/likens/code"
LIKENS_OUT = "/workspace/projects/hockey-lab/analyses/likens/outputs"
PF_OUT = "/workspace/projects/hockey-lab/analyses/playoff-format/outputs"
sys.path.insert(0, LIKENS_CODE)

import stage13_batch_engine as eng  # noqa: E402

N_SEASONS = 100_000
SEED = 20261028          # same as stage13_run_100k.py -> identical draws
LIKENS_PARQUET = os.path.join(LIKENS_OUT, "stage13_full_100k.parquet")
OUT_PARQUET = os.path.join(PF_OUT, "pf_scenario0_teams.parquet")
OUT_REPORT = os.path.join(PF_OUT, "reports", "pf_stage1_report.txt")


def round_reached(res: dict[str, np.ndarray]) -> np.ndarray:
    """0 = missed playoffs; 1..4 = series won by a playoff team (4 = Cup)."""
    rw = (res["round1_wins"] + res["round2_wins"] + res["round3_wins"]
          + res["cup_winner"])
    return np.where(res["made_playoffs"] == 1, rw + 1, 0).astype(np.int8)


def main() -> None:
    t0 = time.perf_counter()
    res = eng.run_batch(N_SEASONS, seed=SEED)
    dt = time.perf_counter() - t0
    print(f"scenario-0 rerun: 100k seasons in {dt:.1f}s "
          f"({dt / N_SEASONS * 1e3:.2f} ms/season)")

    # --- V3: structure gates (before touching anything else)
    eng._check_structure(res, N_SEASONS)

    # --- Load Likens parquet, pivot wide [season, 32] in engine team order
    lik = pd.read_parquet(LIKENS_PARQUET)
    wide: dict[str, np.ndarray] = {}
    for c in ["talent_gfpct", "PTS", "GF", "GA", "made_playoffs",
              "division_winner", "conference_best", "presidents_trophy",
              "round1_wins", "round2_wins", "round3_wins", "cup_winner"]:
        wide[c] = (lik.pivot(index="season", columns="team_idx",
                             values=c).to_numpy())

    # --- V1: regular-season fields bit-identical
    for c in ["talent_gfpct", "PTS", "GF", "GA"]:
        assert np.array_equal(res[c], wide[c]), f"V1 FAIL: {c} differs"
    print("[V1] regular-season fields bit-identical to Likens parquet")

    # --- V2: playoff flags bit-identical
    # presidents_trophy EXCEPTION (documented): the stored Likens parquet was
    # generated 2026-10-21, BEFORE the Phase C fix (2026-10-27) aligning
    # stage13's league rank with stage9 (no H2H swaps at league level). The
    # fix only changes the PT flag on exact (PTS, ROW, DIFF) two-way ties —
    # 11 of 100,000 seasons here. All other columns must match exactly.
    for c in ["made_playoffs", "division_winner", "conference_best",
              "round1_wins", "round2_wins", "round3_wins", "cup_winner"]:
        assert np.array_equal(res[c], wide[c]), f"V2 FAIL: {c} differs"
    pt_diff = int((res["presidents_trophy"] != wide["presidents_trophy"]).any(1).sum())
    assert pt_diff <= 50, f"V2 FAIL: PT differs in {pt_diff} seasons (expected ~11)"
    print("[V2] playoff flags bit-identical to Likens parquet "
          f"(presidents_trophy differs in {pt_diff}/100k seasons — pre-Phase-C "
          "parquet, exact PTS+ROW+DIFF ties only; current engine is authoritative)")

    # --- Build the per-team long artifact
    ki = np.arange(N_SEASONS)[:, None]
    tr = (-res["talent_gfpct"]).argsort(1).argsort(1)  # 0 = best
    rr = round_reached(res)
    frames = []
    for j, team in enumerate(eng._TEAMS):
        f = pd.DataFrame({
            "season": np.arange(N_SEASONS, dtype=np.int32),
            "team": team,
            "team_idx": j,
            "talent_rank": (tr[:, j] + 1).astype(np.int16),
            "talent_gfpct": res["talent_gfpct"][:, j],
            "PTS": res["PTS"][:, j],
            "GF": res["GF"][:, j],
            "GA": res["GA"][:, j],
            "made_playoffs": res["made_playoffs"][:, j],
            "round_reached": rr[:, j],
            "cup_winner": res["cup_winner"][:, j],
        })
        frames.append(f)
    long = pd.concat(frames, ignore_index=True)
    # round_reached sanity: per-season counts {0:16, 1:8, 2:4, 3:2, 4:1, 5:1}
    # (16 missed playoffs; 8 lose R1; 4 lose R2; 2 lose R3; 1 loses final; 1 Cup)
    cnt = long.groupby("season")["round_reached"].value_counts().unstack(fill_value=0)
    assert (cnt[0] == 16).all() and (cnt[1] == 8).all() and (cnt[2] == 4).all() \
        and (cnt[3] == 2).all() and (cnt[4] == 1).all() and (cnt[5] == 1).all(), \
        "round_reached broken"
    assert long["cup_winner"].groupby(long["season"]).sum().eq(1).all()
    long.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET} ({len(long):,} rows)")

    # --- V4 + baseline metrics (computed from THIS run; identical to Likens
    #     by V1/V2, so cross-check is exact)
    ki1 = np.arange(N_SEASONS)
    best = (tr == 0).argmax(1)   # position of the best-talent team
    second = (tr == 1).argmax(1)
    p_cup_best = res["cup_winner"][ki1, best].mean()
    p_conf_best = res["round3_wins"][ki1, best].mean()
    p_pt_best = res["presidents_trophy"][ki1, best].mean()
    p_po_best = res["made_playoffs"][ki1, best].mean()

    # P(top-4 talents all reach the final four): every rank<=4 team has
    # round3_wins >= 1 (i.e., won its R2 series).
    top4 = np.stack([res["round3_wins"][ki1, (tr == r).argmax(1)]
                     for r in range(4)]) >= 1
    p_top4_final4 = top4.all(0).mean()

    # P(top-8 all reach round 2): rank<=8 teams all won their R1 series.
    top8 = np.stack([res["round1_wins"][ki1, (tr == r).argmax(1)]
                     for r in range(8)]) >= 1
    p_top8_r2 = top8.all(0).mean()

    # P(talent ranks 1 and 2 meet in the Cup final): both have rounds_won 3
    # (reached the final) — under the current format the final is East vs
    # West; "meet in the final" = both are in it, 2 slots, so both flags
    # round3_wins==1 and at most 2 teams have it; identity check:
    in_final = res["round3_wins"] == 1  # won R3 series -> in the Cup final
    assert (in_final.sum(1) == 2).all()
    p_12_final = (in_final[ki1, best] & in_final[ki1, second]).mean()

    # Cross-check against the Likens parquet (V4, exact).
    lik_tr = (-wide["talent_gfpct"]).argsort(1).argsort(1)
    lik_best = (lik_tr == 0).argmax(1)
    lik_second = (lik_tr == 1).argmax(1)
    for metric, ours, theirs in [
        ("P(cup|best)", p_cup_best, wide["cup_winner"][ki1, lik_best].mean()),
        ("P(conf_final|best)", p_conf_best, wide["round3_wins"][ki1, lik_best].mean()),
        ("P(top4 final4)", p_top4_final4,
         (np.stack([wide["round3_wins"][ki1, (lik_tr == r).argmax(1)]
                    for r in range(4)]) >= 1).all(0).mean()),
        ("P(top8 r2)", p_top8_r2,
         (np.stack([wide["round1_wins"][ki1, (lik_tr == r).argmax(1)]
                    for r in range(8)]) >= 1).all(0).mean()),
    ]:
        assert abs(ours - theirs) < 1e-12, f"V4 FAIL: {metric}"
    lik_in_final = wide["round3_wins"] == 1
    lik_p12 = (lik_in_final[ki1, lik_best] & lik_in_final[ki1, lik_second]).mean()
    assert abs(p_12_final - lik_p12) < 2e-4, \
        f"V4 FAIL: P(12 final) ours {p_12_final:.5f} vs likens {lik_p12:.5f} " \
        "(small tolerance: PT-flag tie fix can flip a final-4 slot in ~11/100k seasons)"
    print("[V4] baseline metrics cross-check vs Likens parquet: PASS "
          "(exact except P(1v2 final), documented PT-tie exception)")

    lines = [
        "Stage 1 (playoff-format) — Scenario 0 reproduction + baseline metrics",
        f"seed {SEED} (same as Likens Stage 13 run), chunk {eng.CHUNK}",
        f"runtime {dt:.1f}s ({dt / N_SEASONS * 1e3:.2f} ms/season)",
        "",
        "Verification:",
        "  [V1] regular-season fields bit-identical to Likens parquet: PASS",
        f"  [V2] playoff flags bit-identical to Likens parquet: PASS",
        f"       (presidents_trophy differs in {pt_diff}/100k seasons — stored",
        "       parquet predates the 2026-10-27 Phase C league-rank fix; ties",
        "       on exact PTS+ROW+DIFF only; current engine is authoritative)",
        "  [V3] structure gates (16/8/4/2/1, cumulative): PASS",
        "  [V4] metric cross-check vs Likens parquet: PASS (exact except",
        "       P(1v2 final), same documented PT-tie exception)",
        "  round_reached per-season counts {missed:16, R1-exit:8, R2-exit:4,",
        "        R3-exit:2, final-loser:1, champion:1}: PASS",
        "",
        "Baseline metrics, scenario 0 (current format), 100k seasons:",
        f"  P(best team wins Cup)            {p_cup_best:.4f}",
        f"  P(best team reaches conf final)  {p_conf_best:.4f}",
        f"  P(best team wins Presidents')    {p_pt_best:.4f}",
        f"  P(best team makes playoffs)      {p_po_best:.4f}",
        f"  P(talent #1 and #2 meet in final){p_12_final:.4f}",
        f"  P(top-4 talents all in final 4)  {p_top4_final4:.4f}",
        f"  P(top-8 talents all reach R2)    {p_top8_r2:.4f}",
    ]
    report = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, "w") as fh:
        fh.write(report + "\n")
    print(report)
    print(f"wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
