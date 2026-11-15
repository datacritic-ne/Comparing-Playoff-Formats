#!/usr/bin/env python3
"""Stage 8b (playoff-format) — conditional headline metrics, STANDINGS-RANK variant.

Identical machinery to pf_conditional.py (structural possibility via R1-
outcome enumeration, same replay, same validation) with ONE change: the
"good teams" are classified by REGULAR-SEASON LEAGUE STANDINGS RANK
(1 = best record under the league chain PTS -> ROW -> DIFF, no H2H — the
same chain and replay as pf_rank_matrix.py / the rank-matrix table)
instead of true-talent GF% rank.

So per scenario we report, for the season's #1-#2 / #1-#4 / #1-#8 RECORD
teams:
  p_possible — fraction of seasons where the format structurally allows
               the outcome;
  p_cond     — P(occurred | possible);
  p_uncond   — unconditional probability.

Notes:
  - The target TEAMS differ per season (standings ranks are seasonal), so
    occurrence flags are computed per season on the parquet round/cup
    fields at the standings-rank teams' indices.
  - Validation identical: occurred => possible asserted for every season;
    possibility logic already unit-checked in pf_conditional.py (imported
    from there — no duplication).

Outputs:
  outputs/pf_conditional_standings.csv
  outputs/reports/pf_stage8b_report.txt

Usage: ../likens/code/.venv/bin/python pf_conditional_standings.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

LI = "/workspace/projects/hockey-lab/analyses/likens/code"
PF = "/workspace/projects/hockey-lab/analyses/playoff-format"
sys.path.insert(0, LI)
sys.path.insert(0, f"{PF}/code")

import stage13_batch_engine as eng  # noqa: E402
from pf_conditional import (  # noqa: E402
    N_SEASONS, SEED, SCEN_NAMES, ORDER, RESEEDED_SCEN, CONF_SCOPED,
    R2_MAP, METRICS, possible_reach, r1_pairs_from_order, season_layout,
)

OUT_CSV = f"{PF}/outputs/pf_conditional_standings.csv"
OUT_REPORT = f"{PF}/outputs/reports/pf_stage8b_report.txt"


def load_parquet_arrays(scen: int) -> dict[str, np.ndarray]:
    base = "pf_scenario0" if scen == 0 else f"pf_s{scen}"
    t = pd.read_parquet(f"{PF}/outputs/{base}_teams.parquet")
    assert t["season"].nunique() == N_SEASONS
    piv = lambda c: t.pivot(index="season", columns="team_idx",
                            values=c).to_numpy()
    return {"rr": piv("round_reached"), "po": piv("made_playoffs"),
            "cup": piv("cup_winner")}


def replay_league_orders() -> np.ndarray:
    """[K, 32] league_order (order[k, c] = team at league rank c+1) from
    the same chunked replay as pf_rank_matrix.py (bit-identical seasons)."""
    orders = []
    done = 0
    while done < N_SEASONS:
        CH = min(eng.CHUNK, N_SEASONS - done)
        rng = np.random.default_rng(SEED + done)
        talent, pts, gf, ga, row, diff, h2h, gf_rate, ga_rate = \
            eng._regular_season_chunk(CH, rng)
        _, _, league_order = eng._rank_all(pts, row, diff, h2h)
        orders.append(league_order)
        done += CH
    return np.concatenate(orders, 0)


def compute_scenario(scen: int, parq: dict[str, np.ndarray],
                     lg_order: np.ndarray) -> pd.DataFrame:
    rr, po, cup = parq["rr"], parq["po"], parq["cup"]
    K = rr.shape[0]
    kb = np.arange(K)
    # standings-rank target teams for this season set
    rank1 = lg_order[:, 0]
    rank2 = lg_order[:, 1]
    top4 = [set(lg_order[k, :4].tolist()) for k in range(K)]
    top8 = [set(lg_order[k, :8].tolist()) for k in range(K)]

    occ = {
        "best_cup": cup[kb, rank1] == 1,
        "top2_final": (rr[kb, rank1] >= 4) & (rr[kb, rank2] >= 4),
        "top4_f4": np.array([rr[k, list(top4[k])].min() >= 3
                             for k in range(K)]),
        "top8_r2": np.array([rr[k, list(top8[k])].min() >= 2
                             for k in range(K)]),
    }
    poss = {m: np.zeros(K, dtype=bool) for m in METRICS}
    poss["best_cup"] = po[kb, rank1] == 1

    done = 0
    t0 = time.perf_counter()
    while done < K:
        CH = min(eng.CHUNK, K - done)
        rng = np.random.default_rng(SEED + done)
        talent, pts, gf, ga, row, diff, h2h, gf_rate, ga_rate = \
            eng._regular_season_chunk(CH, rng)
        div_order, conf_order, league_order = eng._rank_all(pts, row, diff,
                                                            h2h)
        cd = {"div_order": div_order, "conf_order": conf_order,
              "league_order": league_order, "pts": pts, "row": row,
              "diff": diff, "h2h": h2h}
        for kk in range(CH):
            k = done + kk
            pairs, seeds_of, conf_of = season_layout(scen, cd, kk)
            t2 = {int(rank1[k]), int(rank2[k])}
            p2 = (po[k, list(t2)] == 1).all()
            if p2 and (scen in CONF_SCOPED or scen in (0, 7)):
                p2 = conf_of[int(rank1[k])] != conf_of[int(rank2[k])]
            if p2:
                poss["top2_final"][k] = possible_reach(
                    pairs, t2, 4, scen in RESEEDED_SCEN,
                    scen in CONF_SCOPED, seeds_of,
                    None if scen in RESEEDED_SCEN else
                    (R2_MAP["conf_div"] if scen == 0 else
                     R2_MAP["conf_fixed"] if scen in (1, 2)
                     else R2_MAP["league_fixed"]))
            t4 = top4[k]
            if (po[k, list(t4)] == 1).all():
                poss["top4_f4"][k] = possible_reach(
                    pairs, t4, 3, scen in RESEEDED_SCEN,
                    scen in CONF_SCOPED, seeds_of,
                    None if scen in RESEEDED_SCEN else
                    (R2_MAP["conf_div"] if scen == 0 else
                     R2_MAP["conf_fixed"] if scen in (1, 2)
                     else R2_MAP["league_fixed"]))
            t8 = top8[k]
            if (po[k, list(t8)] == 1).all():
                okr = not any((a in t8) and (b in t8) for a, b in pairs)
                poss["top8_r2"][k] = okr
        done += CH
    dt = time.perf_counter() - t0
    print(f"  scenario {scen}: replay+enumeration in {dt:.1f}s")

    for m in METRICS:
        bad = int((occ[m] & ~poss[m]).sum())
        assert bad == 0, f"scen {scen} {m}: {bad} occurred but impossible"

    rows = []
    for m in METRICS:
        n_poss = int(poss[m].sum())
        p_cond = (float((occ[m] & poss[m]).sum()) / n_poss
                  if n_poss else float("nan"))
        rows.append({"scenario": scen, "format": SCEN_NAMES[scen],
                     "metric": m, "n_possible": n_poss,
                     "p_possible": round(n_poss / K, 5),
                     "p_cond": round(p_cond, 5),
                     "p_uncond": round(float(occ[m].mean()), 5)})
    return pd.DataFrame(rows)


def main() -> None:
    lg_order = replay_league_orders()
    # cross-check vs the rank-matrix machinery: standings rank 1 team must
    # win the Presidents' Trophy flag rate 0.4159 for the talent-best team;
    # simpler structural check: each season's order is a permutation.
    assert lg_order.shape == (N_SEASONS, 32)
    assert (np.sort(lg_order, axis=1) ==
            np.tile(np.arange(32), (N_SEASONS, 1))).all()
    # the talent-rank-1 team is at league rank 1 in 41.6% of seasons
    t0 = pd.read_parquet(f"{PF}/outputs/pf_scenario0_teams.parquet")
    tal = t0.pivot(index="season", columns="team_idx",
                   values="talent_gfpct").to_numpy()
    tr = (-tal).argsort(1).argsort(1)
    best = (tr == 0).argmax(1)
    rate = float((lg_order[:, 0] == best).mean())
    assert abs(rate - 0.41588) < 5e-5, rate
    print(f"[check] P(talent-best team is standings #1) = {rate:.4f} "
          "(= rank-matrix (1,1) cell) PASS")

    all_rows = []
    for scen in ORDER:
        print(f"scenario {scen} ({SCEN_NAMES[scen]}) ...")
        parq = load_parquet_arrays(scen)
        all_rows.append(compute_scenario(scen, parq, lg_order))
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(df.to_string(index=False))
    print(f"wrote {OUT_CSV}")

    lines = ["Stage 8b — conditional headline metrics, STANDINGS-RANK variant",
             "(teams classified by regular-season league rank, chain "
             "PTS->ROW->DIFF; same replay as pf_rank_matrix.py)", ""]
    for m in METRICS:
        lines.append(f"=== {m} ===")
        for r in df[df["metric"] == m].itertuples():
            lines.append(f"  {r.format:<26} possible {r.p_possible:.3f}   "
                         f"cond {r.p_cond:.3f}   uncond {r.p_uncond:.3f}   "
                         f"(n_poss {r.n_possible:,})")
        lines.append("")
    lines.append("Occurred=>possible asserted for all seasons x scenarios.")
    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
