#!/usr/bin/env python3
"""Stage 8 (playoff-format) — conditional (possibility-adjusted) headline metrics.

Motivation: the unconditional headline probabilities conflate two things —
whether the FORMAT structurally permits an outcome, and whether the teams
delivered once it is permitted. For each headline metric we report, per
scenario:
  p_possible — fraction of the 100k seasons in which at least one series-
               outcome realization consistent with the format's pairing
               rules produces the outcome;
  p_cond     — P(outcome occurred | possible);
  p_uncond   — the unconditional probability (cross-checked against
               pf_table_headline.csv).

"Possible" is STRUCTURAL (user decision 2026-11-13): drawn talents/standings
are fixed; we ask whether some realization of series outcomes could produce
the outcome. Implemented by enumerating round-1 winner assignments (targets'
series are forced wins; other series branch) and replaying the exact R2-R4
pairing rules of each format — fixed topologies via the same R1-pair-index
maps pf_postseason.py uses, reseeded pairing via the 1998-99 top-vs-lowest
rule (conference halves for 4/5/7, league-wide for 6). No hand-coded
impossibility conditions: the enumeration enforces every constraint.

Per-metric events (round_reached semantics per planning doc §6: rr k = won
the round-(k-1) series; "in the final" = rr >= 4):
  best_cup   — talent rank 1 wins the Cup. Structurally possible iff the
               team makes the playoffs (any single team can run the table;
               opponents are free) — no enumeration needed.
  top2_final — talent ranks 1-2 both in the Cup final (they are the two
               finalists, hence they meet). Enumerate to round 4.
  top4_f4    — talent ranks 1-4 all reach the final four (rr >= 3).
               Enumerate to round 3.
  top8_r2    — talent ranks 1-8 all win R1 (rr >= 2). Possible iff all 8
               make the playoffs and no two share an R1 pair — direct
               check, no enumeration (winning one series is always
               structurally possible for any team).

Prefilters (necessary conditions, pure speed): playoff membership of the
target set; different conferences for top2_final under conference formats.

Validation: for every season, occurred => possible is asserted (the actual
realization is a witness of possibility) across all 8 scenarios — a strong
end-to-end check of the enumeration.

Outputs:
  outputs/pf_conditional.csv
  outputs/reports/pf_stage8_report.txt

Usage: ../likens/code/.venv/bin/python pf_conditional.py
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
from pf_brackets import build_seed_orders  # noqa: E402

N_SEASONS = 100_000
SEED = 20261028
SCEN_NAMES = {0: "Current format", 7: "Current with reseeding",
              1: "2-6 bracket", 4: "2-6 with reseeding",
              2: "1-8 bracket", 5: "1-8 with reseeding",
              3: "1-16 bracket", 6: "1-16 with reseeding"}
ORDER = [0, 7, 1, 4, 2, 5, 3, 6]
RESEEDED_SCEN = {4, 5, 6, 7}
CONF_SCOPED = {1, 2, 4, 5, 7}          # conference formats (final = E vs W)

OUT_CSV = f"{PF}/outputs/pf_conditional.csv"
OUT_REPORT = f"{PF}/outputs/reports/pf_stage8_report.txt"

METRICS = ["best_cup", "top2_final", "top4_f4", "top8_r2"]
# target_round per metric: the round the targets must all reach
# (2 = R2, 3 = final four, 4 = Cup final). best_cup handled separately.
TARGET_ROUND = {"top2_final": 4, "top4_f4": 3, "top8_r2": 2}


# =====================================================================
# Pairing rules (mirror pf_postseason.py exactly, on R1-pair indices)
# =====================================================================

# R2 pairing as (R1-pair index, R1-pair index), per scenario family.
# Conference fixed brackets (scen 1/2): classic pyramid (0,3),(1,2).
# Scenario 0 (division bracket): the top division's two R1 series winners
#   meet, and the other division's two meet — per conference the R1-pair
#   partition is {(0,2),(1,3)} regardless of which division is 'top'
#   (slots s+0/s+2 always hold the top division's teams; the partition is
#   what matters for reachability, since R3 pairs the conference's two R2
#   winners either way).
# League fixed (scen 3): (0,1),(2,3),(4,5),(6,7).
R2_MAP = {
    "conf_fixed": [(0, 3), (1, 2), (4, 7), (5, 6)],
    "conf_div":   [(0, 2), (1, 3), (4, 6), (5, 7)],
    "league_fixed": [(0, 1), (2, 3), (4, 5), (6, 7)],
}
# R3 pairing as (R2-pair index, R2-pair index): the two conference R2
# winners meet (conf), or bracket halves meet (league) — same index map.
R3_MAP = [(0, 1), (2, 3)]


def reseed_pair(survivors: list[int], seeds_of: dict[int, int],
                conf_scoped: bool, half_bounds: list[tuple[int, int]]
                ) -> list[tuple[int, int]]:
    """1998-99 reseeded pairing: within each half, sort by original seed
    (best = lowest first); top remaining meets lowest remaining, etc.
    half_bounds: index ranges of the halves in the survivors list
    (conf-scoped: [(0, n/2), (n/2, n)]; league: [(0, n)])."""
    pairs = []
    for a, b in half_bounds:
        grp = sorted(survivors[a:b], key=lambda t: seeds_of[t])
        pairs += [(grp[i], grp[len(grp) - 1 - i]) for i in range(len(grp) // 2)]
    return pairs


def possible_reach(r1_pairs: list[tuple[int, int]], targets: set[int],
                   final_round: int, reseeded: bool, conf_scoped: bool,
                   seeds_of: dict[int, int] | None,
                   r2_map: list[tuple[int, int]] | None) -> bool:
    """Can all `targets` survive until `final_round` begins (winning every
    prior series, never paired with each other in a prior round)?

    Enumerates R1 winner assignments: targets' series are forced wins (a
    target-target R1 pair => impossible: one must lose); free series branch
    over both winners (needed because survivors' seeds drive the reseeded
    pairings). Then walks rounds 2..final_round-1 the same way."""
    assert not (reseeded and seeds_of is None)
    # R1: force target wins; fail outright if two targets are paired
    forced: dict[int, int] = {}
    free_idx: list[int] = []
    for i, (a, b) in enumerate(r1_pairs):
        ta, tb = a in targets, b in targets
        if ta and tb:
            return False                     # one of them must lose R1
        if ta:
            forced[i] = a
        elif tb:
            forced[i] = b
        else:
            free_idx.append(i)

    n_free = len(free_idx)
    # halves for reseeded pairing of 8 survivors: conf formats pair within
    # East (R1 pairs 0-3) / West (4-7); league pairs globally.
    halves8 = [(0, 4), (4, 8)] if conf_scoped else [(0, 8)]

    for mask in range(1 << n_free):
        w1 = [None] * 8
        for i, w in forced.items():
            w1[i] = w
        m = mask
        for k, i in enumerate(free_idx):
            a, b = r1_pairs[i]
            w1[i] = a if (m >> k) & 1 else b
        # --- round 2 pairing
        if reseeded:
            r2 = reseed_pair(w1, seeds_of, conf_scoped, halves8)
        else:
            r2 = [(w1[i], w1[j]) for i, j in r2_map]
        # targets must all survive R2 (needed whenever final_round >= 3)
        r2_opts = []
        ok = True
        for a, b in r2:
            ta, tb = a in targets, b in targets
            if ta and tb:
                ok = False
                break
            if ta:
                r2_opts.append((a,))
            elif tb:
                r2_opts.append((b,))
            else:
                r2_opts.append((a, b))
        if not ok:
            continue
        if final_round == 3:
            return True          # targets can win R2 -> all four in R3
        # --- final_round == 4: branch over the non-target R2 series,
        # because the reseeded R3 pairing depends on survivors' seeds.
        halves4 = [(0, 2), (2, 4)] if conf_scoped else [(0, 4)]
        n2 = sum(1 for o in r2_opts if len(o) == 2)
        for mask2 in range(1 << n2):
            w2 = []
            m = mask2
            for o in r2_opts:
                if len(o) == 1:
                    w2.append(o[0])
                else:
                    w2.append(o[m & 1])
                    m >>= 1
            if reseeded:
                r3 = reseed_pair(w2, seeds_of, conf_scoped, halves4)
            else:
                r3 = [(w2[i], w2[j]) for i, j in R3_MAP]
            if not any(a in targets and b in targets for a, b in r3):
                return True      # targets win R3 -> both in the final
    return False


# =====================================================================
# Per-season R1 layouts
# =====================================================================

def r1_pairs_from_order(order: np.ndarray, conf_scoped: bool
                        ) -> list[tuple[int, int]]:
    """R1 pairs for scenarios 1-6 from the [16] seed order (East 0-7,
    West 8-15 for conference formats; league slots for 3/6)."""
    if conf_scoped:
        idx = [(0, 7), (1, 6), (2, 5), (3, 4), (8, 15), (9, 14), (10, 13),
               (11, 12)]
    else:
        idx = [(0, 15), (7, 8), (3, 12), (4, 11), (2, 13), (5, 10),
               (1, 14), (6, 9)]
    return [(int(order[i]), int(order[j])) for i, j in idx]


# =====================================================================
# Main computation
# =====================================================================

def load_parquet_arrays(scen: int) -> dict[str, np.ndarray]:
    """Per-team wide arrays from the scenario parquet (rule 18 source)."""
    base = "pf_scenario0" if scen == 0 else f"pf_s{scen}"
    t = pd.read_parquet(f"{PF}/outputs/{base}_teams.parquet")
    assert t["season"].nunique() == N_SEASONS
    piv = lambda c: t.pivot(index="season", columns="team_idx",
                            values=c).to_numpy()
    return {"rr": piv("round_reached"), "po": piv("made_playoffs"),
            "cup": piv("cup_winner"),
            "tal": piv("talent_gfpct").astype(np.float64)}


def season_layout(scen: int, chunk_data: dict, k: int
                  ) -> tuple[list[tuple[int, int]], dict[int, int] | None,
                             dict[int, int]]:
    """R1 pairs + seed numbers + conf membership for season k of chunk."""
    div_order, conf_order, league_order = (chunk_data["div_order"],
                                           chunk_data["conf_order"],
                                           chunk_data["league_order"])
    conf_of: dict[int, int] = {}
    for c in range(2):
        for t in conf_order[k, c, :]:
            conf_of[int(t)] = c
    if scen in (0, 7):
        hi1, lo1 = eng._seed_brackets(
            div_order[k:k + 1], conf_order[k:k + 1],
            chunk_data["pts"][k:k + 1], chunk_data["row"][k:k + 1],
            chunk_data["diff"][k:k + 1], chunk_data["h2h"][k:k + 1])
        pairs = [(int(hi1[0, s]), int(lo1[0, s])) for s in range(8)]
        seeds_of = None
        if scen == 7:
            seeds_of = {}
            for c in range(2):
                for pos, t in enumerate(conf_order[k, c, :]):
                    seeds_of[int(t)] = pos + 1     # conf-rank 1..16
        return pairs, seeds_of, conf_of
    conf_scoped = scen in CONF_SCOPED
    if scen in (1, 2, 4, 5):
        order, _ = build_seed_orders(div_order[k:k + 1], conf_order[k:k + 1],
                                     league_order[k:k + 1],
                                     chunk_data["pts"][k:k + 1], scen)
        order = order[0]
    else:  # 3/6
        order = league_order[k, :16]
    pairs = r1_pairs_from_order(order, conf_scoped)
    seeds_of = None
    if scen in RESEEDED_SCEN:
        if conf_scoped:
            # seed numbers = position within conference (1-8)
            seeds_of = {int(order[j]): (j % 8) + 1 for j in range(16)}
        else:
            seeds_of = {int(order[j]): j + 1 for j in range(16)}
    return pairs, seeds_of, conf_of


def compute_scenario(scen: int, parq: dict[str, np.ndarray]
                     ) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replay + possibility + occurrence for one scenario."""
    tr = (-parq["tal"]).argsort(1).argsort(1)          # [K,32], 0 = best
    K = tr.shape[0]
    rr, po, cup = parq["rr"], parq["po"], parq["cup"]
    conf_family = scen in CONF_SCOPED or scen in (0, 7)

    # per-season target team indices
    best = (tr == 0).argmax(1)
    second = (tr == 1).argmax(1)
    top4 = [set(np.where(tr[k] < 4)[0].tolist()) for k in range(K)]
    top8 = [set(np.where(tr[k] < 8)[0].tolist()) for k in range(K)]

    # occurrence flags (from the parquet — rule 18)
    occ = {
        "best_cup": cup[np.arange(K), best] == 1,
        "top2_final": (rr[np.arange(K), best] >= 4)
                      & (rr[np.arange(K), second] >= 4),
        "top4_f4": np.array([rr[k, list(top4[k])].min() >= 3
                             for k in range(K)]),
        "top8_r2": np.array([rr[k, list(top8[k])].min() >= 2
                             for k in range(K)]),
    }

    poss = {m: np.zeros(K, dtype=bool) for m in METRICS}
    # best_cup: possible iff in playoffs (no enumeration)
    poss["best_cup"] = po[np.arange(K), best] == 1

    # replay in chunks (same rng streams as the primary run)
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
            t2 = {int(best[k]), int(second[k])}
            # top2_final prefilter: both in playoffs (+ diff conf for
            # conference formats — finalists are the conference champs)
            p2 = (po[k, list(t2)] == 1).all()
            if p2 and conf_family:
                p2 = conf_of[int(best[k])] != conf_of[int(second[k])]
            if p2:
                poss["top2_final"][k] = possible_reach(
                    pairs, t2, 4, scen in RESEEDED_SCEN,
                    scen in CONF_SCOPED, seeds_of,
                    None if scen in RESEEDED_SCEN else
                    (R2_MAP["conf_div"] if scen == 0 else R2_MAP["conf_fixed"]
                     if scen in (1, 2) else R2_MAP["league_fixed"]))
            t4 = top4[k]
            if (po[k, list(t4)] == 1).all():
                poss["top4_f4"][k] = possible_reach(
                    pairs, t4, 3, scen in RESEEDED_SCEN,
                    scen in CONF_SCOPED, seeds_of,
                    None if scen in RESEEDED_SCEN else
                    (R2_MAP["conf_div"] if scen == 0 else R2_MAP["conf_fixed"]
                     if scen in (1, 2) else R2_MAP["league_fixed"]))
            t8 = top8[k]
            # top8_r2: all in playoffs AND no two of them share an R1 pair
            # (a pair with BOTH endpoints in t8 kills one of them; note the
            # check must catch two DIFFERENT targets in one pair, not just
            # a repeated team)
            if (po[k, list(t8)] == 1).all():
                okr = not any((a in t8) and (b in t8) for a, b in pairs)
                poss["top8_r2"][k] = okr
        done += CH
    dt = time.perf_counter() - t0
    print(f"  scenario {scen}: replay+enumeration in {dt:.1f}s")

    # --- validation: occurred => possible, every season
    for m in METRICS:
        bad = int((occ[m] & ~poss[m]).sum())
        assert bad == 0, f"scen {scen} {m}: {bad} seasons occurred but " \
                         "structurally impossible — enumeration bug"
    # unconditional probabilities from the parquet must match
    # pf_table_headline.csv exactly (same data, same formulas)
    headline = pd.read_csv(f"{PF}/outputs/pf_table_headline.csv")
    hrow = headline[headline["scenario"] == scen].iloc[0]
    uncond = {
        "best_cup": float(occ["best_cup"].mean()),
        "top2_final": float(occ["top2_final"].mean()),
        "top4_f4": float(occ["top4_f4"].mean()),
        "top8_r2": float(occ["top8_r2"].mean()),
    }
    for m, col in [("top8_r2", "top8_r2__uncond"),
                   ("top4_f4", "top4_f4__uncond"),
                   ("top2_final", "top2_final__uncond"),
                   ("best_cup", "best_cup__uncond")]:
        assert abs(uncond[m] - float(hrow[col])) < 1e-9, \
            f"scen {scen} {m}: {uncond[m]} vs headline {hrow[col]}"

    rows = []
    for m in METRICS:
        n_poss = int(poss[m].sum())
        p_poss = n_poss / K
        p_cond = (float((occ[m] & poss[m]).sum()) / n_poss
                  if n_poss else float("nan"))
        rows.append({"scenario": scen, "format": SCEN_NAMES[scen],
                     "metric": m, "n_possible": n_poss,
                     "p_possible": round(p_poss, 5),
                     "p_cond": round(p_cond, 5),
                     "p_uncond": round(uncond[m], 5)})
    return pd.DataFrame(rows), {"replay_s": dt}


def _unit_checks() -> None:
    """Hand-verifiable structural cases on constructed layouts."""
    E = list(range(0, 8))       # East teams, seed = index+1
    W = list(range(8, 16))      # West teams, seed = index-7
    # --- scenario 2 (fixed 1-8): order = [E0..E7, W0..W7]
    order = np.array(E + W)
    pairs = r1_pairs_from_order(order, True)
    assert pairs == [(0, 7), (1, 6), (2, 5), (3, 4),
                     (8, 15), (9, 14), (10, 13), (11, 12)]
    seeds = {t: (i % 8) + 1 for i, t in enumerate(order)}
    # seeds 1 and 4 (E0, E3): same R2 pair (0,3) -> can't both reach F4
    assert not possible_reach(pairs, {0, 3}, 3, False, True, seeds,
                              R2_MAP["conf_fixed"])
    # seeds 1,2,9,10 (E0,E1,W0,W1): 2 per conf, different halves -> possible
    assert possible_reach(pairs, {0, 1, 8, 9}, 3, False, True, seeds,
                          R2_MAP["conf_fixed"])
    # seeds 1 and 2 both in final: same conference -> impossible
    assert not possible_reach(pairs, {0, 1}, 4, False, True, seeds,
                              R2_MAP["conf_fixed"])
    # seeds 1 (E) and 9 (W): possible
    assert possible_reach(pairs, {0, 8}, 4, False, True, seeds,
                          R2_MAP["conf_fixed"])
    # seeds 1 and 8: R1 opponents -> impossible even for top2
    assert not possible_reach(pairs, {0, 7}, 4, False, True, seeds,
                              R2_MAP["conf_fixed"])
    # --- scenario 5 (reseeded 1-8): same R1, reseeded R2/R3
    # seeds 1,4 (E) + 9,10 (W): East survivors {1,4,+2 free} can avoid a
    # 1-4 collision (e.g. survivors {1,4,5,6} -> (1,6),(4,5)) -> possible
    assert possible_reach(pairs, {0, 3, 8, 9}, 3, True, True, seeds, None)
    # seeds 1,2,3,4 all East: East final four = 4 teams -> (1,4) or (2,3)
    # collision unavoidable -> impossible
    assert not possible_reach(pairs, {0, 1, 2, 3}, 3, True, True, seeds, None)
    # --- scenario 3 (fixed 1-16): R1 pairs (0,15),(7,8),(3,12),(4,11),
    # (2,13),(5,10),(1,14),(6,9); seed s -> slot s-1.
    lorder = np.arange(16)
    lpairs = r1_pairs_from_order(lorder, False)
    lseeds = {t: t + 1 for t in range(16)}
    # seeds 1 (pair 0) and 2 (pair 6): opposite halves -> final possible
    assert possible_reach(lpairs, {0, 1}, 4, False, False, lseeds,
                          R2_MAP["league_fixed"])
    # seeds 2 (pair 6 -> R2 pair 3) and 3 (pair 4 -> R2 pair 2): same R3
    # branch (2,3) -> meet in the SEMI -> final impossible
    assert not possible_reach(lpairs, {1, 2}, 4, False, False, lseeds,
                              R2_MAP["league_fixed"])
    # --- scenario 6 (reseeded 1-16): seeds 2,3 CAN avoid each other via
    # upsets (survivor-set dependent) -> possible
    assert possible_reach(lpairs, {1, 2}, 4, True, False, lseeds, None)
    # --- scenario 0 (division bracket): R2 partition {(0,2),(1,3)} per
    # conference.
    epairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13),
              (14, 15)]
    zseeds = {t: t for t in range(16)}
    # targets in R1 pairs 0 and 2 (same conference, same R2 pair) ->
    # impossible for F4
    assert not possible_reach(epairs, {0, 4}, 3, False, True, zseeds,
                              R2_MAP["conf_div"])
    # targets in pairs 0 and 1 (different R2 pairs, same conf) -> possible
    assert possible_reach(epairs, {0, 2}, 3, False, True, zseeds,
                          R2_MAP["conf_div"])
    # scenario 7 (reseeded, conf-rank seeds): E1=team0 (cr 1) in pair 0,
    # E8=team7 (cr 8) in pair 3, W1=team8 (cr 9) in West pair 5.
    seeds7 = {t: t + 1 for t in range(16)}
    p7 = [(0, 10), (1, 2), (3, 4), (7, 11),        # East pairs 0-3
          (5, 9), (8, 6), (12, 14), (13, 15)]      # West pairs 4-7
    # E1 + W1 in the final: each survives its conference's reseeded R2/R3
    # (E1's R2 partner is never a target; R3 pairs it with the OTHER East
    # R2 winner) -> possible
    assert possible_reach(p7, {0, 8}, 4, True, True, seeds7, None)
    # E1 + E8 in the final: East R2 survivors are exactly the four East R1
    # winners with cranks {1, 2|3, 4|5, 8} — sorted [1, <5, <5, 8] makes
    # top-vs-lowest pair (1, 8) at R2 EVERY time -> one of them must lose
    # -> impossible (the 1998-99 rule FORCES 1 v 8 together at R2)
    assert not possible_reach(p7, {0, 7}, 4, True, True, seeds7, None)
    # top4_f4 with E1(0), E5(4), W1(8), W5(12): West R2 survivors' cranks
    # are {6|10, 9, 13, 14|16} — the favorite branch (cr 6 winner) sorts
    # [6,9,13,14|16] -> (9,13) collision = targets 8 v 12 paired at R2;
    # but the UPSET branch (team 9, cr 10, wins pair 4) sorts
    # [9,10,13,14|16] -> (9,14|16),(10,13) -> separated. Survivor-set
    # dependence: only SOME R1 outcomes permit the outcome -> possible.
    assert possible_reach(p7, {0, 4, 8, 12}, 3, True, True, seeds7, None)
    # top8_r2 inline check regression: conference seeds 1 and 8 share an
    # R1 pair -> two targets in one pair -> impossible
    t8case = {0, 7}
    assert any((a in t8case) and (b in t8case) for a, b in pairs)
    # (pairs from scenario-2 layout above contain (0,7))
    print("[unit] all structural possibility checks PASS")


def main() -> None:
    _unit_checks()
    all_rows = []
    timings = {}
    for scen in ORDER:
        print(f"scenario {scen} ({SCEN_NAMES[scen]}) ...")
        parq = load_parquet_arrays(scen)
        rows, timing = compute_scenario(scen, parq)
        all_rows.append(rows)
        timings[scen] = timing
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(df.to_string(index=False))
    print(f"wrote {OUT_CSV}")

    # ---- report
    lines = ["Stage 8 — conditional (possibility-adjusted) headline metrics",
             f"({N_SEASONS:,} seasons per scenario; structural possibility "
             "via R1-outcome enumeration)", ""]
    for m in METRICS:
        lines.append(f"=== {m} ===")
        sub = df[df["metric"] == m]
        for r in sub.itertuples():
            lines.append(f"  {r.format:<26} possible {r.p_possible:.3f}   "
                         f"cond {r.p_cond:.3f}   uncond {r.p_uncond:.3f}   "
                         f"(n_poss {r.n_possible:,})")
        lines.append("")
    lines.append("Definitions: possible = some series-outcome realization "
                 "consistent with the\nformat produces the outcome; cond = "
                 "P(occurred | possible); uncond = the\nheadline probability "
                 "(= pf_table_headline.csv). best_cup possible = best team "
                 "makes\nthe playoffs (single team can always run the table). "
                 "Occurred=>possible asserted\nfor all seasons x scenarios.")
    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
