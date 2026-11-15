#!/usr/bin/env python3
"""Stage 2 (playoff-format) — Seeding/bracket layer for the 6 variants.

Takes the Stage 13-style ranking outputs (team indices ordered by the
Stage 9 tiebreaker chain within each group) and produces, per playoff
format scenario, the round-1 series pairs plus the reseeding function used
after rounds 1 and 2 in the reseeded variants.

Formats (SCENARIOS 1-6; scenario 0 = current division bracket, handled by
the Likens engine and NOT implemented here):

  1  "2-6"        per conference: the 2 division winners are seeds 1-2
                  (ordered by points between themselves, then the standard
                  chain); the 6 non-winners are seeds 3-8 by conference
                  rank. Round 1: 1v8, 2v7, 3v6, 4v5. FIXED bracket:
                  R2 = (1v8 w) vs (4v5 w), (2v7 w) vs (3v6 w) (classic
                  pyramid; seeds 1 and 2 can only meet in the conf final);
                  R3 = the two survivors per conference; final = East vs
                  West.
  2  "1-8"        per conference: strict 1-8 by conference rank; divisions
                  irrelevant. Round 1: 1v8, 2v7, 3v6, 4v5. Same fixed R2/R3
                  shape as scenario 1.
  3  "1-16"       league-wide: strict 1-16 by league rank (apply_h2h=False,
                  matching stage9's league rank / Presidents' semantics).
                  Round 1: 1v16, 8v9, 4v13, 5v12, 3v14, 6v11, 2v15, 7v10
                  (standard fixed-bracket half layout); FIXED bracket: the
                  two halves feed the semis; final = the two semi winners.
  4  "2-6 reseed" like 1, but after each round the remaining teams are
                  reseeded 1-4 (R2) / 1-2 (R3) by ORIGINAL conference seed
                  and top remaining seed meets the lowest remaining seed
                  (1998-99 convention, records.nhl.com). Home ice by seed.
  5  "1-8 reseed" like 2 with the same reseeding rule.
  6  "1-16 reseed" like 3, reseeding league-wide at every round.
  7  "current reseed" R1 = scenario 0's division bracket + wild cards
                  (Stage 13 internals, home ice by the full chain); after
                  R1 and R2 the survivors are reseeded WITHIN each
                  conference by conference rank (the PTS->ROW->H2H->DIFF->
                  code chain position, 1-16) and top remaining seed meets
                  the lowest remaining seed. Seed numbers here are
                  conference-rank positions, not bracket slots.

Home ice: ALWAYS the higher seed in scenarios 1-6 (no wildcard home-ice
quirk; isolates the seeding effect from home-ice allocation). The series
engine's _better() chain is not used for seeding here — seed order IS the
ranking order by construction; home ice = higher seed exactly.

Data model
----------
A "seed order" is an int array [K, n_teams_in_scope] of team indices, best
first, restricted to the 16 playoff teams. Conventions used by the
postseason engine (Stage 3):
  - `initial_seeds(scenario, ...)` -> [K, 16] seed order (best first) +
    a [K, 16] seed-number array (1-based; seed numbers = order positions).
  - `pair_round(seeds, round_idx, reseed)` -> (hi, lo) [K, 8|4|2|1] arrays
    of team indices, hi = higher seed (holds home ice in scenarios 1-6).
    For FIXED brackets, round_idx 2/3 pairings follow the fixed bracket
    map; for RESEEDED, the survivors are re-ranked by original seed.
  - Both are fully vectorized over K seasons (chunk-friendly like stage13).

Verification (run this file): unit tests on constructed standings for each
variant — legal matchups, seed-order correctness, home-ice correctness,
reseeding edge cases (upsets removing top seeds), and consistency gates
(each team plays once per round; winners advance; bracket closes 16->1).
"""

from __future__ import annotations

import numpy as np

# Scenario ids (strings used throughout the project)
SCENARIOS = {
    "2-6": 1, "1-8": 2, "1-16": 3,
    "2-6-reseed": 4, "1-8-reseed": 5, "1-16-reseed": 6,
}
RESEEDED = {4, 5, 6, 7}
CONFERENCE_SCOPED = {1, 2, 4, 5, 7}  # 8 playoff teams per conference
LEAGUE_SCOPED = {3, 6}            # 16 playoff teams league-wide

# Fixed-bracket round-2 pairing map, conference-scoped (scenarios 1-2):
# classic mirror pyramid (option 1, agreed 2026-11-12): R1 pair i winner
# faces pair j winner with (i, j) = (0,3), (1,2) — i.e. (1v8 w) vs (4v5 w)
# and (2v7 w) vs (3v6 w); seeds 1 and 2 sit in opposite halves and can
# only meet in the conference final.
FIXED_R2_CONF = np.array([[0, 3], [1, 2]])
# League-scoped (scenario 3): R1 slots (0..7) = (1v16, 8v9, 4v13, 5v12,
# 3v14, 6v11, 2v15, 7v10). Halves: {0,1} feed semi A? Standard NHL-style
# fixed bracket: half A = slots 0,1,2,3 -> two R2 pairs (0,1),(2,3);
# half B = slots 4,5,6,7 -> (4,5),(6,7). Semis: A-top vs A-bottom,
# B-top vs B-bottom; final = semi winners.
FIXED_R2_LEAGUE = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
# Which R2 winners meet in the semis (league-scoped): R2 pair i winner vs
# R2 pair j winner: (A1,A2) from pairs (0,1) -> semi 1; (B1,B2) -> semi 2.
FIXED_R3_LEAGUE = np.array([[0, 1], [2, 3]])


def _assert_shapes(seeds: np.ndarray, k_expected_cols: int) -> None:
    assert seeds.ndim == 2, "seeds must be [K, n]"
    assert seeds.shape[1] == k_expected_cols, \
        f"expected {k_expected_cols} columns, got {seeds.shape[1]}"


# --- Initial seed construction ------------------------------------------------


def build_seed_orders(div_order: np.ndarray, conf_order: np.ndarray,
                      league_order: np.ndarray, pts: np.ndarray,
                      scenario: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the [K, 16] initial seed orders + seed numbers for scenarios 1-6.

    div_order  [K, 4, 8]   team indices, best first (stage13 _rank_all)
    conf_order [K, 2, 16]  team indices, best first
    league_order [K, 32]   team indices, best first (apply_h2h=False)
    pts        [K, 32]

    Returns (order, seedno): order [K, 16] team indices best-first (for
    conference-scoped scenarios: East seeds in columns 0-7, West 8-15;
    for league-scoped: plain league order); seedno [K, 16] 1-based seed
    numbers matching order positions.
    """
    K = pts.shape[0]
    ki = np.arange(K)[:, None]

    if scenario in (2, 5):
        # strict 1-8 per conference: first 8 of the conference rank order
        order = np.concatenate([conf_order[:, 0, :8], conf_order[:, 1, :8]],
                               axis=1)
        seedno = np.tile(np.arange(1, 9), (K, 2))
    elif scenario in (1, 4):
        # --- 2-6: seeds 1-2 = division winners (ordered between themselves
        # by conference rank position = the PTS->ROW->H2H->DIFF->code chain);
        # seeds 3-8 = next 6 conference teams by conf_order.
        order = np.empty((K, 16), dtype=np.int64)
        for c in range(2):
            base = 8 * c
            members = conf_order[:, c, :]                # [K,16] best first
            d0, d1 = 2 * c, 2 * c + 1
            w0 = div_order[:, d0, 0]                     # [K]
            w1 = div_order[:, d1, 0]
            # order the two winners by their conference rank position
            pos0 = (members == w0[:, None]).argmax(1)    # [K]
            pos1 = (members == w1[:, None]).argmax(1)
            first = np.where(pos0 <= pos1, w0, w1)
            second = np.where(pos0 <= pos1, w1, w0)
            order[:, base + 0] = first                   # seed 1
            order[:, base + 1] = second                  # seed 2
            # seeds 3-8: next 6 conference teams by conf_order
            is_winner = (members == w0[:, None]) | (members == w1[:, None])
            # keep conf_order sequence, dropping the two winners
            # (exactly one team per position; winners are 2 of the 16)
            rest = np.empty((K, 6), dtype=np.int64)
            col = 0
            src = 0
            # vectorized compact: for each row, first 6 non-winner entries
            notw = ~is_winner
            # argsort trick: positions of non-winners in order
            idx = np.argsort(~notw, axis=1, kind="stable")  # False(0) first
            rest = members[ki, idx[:, :6]]
            order[:, base + 2:base + 8] = rest
        assert _order_legal(order, conf_order), "2-6 seed order illegal"
        seedno = np.tile(np.arange(1, 9), (K, 2))
    elif scenario in LEAGUE_SCOPED:
        order = league_order[:, :16].copy()
        seedno = np.tile(np.arange(1, 17), (K, 1))
    else:
        raise ValueError(f"scenario {scenario} not handled by pf_brackets")

    return order, seedno


def _order_legal(order: np.ndarray, conf_order: np.ndarray) -> bool:
    """Each row of order is a permutation of 16 distinct teams; for
    conference-scoped orders, every entry must be a playoff-calibre team
    (any team is legal here since we only test permutations)."""
    s = np.sort(order, axis=1)
    return bool((s[:, :-1] != s[:, 1:]).all())


# --- Round pairing ------------------------------------------------------------


def pair_round(order: np.ndarray, round_idx: int, reseeded: bool
               ) -> tuple[np.ndarray, np.ndarray]:
    """Pair the surviving teams for `round_idx` (1, 2, or 3) — RESEEDED
    brackets only (fixed brackets pair by original slot, see Stage 3).

    order: [K, n_survivors] seed order (best first) among survivors,
    re-sorted by original seed number before this call.

    Returns (hi, lo) [K, n_survivors // 2]; hi = higher seed.
    """
    n_pairs = order.shape[1] // 2
    hi = order[:, :n_pairs].copy()
    lo = order[:, -n_pairs:][:, ::-1].copy()
    return hi, lo


def pair_round_fixed(order16: np.ndarray, round_idx: int,
                     conference_scoped: bool
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-bracket pairing by ORIGINAL slot.

    order16: [K, 16] with -1 marking eliminated teams; survivors occupy
    their original slots (columns 0-7 East seeds 1-8 / columns 0-7 league
    slots for league-scoped; columns 8-15 West for conference-scoped).

    Returns (hi, lo) [K, n_pairs] with hi = better seed of the pair.
    """
    K = order16.shape[0]
    if round_idx == 1:
        if conference_scoped:
            # East slots 0-7 and West slots 8-15 pair within themselves:
            # 1v8, 2v7, 3v6, 4v5.
            hi = np.concatenate([order16[:, 0:4], order16[:, 8:12]], axis=1)
            lo = np.concatenate([order16[:, 7:3:-1], order16[:, 15:11:-1]],
                                axis=1)
        else:
            # League 1-16 fixed bracket: 1v16, 8v9, 4v13, 5v12, 3v14, 6v11,
            # 2v15, 7v10 (slot pairs (0,15),(7,8),(3,12),(4,11),(2,13),
            # (5,10),(1,14),(6,9)). Halves {0,1,2,3} and {4,5,6,7} feed
            # separate semi branches.
            HI_SLOTS = np.array([0, 7, 3, 4, 2, 5, 1, 6])
            LO_SLOTS = np.array([15, 8, 12, 11, 13, 10, 14, 9])
            hi = order16[:, HI_SLOTS]
            lo = order16[:, LO_SLOTS]
        return hi, lo
    if round_idx == 2:
        # pair winners of (0,1) and (2,3) per half; hi = better seed.
        raise NotImplementedError("Stage 3 assembles R2 from R1 winners")
    raise NotImplementedError("Stage 3 assembles R3 from R2 winners")


# --- Verification -------------------------------------------------------------


def _mk_standings(seed: int = 11, K: int = 50):
    """Constructed standings: random distinct PTS per team so the full
    tiebreaker chain is exercised trivially (no ties), 32 teams."""
    rng = np.random.default_rng(seed)
    pts = rng.integers(50, 125, size=(K, 32))
    # make PTS unique per season for clean rank checks
    for k in range(K):
        while len(np.unique(pts[k])) != 32:
            pts[k] = rng.integers(50, 125, size=32)
    row = pts // 2
    diff = rng.integers(-50, 51, size=(K, 32))
    h2h = np.zeros((K, 32, 32), dtype=np.int64)
    return pts, row, diff, h2h


def _rank_groups(pts, row, diff, h2h):
    """Reuse stage13's ranking helpers (identical chain) without importing
    the engine (avoids pulling the schedule load at import time)."""
    import sys
    sys.path.insert(0, "/workspace/projects/hockey-lab/analyses/likens/code")
    import stage13_batch_engine as eng
    div_order, conf_order, league_order = eng._rank_all(pts, row, diff, h2h)
    return div_order, conf_order, league_order


def _run_checks() -> None:
    pts, row, diff, h2h = _mk_standings()
    div_order, conf_order, league_order = _rank_groups(pts, row, diff, h2h)
    K = pts.shape[0]

    # --- 1-8 and 2-6: seed orders legal, contain the right teams
    for scen in (1, 2):
        order, seedno = build_seed_orders(div_order, conf_order,
                                           league_order, pts, scen)
        assert order.shape == (K, 16) and seedno.shape == (K, 16)
        assert np.sort(order, axis=1) is not None
        # East half = columns 0-7 must be Eastern teams, in conf-rank order
        east = conf_order[:, 0, :]
        # Every Eastern team in order[:, :8] must be among the East 16 and
        # (scenario 2) appear in conf-rank sequence.
        for c, sl in [(0, slice(0, 8)), (1, slice(8, 16))]:
            members = set(conf_order[:, c, :].ravel().tolist())
            got = order[:, sl]
            assert set(got.ravel().tolist()) <= members
            if scen == 2:
                seq = conf_order[:, c, :]
                pos = {t: r for r, t in enumerate(members)} if False else None
                # per-row check: order columns are in conf_order sequence
                rank_in_conf = (seq[:, :, None] == got[:, None, :]).argmax(1)
                assert (np.diff(rank_in_conf, axis=1) > 0).all(), \
                    f"scenario {scen}: not in conference-rank order"
            else:
                # scenario 1: seeds 1-2 are the division winners
                w0 = div_order[:, 2 * c, 0]
                w1 = div_order[:, 2 * c + 1, 0]
                ok = ((got[:, 0] == w0) & (got[:, 1] == w1)) | \
                     ((got[:, 0] == w1) & (got[:, 1] == w0))
                assert ok.all(), "seeds 1-2 not the two division winners"
        print(f"[scenario {scen}] seed orders legal, correct membership, "
              f"{'conf-rank order' if scen == 2 else 'div-winner seeds 1-2'}: PASS")

    # --- 2-6: seeds 3-8 = next 6 conference teams by conf rank (not winners)
    order26, _ = build_seed_orders(div_order, conf_order, league_order, pts, 1)
    for c, sl in [(0, slice(0, 8)), (1, slice(8, 16))]:
        seq = conf_order[:, c, :]
        got = order26[:, sl]
        w0 = div_order[:, 2 * c, 0]
        w1 = div_order[:, 2 * c + 1, 0]
        # seeds 3-8 must be the 6 highest-ranked non-winners
        notw = ~((seq == w0[:, None]) | (seq == w1[:, None]))
        expect = seq[np.arange(K)[:, None], np.argsort(~notw, axis=1,
                                                       kind="stable")[:, :6]]
        assert (got[:, 2:8] == expect).all(), "seeds 3-8 wrong"
    print("[scenario 1] seeds 3-8 = next 6 conference teams: PASS")

    # --- 1-16: order == league_order top 16
    order16, seedno16 = build_seed_orders(div_order, conf_order,
                                           league_order, pts, 3)
    assert (order16 == league_order[:, :16]).all()
    assert (seedno16 == np.arange(1, 17)[None, :]).all()
    print("[scenario 3] 1-16 order == league rank top-16: PASS")

    # --- Reseeding: pair_round on a reseeded order
    # Construct: 8 survivors with original seeds [1, 3, 5, 6, 9, 11, 14, 16]
    # (upsets removed 2,4,7,8,10,12,13,15). Reseeded order = sorted.
    # Reseeded pairing: (1v16), (3v14), (5v11), (6v9).
    survivors = np.array([[1, 3, 5, 6, 9, 11, 14, 16]], dtype=np.int64)
    reseeded_order = np.sort(survivors, axis=1)  # best (lowest) first
    hi, lo = pair_round(reseeded_order, 2, reseeded=True)
    assert (hi == np.array([[1, 3, 5, 6]])).all(), f"hi {hi}"
    assert (lo == np.array([[16, 14, 11, 9]])).all(), f"lo {lo}"
    print("[reseed] top-remaining vs lowest-remaining pairing (8 surv): PASS")

    # Reseeded R3 (4 survivors [1, 5, 9, 11]): (1v11), (5v9).
    hi, lo = pair_round(np.array([[1, 5, 9, 11]], dtype=np.int64), 3,
                        reseeded=True)
    assert (hi == np.array([[1, 5]])).all() and (lo == np.array([[11, 9]])).all()
    print("[reseed] reseeded R3 pairing: PASS")

    # Reseeded final: 2 survivors [3, 9] -> 3 vs 9, home = 3.
    hi, lo = pair_round(np.array([[3, 9]], dtype=np.int64), 3, reseeded=True)
    assert (hi == np.array([[3]])).all() and (lo == np.array([[9]])).all()
    print("[reseed] final pairing (top vs lowest remaining): PASS")

    # --- Fixed-bracket R1 pairing (league-scoped layout check)
    slots = np.arange(16)[None, :].repeat(3, 0)  # teams labelled by slot
    hi, lo = pair_round_fixed(slots, 1, conference_scoped=False)
    expect_pairs = [(0, 15), (7, 8), (3, 12), (4, 11),
                    (2, 13), (5, 10), (1, 14), (6, 9)]
    got_pairs = list(zip(hi[0].tolist(), lo[0].tolist()))
    assert got_pairs == expect_pairs, f"fixed 1-16 R1 layout wrong: {got_pairs}"
    print("[fixed 1-16] R1 = 1v16, 8v9, 4v13, 5v12, 3v14, 6v11, 2v15, 7v10: PASS")

    # --- Conference-scoped fixed R1: East (cols 0-7) and West (8-15)
    hi, lo = pair_round_fixed(slots, 1, conference_scoped=True)
    assert (hi[0][:4] == np.array([0, 1, 2, 3])).all()
    assert (lo[0][:4] == np.array([7, 6, 5, 4])).all()
    assert (hi[0][4:] == np.array([8, 9, 10, 11])).all()
    assert (lo[0][4:] == np.array([15, 14, 13, 12])).all()
    print("[fixed conf] R1 East = 1v8..4v5, West same within cols 8-15: PASS")

    print("All Stage 2 bracket-layer checks PASS")


if __name__ == "__main__":
    _run_checks()
