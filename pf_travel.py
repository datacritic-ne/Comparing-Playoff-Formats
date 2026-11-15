#!/usr/bin/env python3
"""Stage 5 (playoff-format) — Playoff travel module.

Computes playoff travel burden per scenario from the Stage 4 series logs.

Model (agreed in planning §4):
- Distance = great-circle miles between the two teams' HOME ARENA CITIES
  (haversine, earth radius 3958.8 mi). One round trip = 2x the distance.
- A best-of-7 series with pattern 2-2-1-1-1 (higher seed hosts games
  1,2,5,7) contains C(k, 2)? No — the AWAY team makes exactly `games`
  one-way trips? Convention (documented): each AWAY GAME is one leg; the
  away team's travel for a series = (number of away games it played) x
  (round-trip distance between the cities), because after each away game
  the team returns home (no multi-city road swings modeled — the standard
  simplification for schedule-level burden; away teams in a series do not
  stay over between games 2 and 5 etc.).
  So per series: away games of the lower seed = pattern positions where
  the HIGHER seed hosts = 4 minus (lower-seed home games). Pattern
  2-2-1-1-1: higher seed hosts 4 potential home slots (g1,2,5,7); if the
  series lasts g games, the higher seed hosts ceil? Exact count depends
  on g; compute per game from the 2-2-1-1-1 pattern and the winner.
- The higher seed also travels to its away games. Both teams' travel is
  summed into the league total (each series contributes the same total
  either way: (away games played by lower seed + away games played by
  higher seed) x distance; the two are complementary and both count).
- League total per scenario = sum over all series of all seasons.

Public API:
  CITY_COORDS: dict team_abbrev -> (lat, lon) — home arena cities
  great_circle_miles(a, b) -> float
  series_travel_miles(hi_team, lo_team, games, hi_won) -> per-game array
  run() -> per-scenario totals DataFrame + CSV outputs

Verification (run this file):
  1. Known city-pair distances hand-checked (e.g. TOR-MTL ~310 mi air,
     NYR-BOS ~185, ANA-SJ ~340) within tolerance vs published figures
  2. Symmetry: d(a,b) == d(b,a); d(a,a) == 0
  3. Worked example: a 7-game and a 4-game series with known winner ->
     exact per-game legs and totals
  4. Per-scenario totals: every logged series accounted for; per-season
     series count {15 series x?} = 8+4+2+1 = 15 per season per scenario
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

PF_OUT = "/workspace/projects/hockey-lab/analyses/playoff-format/outputs"

# Home arena CITY coordinates (lat, lon) — city precision per the planning
# decision (venue-vs-city immaterial at league scale; city centroids used).
# Sources: standard arena city locations (approximate to ~0.01 deg).
CITY_COORDS = {
    "ANA": (33.8075, -117.8762),  # Anaheim (Honda Center)
    "BOS": (42.0961, -71.0708),   # Boston (TD Garden area)
    "BUF": (42.8766, -78.8766),   # Buffalo
    "CGY": (51.0506, -114.0708),  # Calgary
    "CAR": (35.8032, -78.6062),   # Raleigh
    "CHI": (41.8807, -87.6742),   # Chicago
    "COL": (39.7439, -105.0077),  # Denver
    "CBJ": (39.9667, -83.0133),   # Columbus
    "DAL": (32.7906, -96.8106),   # Dallas
    "DET": (42.1329, -83.0547),   # Detroit
    "EDM": (53.5252, -113.4990),  # Edmonton
    "FLA": (26.1584, -80.3252),   # Sunrise FL
    "LAK": (33.9431, -118.3417),  # Los Angeles area (El Segundo/Inglewood)
    "MIN": (44.9793, -93.2736),   # St Paul
    "MTL": (45.4962, -73.5695),   # Montreal
    "NSH": (36.1592, -86.7797),   # Nashville
    "NJD": (40.6597, -74.1694),   # Newark
    "NYI": (40.6855, -73.5254),   # Elmont NY
    "NYR": (40.7505, -73.9934),   # Manhattan
    "OTT": (45.3508, -75.5517),   # Ottawa
    "PHI": (39.9014, -75.1724),   # Philadelphia
    "PIT": (40.4395, -79.9907),   # Pittsburgh
    "SJS": (37.3325, -121.8310),  # San Jose
    "SEA": (47.5952, -122.3321),  # Seattle
    "STL": (38.6348, -90.2844),   # St Louis
    "TBL": (27.7681, -82.3895),   # Tampa
    "TOR": (43.6433, -79.3793),   # Toronto
    "UTA": (40.7685, -111.8890),  # Salt Lake City
    "VAN": (49.2768, -123.1120),  # Vancouver
    "VGK": (36.1183, -115.1518),  # Las Vegas (Paradise)
    "WPG": (49.8919, -97.1384),   # Winnipeg
    "WSH": (38.8979, -77.0213),   # Washington DC
}

R_EARTH_MI = 3958.8


def great_circle_miles(a: str, b: str) -> float:
    """Haversine great-circle distance in statute miles between the home
    cities of two team abbreviations."""
    lat1, lon1 = np.radians(CITY_COORDS[a])
    lat2, lon2 = np.radians(CITY_COORDS[b])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH_MI * np.arcsin(np.sqrt(h))


# 2-2-1-1-1: game g is hosted by the higher seed where True.
HOME_PATTERN_HIGHER = np.array([True, True, False, False, True, False, True])


def series_legs(games: int, hi_won: bool) -> tuple[int, int]:
    """Return (away_games_lower_seed, away_games_higher_seed) for a series
    of `games` games won by the higher seed (hi_won) or lower seed.

    Away games for the higher seed = games hosted by the lower seed among
    the first `games` games = count of False in the pattern prefix.
    """
    prefix = HOME_PATTERN_HIGHER[:games]
    hi_home = int(prefix.sum())
    lo_home = games - hi_home
    # lower seed's away games = games hosted by higher seed (hi_home);
    # higher seed's away games = lo_home.
    return hi_home, lo_home


def team_series_travel(hi_team: str, lo_team: str, games: int,
                       hi_won: bool) -> tuple[float, float]:
    """Travel miles for (higher-seed team, lower-seed team) in one series:
    each away game = one round trip (2 x distance). Winner determinate only
    through the length; the home pattern alone defines away counts."""
    d = great_circle_miles(hi_team, lo_team)
    hi_away, lo_away = series_legs(games, hi_won)
    return hi_away * 2 * d, lo_away * 2 * d


def run() -> pd.DataFrame:
    """Compute league-total playoff travel per scenario (scenarios 1-6
    from their series logs; scenario 0 handled by adding its own log via
    the same engine if present — current format uses the Stage 13 engine
    whose series logs were not stored, so scenario 0 travel is NOT
    computable from the Likens parquet; it IS computable by re-running
    the Stage 3 engine with scenario 0 — done here via the Stage 4-style
    pass over scenario 0 for completeness of the 7-format comparison)."""
    # Index order MUST match the engine's team_idx space (the series logs
    # store integer team indices from eng._TEAMS). Fixed 2026-11-12: this
    # list was previously hardcoded in a different order (CGY before CAR,
    # NJD/NSH and SEA/SJS swapped), silently mispairing arena distances.
    import sys as _sys
    _sys.path.insert(0, "/workspace/projects/hockey-lab/analyses/likens/code")
    import stage13_batch_engine as _eng
    TEAMS = list(_eng._TEAMS)
    assert len(TEAMS) == 32 and set(TEAMS) == set(CITY_COORDS), (
        "engine team list and CITY_COORDS disagree")
    idx = {t: i for i, t in enumerate(TEAMS)}
    TEAMS = list(idx)
    # Precompute distance matrix
    D = np.zeros((32, 32))
    for i, a in enumerate(TEAMS):
        for j, b in enumerate(TEAMS):
            D[i, j] = 0 if i == j else great_circle_miles(a, b)

    rows = []
    for scen in range(0, 8):
        t0 = time.perf_counter()
        fname = (f"pf_scenario0_series.parquet" if scen == 0
                 else f"pf_s{scen}_series.parquet")
        path = os.path.join(PF_OUT, fname)
        if not os.path.exists(path):
            rows.append({"scenario": scen, "total_miles": np.nan,
                         "mean_per_season": np.nan, "n_series": 0,
                         "note": "series log missing"})
            continue
        ser = pd.read_parquet(path)
        hi = ser["hi_team"].to_numpy()
        lo = ser["lo_team"].to_numpy()
        games = ser["games"].to_numpy()
        # per-series total = (hi_home_games + lo_home_games) * 2 * d
        # where home games split by pattern; winner-independent
        d = D[hi, lo]
        pat = HOME_PATTERN_HIGHER
        prefix_hi_home = np.concatenate([[0], np.cumsum(pat)])
        hi_home = prefix_hi_home[games]
        lo_home = games - hi_home
        total = (hi_home + lo_home) * 2 * d  # both teams' away round trips
        K = ser["season"].nunique()
        rows.append({
            "scenario": scen,
            "total_miles": total.sum(),
            "mean_per_season": total.sum() / K,
            "mean_per_series": total.mean(),
            "n_series": len(ser),
            "note": "",
        })
        print(f"scenario {scen}: {len(ser):,} series, "
              f"league total {total.sum()/1e6:.1f}M miles "
              f"({time.perf_counter()-t0:.1f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PF_OUT, "pf_travel_totals.csv"), index=False)
    return df


# --- Verification -------------------------------------------------------------


def _run_checks() -> None:
    # 1. Known distances (air miles, city-center approximations)
    checks = [
        ("TOR", "MTL", 310, 40),    # Toronto-Montreal ~336 air
        ("NYR", "BOS", 185, 30),    # NYC-Boston ~190
        ("ANA", "SJS", 340, 40),    # Anaheim-San Jose ~340
        ("VAN", "SEA", 120, 30),    # Vancouver-Seattle ~125
        ("EDM", "CGY", 160, 30),    # Edmonton-Calgary ~175
        ("NYR", "NJD", 10, 10),     # Manhattan-Newark ~10
        ("FLA", "TBL", 190, 40),    # Sunrise-Tampa ~200
        ("WPG", "MIN", 375, 40),    # Winnipeg-St Paul ~380
    ]
    for a, b, target, tol in checks:
        d = great_circle_miles(a, b)
        assert abs(d - target) <= tol, f"{a}-{b}: {d:.0f} vs {target}±{tol}"
        print(f"[dist] {a}-{b}: {d:.0f} mi (target {target}±{tol}) PASS")

    # 2. Symmetry
    import random
    teams = list(CITY_COORDS)
    for _ in range(200):
        a, b = random.sample(teams, 2)
        assert great_circle_miles(a, b) == great_circle_miles(b, a)
    for a in teams:
        assert great_circle_miles(a, a) == 0.0
    print("[symmetry] d(a,b)=d(b,a), d(a,a)=0: PASS")

    # 3. Worked example: 7-game series (all games played), hi won
    # pattern T T F F T F T: hi hosts 1,2,5,7 (4 home), lo hosts 3,4,6 (3)
    hi_away, lo_away = series_legs(7, True)
    assert hi_away == 4 and lo_away == 3, (hi_away, lo_away)
    # 4-game sweep: games 1-4 only: T T F F -> hi hosts 2, lo hosts 2
    hi_away, lo_away = series_legs(4, True)
    assert hi_away == 2 and lo_away == 2
    # 5-game: T T F F T -> hi 3, lo 2
    hi_away, lo_away = series_legs(5, True)
    assert hi_away == 3 and lo_away == 2
    # 6-game: T T F F T F -> hi 3, lo 3
    hi_away, lo_away = series_legs(6, True)
    assert hi_away == 3 and lo_away == 3
    print("[legs] 2-2-1-1-1 leg counts for 4/5/6/7-game series: PASS")

    # total per series independent of winner: (hi_home + lo_home) = games
    for g in range(4, 8):
        h, l = series_legs(g, True)
        assert h + l == g
    print("[legs] hi_home + lo_home == games: PASS")

    # 4. Series accounting on a real log (scenario 3)
    ser = pd.read_parquet(os.path.join(PF_OUT, "pf_s3_series.parquet"))
    K = ser["season"].nunique()
    counts = ser.groupby(["season", "round"]).size().unstack(fill_value=0)
    assert (counts[1] == 8).all() and (counts[2] == 4).all() \
        and (counts[3] == 2).all() and (counts[4] == 1).all(), \
        "series counts per round wrong"
    assert len(ser) == 15 * K
    print(f"[log] scenario 3: {len(ser):,} series = 15 x {K:,} seasons: PASS")

    print("All Stage 5 travel-module checks PASS")


if __name__ == "__main__":
    _run_checks()
    df = run()
    print(df.to_string(index=False))
