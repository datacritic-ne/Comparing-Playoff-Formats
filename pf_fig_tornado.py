#!/usr/bin/env python3
"""Playoff-format — tornado (population-pyramid) figure: mean series-side GF% by round.

For each format and playoff round, computes the mean talent GF% of the
STRONGER side (higher drawn talent_gfpct) and the WEAKER side of every
series, from the per-series logs joined to the per-team parquets.
Drawn as a tornado plot: 4 panels (R1, R2, conference finals/semis, Cup
final), one shared y-axis, formats ordered top-to-bottom per the user's
spec; bars anchored at a central axis, stronger side extending LEFT with
length = its mean GF%, weaker side extending RIGHT with length = its mean
GF% (no baseline subtraction — bar lengths read directly as GF%).

Artifacts (rule 18):
  outputs/pf_tornado_values.csv   (scenario x round x side means)
  outputs/pf_fig_tornado.png      (the 4-panel figure)

Inputs (existing): outputs/pf_scenario0_{teams,series}.parquet and
outputs/pf_s{1..7}_{teams,series}.parquet.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PF_OUT = "/workspace/projects/hockey-lab/analyses/playoff-format/outputs"

# Display order top->bottom, per user spec 2026-11-12.
SCENARIOS = [
    (0,  "Current format"),
    (7,  "Current with reseeding"),
    (1,  "2-6 bracket"),
    (4,  "2-6 with reseeding"),
    (2,  "1-8 bracket"),
    (5,  "1-8 with reseeding"),
    (3,  "1-16 bracket"),
    (6,  "1-16 with reseeding"),
]

ROUNDS = [1, 2, 3, 4]
ROUND_TITLES = [
    "Round 1",
    "Round 2",
    "Conference finals*",
    "Cup final",
]


def load_scenario(sid: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = "pf_scenario0" if sid == 0 else f"pf_s{sid}"
    teams = pd.read_parquet(f"{PF_OUT}/{base}_teams.parquet")
    series = pd.read_parquet(f"{PF_OUT}/{base}_series.parquet")
    return teams, series


def side_means(sid: int) -> pd.DataFrame:
    """Mean talent GF% of the stronger (higher talent_gfpct) and weaker
    side of each series, per round."""
    teams, series = load_scenario(sid)
    # talent GF% per (season, team_idx)
    talent = teams.set_index(["season", "team_idx"])["talent_gfpct"]
    hi_gf = series.set_index(["season", "hi_team"]).index.map(talent)
    lo_gf = series.set_index(["season", "lo_team"]).index.map(talent)
    hi_gf = np.asarray(hi_gf, dtype=float)
    lo_gf = np.asarray(lo_gf, dtype=float)
    strong = np.where(hi_gf >= lo_gf, hi_gf, lo_gf)
    weak = np.where(hi_gf >= lo_gf, lo_gf, hi_gf)
    df = pd.DataFrame({
        "round": series["round"].to_numpy(),
        "strong_gf": strong,
        "weak_gf": weak,
    })
    out = df.groupby("round").agg(
        strong_mean=("strong_gf", "mean"),
        weak_mean=("weak_gf", "mean"),
        n_series=("strong_gf", "size"),
    ).reset_index()
    out["scenario"] = sid
    return out


def main() -> None:
    rows = []
    for sid, label in SCENARIOS:
        m = side_means(sid)
        m.insert(0, "label", label)
        rows.append(m)
        # checks: n_series per round must be 800k/400k/200k/100k
        expected = {1: 800_000, 2: 400_000, 3: 200_000, 4: 100_000}
        got = dict(zip(m["round"], m["n_series"]))
        for r, exp in expected.items():
            assert got[r] == exp, f"s{sid} round {r}: {got[r]} != {exp}"
        # structural sanity: strong >= weak in every scenario/round
        assert (m["strong_mean"] > m["weak_mean"]).all(), f"s{sid} side ordering"
    vals = pd.concat(rows, ignore_index=True)
    vals["gap"] = vals["strong_mean"] - vals["weak_mean"]

    # Cross-check vs existing matchup-quality gap means (pf_matchup_quality.csv)
    mq = pd.read_csv(f"{PF_OUT}/pf_matchup_quality.csv")
    mq_cols = {c.lower(): c for c in mq.columns}
    print("matchup_quality.csv columns:", list(mq.columns))
    print(vals.pivot_table(index="label", columns="round", values="gap").round(4))
    print(mq.head(10))

    vals.to_csv(f"{PF_OUT}/pf_tornado_values.csv", index=False)
    print(f"wrote {PF_OUT}/pf_tornado_values.csv")

    # ---- figure ----
    fig, axes = plt.subplots(
        1, 4, figsize=(14, 5.2), sharey=True,
        gridspec_kw={"wspace": 0.06},
    )
    y = np.arange(len(SCENARIOS))[::-1]  # first format at top
    for ax, r, title in zip(axes, ROUNDS, ROUND_TITLES):
        sub = vals[vals["round"] == r].set_index("scenario").loc[
            [s for s, _ in SCENARIOS]]
        # stronger side extends LEFT (length = mean GF%); weaker RIGHT
        ax.barh(y, -sub["strong_mean"], height=0.62, color="#001f5b",
                edgecolor="white", linewidth=0.4)
        ax.barh(y, sub["weak_mean"], height=0.62, color="#9a9a9a",
                edgecolor="white", linewidth=0.4)
        # value labels INSIDE bars (right side of left bars, left side of right)
        for yi, s_v, w_v in zip(y, sub["strong_mean"], sub["weak_mean"]):
            ax.text(-s_v + 0.012, yi, f"{s_v*100:.1f}", ha="left",
                    va="center", fontsize=7.5, color="white", fontweight="bold")
            ax.text(w_v - 0.012, yi, f"{w_v*100:.1f}", ha="right",
                    va="center", fontsize=7.5, color="black", fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-0.65, 0.65)
        ax.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
        ax.set_xticklabels(["60", "40", "20", "0", "20", "40", "60"])
        ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Mean series-side talent GF% (%)", fontsize=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for _, lab in SCENARIOS], fontsize=10)
    fig.suptitle(
        "Mean talent GF% of stronger (left) vs weaker (right) side, "
        "by playoff round", fontsize=12, y=0.995)
    fig.text(0.5, 0.005,
             "*Semifinals in the 1-16 formats (no conferences). "
             "Stronger side = higher drawn talent GF%. Bars anchored at 0; "
             "lengths are mean GF% values.",
             ha="center", fontsize=8, color="gray")
    fig.tight_layout(rect=[0.01, 0.02, 1, 0.97])
    fig.savefig(f"{PF_OUT}/pf_fig_tornado.png", dpi=200)
    plt.close(fig)
    print(f"wrote {PF_OUT}/pf_fig_tornado.png")


if __name__ == "__main__":
    main()
