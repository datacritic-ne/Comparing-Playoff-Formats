#!/usr/bin/env python3
"""Playoff-format — headline-probability table (PNG for Substack).

Stage 8 version: for each of the 4 headline metrics we now report THREE
columns per format — P(possible) [the format structurally allows the
outcome in that fraction of seasons], P(occurred | possible), and the
unconditional P [kept for continuity]. Sources (rule 18):
  - unconditional: per-team parquets (same formulas as before; asserted
    equal to the Stage 1/4/7 report values via EXPECT)
  - possible / conditional: outputs/pf_conditional.csv (Stage 8, produced
    by pf_conditional.py — never recomputed here)

Rows: 8 formats in the tornado-figure display order.

Artifacts:
  outputs/pf_table_headline.csv   (now with the 3 columns per metric)
  outputs/pf_table_headline.png   (12 data columns; grouped header)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PF_OUT = "/workspace/projects/hockey-lab/analyses/playoff-format/outputs"

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

# Expected UNCONDITIONAL values cross-check (Stage 4 report; s1/s2 corrected
# 2026-11-12 after the crossed-R2 topology fix; s0/s7 from Stage 1/7).
EXPECT = {
    0: (0.0008, 0.0076, 0.0702, 0.2753),
    7: (0.0008, 0.0141, 0.0727, 0.2802),
    1: (0.0017, 0.0120, 0.0723, 0.2784),
    4: (0.0017, 0.0142, 0.0749, 0.2818),
    2: (0.0022, 0.0120, 0.0730, 0.2782),
    5: (0.0022, 0.0148, 0.0753, 0.2817),
    3: (0.0028, 0.0169, 0.0915, 0.2829),
    6: (0.0028, 0.0222, 0.1089, 0.2899),
}

METRIC_KEYS = ["top8_r2", "top4_f4", "top2_final", "best_cup"]
UNCOND_COL = {"top8_r2": "p_top8_r2", "top4_f4": "p_top4_cf",
              "top2_final": "p_1v2_final", "best_cup": "p_best_cup"}
METRIC_LABEL = {
    "top8_r2": "Top 8 teams reach round 2",
    "top4_f4": "Top 4 reach conf. finals*",   # *semis in 1-16 formats
    "top2_final": "Top 2 teams meet in the final",
    "best_cup": "Best team wins the Cup",
}


def headline_for(sid: int) -> tuple[float, float, float, float]:
    """Unconditional probabilities from the per-team parquets (unchanged
    from the pre-Stage-8 version of this script)."""
    base = "pf_scenario0" if sid == 0 else f"pf_s{sid}"
    t = pd.read_parquet(f"{PF_OUT}/{base}_teams.parquet")
    n_seasons = t["season"].nunique()
    assert n_seasons == 100_000, n_seasons

    def all_of(ranks: list[int], rr: int) -> float:
        sub = t[t["talent_rank"].isin(ranks)]
        g = sub.groupby("season")["round_reached"].min()
        return float((g >= rr).mean())

    p_top8_r2 = all_of(list(range(1, 9)), 2)
    p_top4_cf = all_of([1, 2, 3, 4], 3)
    p_12_final = all_of([1, 2], 4)
    p_best_cup = float(t[t["talent_rank"] == 1]["cup_winner"].mean())
    return p_top8_r2, p_top4_cf, p_12_final, p_best_cup


def main() -> None:
    cond = pd.read_csv(f"{PF_OUT}/pf_conditional.csv")
    assert len(cond) == 32, len(cond)   # 8 scenarios x 4 metrics

    rows = []
    for sid, label in SCENARIOS:
        vals = headline_for(sid)
        exp = EXPECT[sid]
        # cross-check the unconditional values against Stage 1/4/7 reports
        for v, e in zip(vals, exp):
            assert abs(v - e) < 5e-4, f"s{sid}: {vals} vs expected {exp}"
        # pull the Stage 8 possibility/conditional values
        sub = cond[cond["scenario"] == sid].set_index("metric")
        assert len(sub) == 4
        # consistency: p_uncond in pf_conditional.csv must equal ours
        row = {"scenario": sid, "format": label}
        for mkey, v in zip(METRIC_KEYS, vals):
            row[f"{mkey}__poss"] = float(sub.loc[mkey, "p_possible"])
            row[f"{mkey}__cond"] = float(sub.loc[mkey, "p_cond"])
            row[f"{mkey}__uncond"] = v
            assert abs(float(sub.loc[mkey, "p_uncond"]) - v) < 1e-9, \
                f"s{sid} {mkey}: uncond mismatch vs pf_conditional.csv"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{PF_OUT}/pf_table_headline.csv", index=False)
    print(df.to_string(index=False))
    print(f"wrote {PF_OUT}/pf_table_headline.csv")

    # ---- PNG table (12 data columns, grouped header) ----
    # Column layout per metric: possible | occurred|possible | uncond
    cell_text = []
    for r in df.itertuples():
        row_cells = []
        for mkey in METRIC_KEYS:
            row_cells += [
                f"{getattr(r, mkey + '__poss'):.3f}",
                f"{getattr(r, mkey + '__cond'):.3f}",
                f"{getattr(r, mkey + '__uncond'):.3f}",
            ]
        cell_text.append(row_cells)

    ncols_data = 12
    col_headers = ["P(Possible)", "P(Outcome|Possible)", "P(Outcome)"] * 4
    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    ax.axis("off")
    tab = ax.table(cellText=cell_text, rowLabels=list(df["format"]),
                   colLabels=col_headers, cellLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.5)

    # group header bands above the column headers
    # (matplotlib tables can't span cells natively; draw text + lines over)
    # custom column widths: P(Outcome|Possible) columns wider so their
    # header fits; the flanking columns narrower (total unchanged)
    col_w = [0.92, 1.16, 0.92] * 4
    col_w = [w / sum(col_w) * 12 / 13 for w in col_w]   # + row label col
    for (row, col), cell in tab.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if col is not None and col >= 0:
            cell.set_width(col_w[col])
        if row == 0:  # column-header row
            cell.set_facecolor("#e8ecf4")
            cell.set_text_props(color="#333333", fontsize=6.3)
            cell.set_height(cell.get_height() * 1.35)
        elif col == -1:  # row labels
            cell.set_text_props(fontweight="bold", ha="right")
            cell.set_facecolor("#f2f2f2")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f7f7")
    # stronger vertical separators between metric groups + bold the
    # 'cond' column (the new headline quantity)
    for (row, col), cell in tab.get_celld().items():
        if col is None or col < 0:
            continue
        if col % 3 == 2 and col < ncols_data - 1:
            cell.set_edgecolor("#999999")   # group boundary on the right
        if col % 3 == 1 and row > 0:
            cell.set_text_props(fontweight="bold")
    fig.suptitle("Format Scenarios and Playoff Outcomes by Round "
                 "(100,000 Simulated Seasons per Format)",
                 fontsize=11.5, y=0.99)
    fig.text(0.5, 0.03,
             "*conf. finals = semifinals in the 1-16 formats (no conferences).",
             ha="center", fontsize=8, color="gray")
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.92])

    # metric group titles + underlines above the header row. Positioned
    # from the header cells' RENDERED pixel extents (after tight_layout,
    # in figure coordinates) so alignment over each 3-column group is
    # exact regardless of the table's placement within the axes.
    fig.canvas.draw()
    inv_fig = fig.transFigure.inverted()
    for gi, mkey in enumerate(METRIC_KEYS):
        cols = [3 * gi, 3 * gi + 1, 3 * gi + 2]
        exts = [tab[(0, c)].get_window_extent(fig.canvas.get_renderer())
                for c in cols]
        x0 = min(e.x0 for e in exts)
        x1 = max(e.x1 for e in exts)
        y_top = max(e.y1 for e in exts)
        # underline just above the header row, spanning the group
        uf0 = inv_fig.transform((x0, y_top + 2.0))
        uf1 = inv_fig.transform((x1, y_top + 2.0))
        fig.add_artist(Line2D([uf0[0], uf1[0]], [uf0[1], uf1[1]],
                              transform=fig.transFigure,
                              color="#001f5b", lw=1.1))
        # title centered above the underline
        tc = inv_fig.transform(((x0 + x1) / 2, y_top + 6.0))
        fig.text(tc[0], tc[1], METRIC_LABEL[mkey], ha="center", va="bottom",
                 fontsize=8.6, fontweight="bold")
    fig.savefig(f"{PF_OUT}/pf_table_headline.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {PF_OUT}/pf_table_headline.png")


if __name__ == "__main__":
    main()
