#!/usr/bin/env python3
"""Playoff-format — headline-probability table, STANDINGS-RANK variant (PNG).

Same 12-column layout as pf_table_headline.py but with teams classified by
REGULAR-SEASON LEAGUE RANK (the Presidents'/league chain PTS -> ROW ->
DIFF, no H2H) instead of true-talent rank. Sources (rule 18):
  outputs/pf_conditional_standings.csv  (Stage 8b, by pf_conditional_standings.py)

Per the user's instruction (2026-11-13): for "best team wins the Cup" the
standings-#1 team IS the Presidents' Trophy winner and always makes the
playoffs, so P(possible) = 1 identically and is OMITTED — the metric is
reported as a single column P(Cup | Presidents' Trophy).

Artifacts:
  outputs/pf_table_headline_standings.csv
  outputs/pf_table_headline_standings.png
"""

from __future__ import annotations

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

METRIC_KEYS = ["top8_r2", "top4_f4", "top2_final", "best_cup"]
METRIC_LABEL = {
    "top8_r2": "Top 8 records\nreach round 2",
    "top4_f4": "Top 4 records reach\nconf. finals*",
    "top2_final": "Top 2 records\nmeet in the final",
    "best_cup": "Presidents' Trophy winner\nwins the Cup",
}


def main() -> None:
    cond = pd.read_csv(f"{PF_OUT}/pf_conditional_standings.csv")
    assert len(cond) == 32, len(cond)

    rows = []
    for sid, label in SCENARIOS:
        sub = cond[cond["scenario"] == sid].set_index("metric")
        assert len(sub) == 4
        # best_cup: possible must be identically 1 (standings #1 = the
        # Presidents' Trophy winner, always a playoff team)
        assert sub.loc["best_cup", "p_possible"] == 1.0
        assert sub.loc["best_cup", "p_cond"] == sub.loc["best_cup", "p_uncond"]
        row = {"scenario": sid, "format": label}
        for mkey in METRIC_KEYS:
            row[f"{mkey}__poss"] = float(sub.loc[mkey, "p_possible"])
            row[f"{mkey}__cond"] = float(sub.loc[mkey, "p_cond"])
            row[f"{mkey}__uncond"] = float(sub.loc[mkey, "p_uncond"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{PF_OUT}/pf_table_headline_standings.csv", index=False)
    print(df.to_string(index=False))
    print(f"wrote {PF_OUT}/pf_table_headline_standings.csv")

    # ---- PNG: 11 data columns (best_cup has no 'poss.' column) ----
    cell_text = []
    for r in df.itertuples():
        row_cells = []
        for mkey in METRIC_KEYS:
            row_cells += [f"{getattr(r, mkey + '__poss'):.3f}",
                          f"{getattr(r, mkey + '__cond'):.3f}",
                          f"{getattr(r, mkey + '__uncond'):.3f}"]
        cell_text.append(row_cells)

    col_headers = ["P(Possible)", "P(Outcome|Possible)", "P(Outcome)"] * 4
    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    ax.axis("off")
    tab = ax.table(cellText=cell_text, rowLabels=list(df["format"]),
                   colLabels=col_headers, cellLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.5)
    # custom column widths: P(Outcome|Possible) columns wider so their
    # header fits; the flanking columns narrower (total unchanged)
    col_w = [0.92, 1.16, 0.92] * 4
    col_w = [w / sum(col_w) * 12 / 13 for w in col_w]   # + row label col
    for (row, col), cell in tab.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if col is not None and col >= 0:
            cell.set_width(col_w[col])
        if row == 0:
            cell.set_facecolor("#e8ecf4")
            cell.set_text_props(color="#333333", fontsize=6.3)
            cell.set_height(cell.get_height() * 1.35)
        elif col == -1:
            cell.set_text_props(fontweight="bold", ha="right")
            cell.set_facecolor("#f2f2f2")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f7f7")
    for (row, col), cell in tab.get_celld().items():
        if col is None or col < 0:
            continue
        if col % 3 == 2 and col < 10:
            cell.set_edgecolor("#999999")
        if col % 3 == 1 and row > 0:
            cell.set_text_props(fontweight="bold")
    fig.suptitle("Format Scenarios and Playoff Outcomes by Round, Ranking "
                 "by Regular-Season Performance "
                 "(100,000 Simulated Seasons per Format)",
                 fontsize=11.5, y=0.99)
    fig.text(0.5, 0.03,
             "Teams classified by regular-season league rank (points + "
             "tiebreakers), not true talent. *conf. finals = semifinals "
             "in the 1-16 formats.",
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
        fig.text(tc[0], tc[1], METRIC_LABEL[mkey].replace("\n", " "),
                 ha="center", va="bottom", fontsize=8.6, fontweight="bold")
    fig.savefig(f"{PF_OUT}/pf_table_headline_standings.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {PF_OUT}/pf_table_headline_standings.png")


if __name__ == "__main__":
    main()
