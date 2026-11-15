#!/usr/bin/env python3
"""Figure: talent-rank x standings-rank joint distribution matrix (32x32).

Reads outputs/pf_rank_matrix.csv (must exist — produced by pf_rank_matrix.py;
rule 18: figures never re-derive numbers). Writes
outputs/pf_fig_rank_matrix.png.

Two-panel layout (2026-11-14, user request): the 1,024 numeric cell values
are unreadable at mobile width, so the numeric table (left) is paired with
a filled-contour density of the same probabilities (right) — white-to-navy
color bands only, no lines or labels inside the plot; colorbar in percent.
Bilinear upsampling smooths band edges between cell centers (display only).
Axes are shared in appearance (labels on the left panel; the contour panel
repeats only the tick numbers).

Usage: ../likens/code/.venv/bin/python pf_fig_rank_matrix.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PF_OUT = "/workspace/projects/hockey-lab/analyses/playoff-format/outputs"
SRC = os.path.join(PF_OUT, "pf_rank_matrix.csv")
OUT = os.path.join(PF_OUT, "pf_fig_rank_matrix.png")


def draw_gridlines(ax: plt.Axes) -> None:
    """Thin cell gridlines + stronger separators every 8 ranks."""
    for k in range(33):
        ax.axhline(k - 0.5, color="#bbbbbb", lw=0.4, zorder=1)
        ax.axvline(k - 0.5, color="#bbbbbb", lw=0.4, zorder=1)
    for k in (7.5, 15.5, 23.5):
        ax.axhline(k, color="#888888", lw=0.9, zorder=1.5)
        ax.axvline(k, color="#888888", lw=0.9, zorder=1.5)


def main() -> None:
    df = pd.read_csv(SRC, index_col=0)
    prob = df.iloc[:32, :32].to_numpy()          # [talent_rank, standings_rank]
    assert np.allclose(prob.sum(1), 1.0, atol=1e-6), "row marginals"
    assert np.allclose(prob.sum(0), 1.0, atol=1e-6), "col marginals"
    pct = prob * 100.0

    # white -> navy sequential colormap (matches the Substack navy accents)
    cmap = LinearSegmentedColormap.from_list(
        "white_navy", ["#ffffff", "#dce4f2", "#7f9bc9", "#2c4f8f", "#0d1f4d"])

    fig, (axn, axh) = plt.subplots(
        1, 2, figsize=(16.0, 9.2), dpi=200,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.14})

    # ---- left panel: numeric table (unchanged style) ----
    axn.set_facecolor("white")
    for k in range(32):   # diagonal highlight
        axn.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                    facecolor="#e8e8e8", edgecolor="none",
                                    zorder=0.5))
    draw_gridlines(axn)
    for r in range(32):
        for c in range(32):
            axn.text(c, r, f"{pct[r, c]:.2f}", ha="center", va="center",
                     fontsize=4.6, color="black", zorder=2)
    axn.set_xlim(-0.5, 31.5)
    axn.set_ylim(31.5, -0.5)                     # talent rank 1 at top
    axn.set_xticks(range(32))
    axn.set_xticklabels(range(1, 33), fontsize=7)
    axn.set_yticks(range(32))
    axn.set_yticklabels(range(1, 33), fontsize=7)
    axn.tick_params(length=0)
    for spine in axn.spines.values():
        spine.set_color("#888888")
    axn.set_xlabel("Regular-season finish, league rank", fontsize=10)
    axn.set_ylabel("True-talent rank (league-wide GF%)", fontsize=10)
    axn.set_title("Probabilities (percent of seasons)", fontsize=10.5,
                  fontweight="bold")

    # ---- right panel: filled-contour density of the same matrix ----
    # strictly visual: color bands only — no gridlines, separators,
    # contour-line strokes, or labels. Bilinear 4x upsampling (numpy
    # only; no scipy in the venv) smooths the band edges between the
    # 32x32 cell centers; it adds no information and changes no values.
    R = 4
    idx_lo = np.repeat(np.arange(31), R)                     # 124 samples
    w_lo = np.tile(np.linspace(1, 0, R, endpoint=False), 31)

    def upsample_axis(a, axis):
        a = np.asarray(a, dtype=float)
        lo = np.take(a, idx_lo, axis=axis)
        hi = np.take(a, idx_lo + 1, axis=axis)
        shp = [1] * a.ndim
        shp[axis] = len(w_lo)
        wl = w_lo.reshape(shp)
        out = lo * wl + hi * (1 - wl)
        # append the last original row/col
        tail = np.take(a, [31], axis=axis)
        return np.concatenate([out, tail], axis=axis)

    up = upsample_axis(upsample_axis(pct, 0), 1)
    assert up.shape[0] == up.shape[1] == 125

    cmap_c = plt.get_cmap(cmap)
    n_bands = 9
    levels = np.linspace(0.0, float(pct.max()), n_bands + 1)
    cf = axh.contourf(up, levels=levels, cmap=cmap_c)
    # ticks at the original 32 cell centers in upsampled coordinates
    centers = [k * R for k in range(32)]
    axh.set_xticks(centers)
    axh.set_xticklabels(range(1, 33), fontsize=7)
    axh.set_yticks(centers)
    axh.set_yticklabels([])                      # shared meaning, labels left
    axh.tick_params(length=0)
    for spine in axh.spines.values():
        spine.set_color("#888888")
    axh.invert_yaxis()                           # talent rank 1 at top
    axh.set_xlabel("Regular-season finish, league rank", fontsize=10)
    axh.set_title("Probability density (darker = more likely)",
                  fontsize=10.5, fontweight="bold")

    cbar = fig.colorbar(cf, ax=axh, fraction=0.035, pad=0.02)
    cbar.set_label("% of simulated seasons", fontsize=9)
    cbar.ax.tick_params(labelsize=7.5)

    fig.suptitle("Team True Talent Versus Regular Season Standings "
                 "Performance (100,000 Simulated Seasons)",
                 fontsize=12)
    fig.text(0.02, 0.012,
             "Cell = % of simulated seasons; rows and columns each sum to "
             "100%. Chain: PTS → ROW → DIFF. Right panel: filled-contour "
             "density of the same matrix (bilinear-smoothed for display; "
             "no data altered). Model: Likens locked constants (talent SD "
             "0.0415, GPG 5.4933, HIA 0.2563), 2025-26 schedule.",
             fontsize=6.6, color="#444444")

    fig.tight_layout(rect=(0, 0.025, 1, 0.92))
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
