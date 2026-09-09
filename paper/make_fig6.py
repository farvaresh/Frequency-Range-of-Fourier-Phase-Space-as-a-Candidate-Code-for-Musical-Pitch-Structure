#!/usr/bin/env python3
"""Figure 6 -- recruited Fourier phase-space frequency.
(a) Iranian, all tasks, with the Nyquist frequency k=12 flagged as confounded with the
    chromatic/microtonal binary partition.
(b) Western, all tasks.
(c) Task-matched comparison using the two interval tasks only.
Vector PDF output."""
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white", context="paper")
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
    "font.size": 9, "axes.edgecolor": "#3d3d3d", "axes.linewidth": 0.8,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

full = pd.read_csv("/home/claude/fig5_data.csv")              # all tasks
matched = pd.read_csv("/home/claude/fig_interval_matched.csv")  # interval tasks only
def ser(df, system):
    s = df[df.system == system].sort_values("freq")
    return s.freq.values, s["count"].values.astype(int), int(s.n_sig.iloc[0])

fx, ci, ni = ser(full, "Iranian-17")
_,  cw, nw = ser(full, "Western-12")
_, mi, nmi = ser(matched, "Iranian-17")
_, mw_, nmw = ser(matched, "Western-12")

C_IR, C_IR_D = "#3d6b7d", "#294a57"
C_WE, C_WE_D = "#cc7b5c", "#a2543a"
C_BAND, C_FLAG = "#b9556e", "#8a8a8a"
INK, SUB = "#2b2b2b", "#6f6f6f"

fig, axes = plt.subplots(3, 1, figsize=(6.9, 7.8),
                         gridspec_kw=dict(hspace=0.46, height_ratios=[1, 0.85, 1]))
axA, axB, axC = axes

def dress(ax, ymax, band=True):
    sns.despine(ax=ax, left=True)
    ax.set_xlim(0.3, 12.7); ax.set_ylim(0, ymax)
    ax.set_xticks(range(1, 13))
    ax.set_yticks(np.round(np.linspace(0, ymax, 4) / 10) * 10)
    ax.tick_params(length=0, colors=INK, labelsize=8.2, pad=3)
    ax.grid(axis="y", color="#ececec", linewidth=0.7); ax.set_axisbelow(True)
    if band:
        ax.axvspan(6.5, 12.7, color=C_BAND, alpha=0.05, lw=0, zorder=0)
        ax.axvline(6.5, color=C_BAND, lw=1.0, ls=(0, (5, 2)), alpha=0.45, zorder=1)

# ---- (a) Iranian, all tasks -------------------------------------------------
yA = max(ci) * 1.32; dress(axA, yA)
bars = axA.bar(fx, ci, width=0.72, color=C_IR, edgecolor=C_IR_D, lw=0.6, zorder=3)
bars[11].set_facecolor("white"); bars[11].set_edgecolor(C_FLAG)
bars[11].set_hatch("////"); bars[11].set_linewidth(0.9)
axA.annotate("k = 12 is identical to a chromatic /\nmicrotonal binary partition — excluded\nfrom the confound-free claim",
             xy=(12, ci[11]*0.55), xytext=(9.15, yA*0.70), fontsize=7.6, color=C_FLAG,
             ha="center", va="center", linespacing=1.4,
             arrowprops=dict(arrowstyle="->", color=C_FLAG, lw=0.8,
                             connectionstyle="arc3,rad=0.2"))
axA.text(0.015, 0.95, "Iranian–17 · all nine tasks", transform=axA.transAxes,
         fontsize=10, fontweight="bold", color=C_IR_D, va="top")
axA.text(0.015, 0.845, f"$n$ = {ni:,} FDR-significant units", transform=axA.transAxes,
         fontsize=8, color=SUB, va="top")

# ---- (b) Western, all tasks -------------------------------------------------
yB = max(cw) * 1.45; dress(axB, yB)
axB.bar(fx, cw, width=0.72, color=C_WE, edgecolor=C_WE_D, lw=0.6, zorder=3)
axB.text(0.015, 0.95, "Western–12 · interval tasks", transform=axB.transAxes,
         fontsize=10, fontweight="bold", color=C_WE_D, va="top")
axB.text(0.015, 0.83, f"$n$ = {nw:,} FDR-significant units", transform=axB.transAxes,
         fontsize=8, color=SUB, va="top")
axB.text(0.74, 0.44, "frequencies 7–12 cannot exist\non a 12-tone lattice",
         transform=axB.transAxes, fontsize=8, style="italic", color=C_BAND,
         ha="center", va="center", linespacing=1.4)

# ---- (c) matched, interval tasks only --------------------------------------
yC = max(max(mi), max(mw_)) * 1.62; dress(axC, yC)
w = 0.38
barsC = axC.bar(fx - w/2, mi, width=w, color=C_IR, edgecolor=C_IR_D, lw=0.5, zorder=3,
        label=f"Iranian–17 interval tasks ($n$ = {nmi})")
barsC[11].set_facecolor("white"); barsC[11].set_edgecolor(C_FLAG)
barsC[11].set_hatch("////"); barsC[11].set_linewidth(0.8)
axC.bar(fx + w/2, mw_, width=w, color=C_WE, edgecolor=C_WE_D, lw=0.5, zorder=3,
        label=f"Western–12 interval tasks ($n$ = {nmw})")
axC.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.0),
           ncol=1, handlelength=1.1, handletextpad=0.5, labelspacing=0.45,
           borderaxespad=0.2)
axC.text(0.015, 0.97, "Task-matched comparison", transform=axC.transAxes,
         fontsize=10, fontweight="bold", color=INK, va="top")
axC.text(0.015, 0.86, "interval tasks only \u2014 identical rule in both systems",
         transform=axC.transAxes, fontsize=8, color=SUB, va="top")
n711 = int(mi[6:11].sum())
axC.text(0.70, 0.44, f"{n711}/{nmi} Iranian units ({n711/nmi:.0%})\nat frequencies 7–11;\n0 Western units",
         transform=axC.transAxes, fontsize=8, color=C_BAND, ha="center", va="center",
         fontweight="bold", linespacing=1.45)

for ax, tag in zip(axes, "abc"):
    ax.text(-0.085, 1.13, tag, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left", color=INK)

axC.set_xlabel("Best-fitting Fourier phase-space frequency", fontsize=9.8, color=INK,
               labelpad=6)
fig.text(-0.005, 0.5, "FDR-significant hidden units", va="center", rotation="vertical",
         fontsize=9.8, color=INK)

fig.savefig("/home/claude/paper/figures/fig6_frequency_range.pdf")
fig.savefig("/home/claude/fig6_preview.png", dpi=200)
print("done")
