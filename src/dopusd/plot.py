"""Dibuja results/summary.csv: cuánto más error tiene cada modelo que el naive, con su intervalo."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    t = pd.read_csv("results/summary.csv")
    t = t[t.model != "naive"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    for ax, (h, g) in zip(axes, t.groupby("horizon")):
        base_mae = g.mae.iloc[0] / (1 + g.mae_vs_naive.iloc[0])
        rel = g.mae_vs_naive * 100
        lo = (g.diff_lo / base_mae) * 100
        hi = (g.diff_hi / base_mae) * 100
        colors = ["#b3261e" if l > 0 else "#6b6b6b" for l in lo]
        ax.barh(g.model, rel, color=colors, xerr=[rel - lo, hi - rel], capsize=3, ecolor="#333")
        ax.axvline(0, color="#000", lw=1)
        ax.set_title(f"Horizonte {h} día{'s' if h > 1 else ''}", fontsize=11)
        ax.set_xlabel("Error vs naive (%)")
        ax.invert_yaxis()
    fig.suptitle("Error medio de cada modelo respecto a repetir el último valor (rojo: peor con 95 % de confianza)", fontsize=10)
    fig.tight_layout()
    Path("results").mkdir(exist_ok=True)
    fig.savefig("results/error_vs_naive.png", dpi=130)


if __name__ == "__main__":
    main()
