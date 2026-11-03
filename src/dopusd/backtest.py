"""Validación con origen móvil: en cada origen el modelo ve solo el pasado y se mide contra lo que pasó h días después."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dopusd.models import MODELS, Forecaster

HORIZONS = (1, 7, 30)
MIN_TRAIN = 365
STEP = 7


def load_series(path: Path) -> pd.Series:
    """Serie diaria sin huecos. Un día que falta se interpola en línea recta (hay uno solo en los datos)."""
    s = pd.read_csv(path, parse_dates=["date"]).set_index("date")["usd_dop"]
    return s.asfreq("D").interpolate(method="linear")


@dataclass(frozen=True)
class Fold:
    model: str
    origin: int
    horizon: int
    forecast: float
    actual: float
    last: float  # valor en el origen, para medir si acertó la dirección


def origins(n: int, horizon_max: int, min_train: int = MIN_TRAIN, step: int = STEP) -> range:
    """Índices de origen: hay al menos min_train días de historial y cabe el horizonte más largo."""
    return range(min_train - 1, n - horizon_max, step)


def run(values: np.ndarray, models: dict[str, Forecaster] | None = None, horizons: Sequence[int] = HORIZONS,
        min_train: int = MIN_TRAIN, step: int = STEP) -> pd.DataFrame:
    models = models or MODELS
    rows: list[Fold] = []
    for t in origins(len(values), max(horizons), min_train, step):
        history = values[: t + 1]  # solo hasta el origen, incluido
        for name, model in models.items():
            preds = model(history, horizons)
            for h in horizons:
                rows.append(Fold(name, t, h, preds[h], float(values[t + h]), float(values[t])))
    return pd.DataFrame(rows)


def block_bootstrap_means(diff: np.ndarray, block: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Medias de remuestreos por bloques circulares. Con horizonte h y un origen cada STEP días, los errores de
    orígenes vecinos comparten parte del recorrido, así que remuestrear uno a uno subestimaría la incertidumbre."""
    n = len(diff)
    blocks = -(-n // block)
    starts = rng.integers(0, n, size=(draws, blocks))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
    return diff[idx].mean(axis=1)


def summarize(folds: pd.DataFrame, baseline: str = "naive", seed: int = 7, draws: int = 2000) -> pd.DataFrame:
    """MAE, MAE relativo al baseline con intervalo bootstrap, y acierto de dirección."""
    folds = folds.assign(abs_err=(folds.forecast - folds.actual).abs(),
                         hit=np.sign(folds.forecast - folds["last"]) == np.sign(folds.actual - folds["last"]))
    rng = np.random.default_rng(seed)
    out = []
    for (model, h), g in folds.groupby(["model", "horizon"]):
        base = folds[(folds.model == baseline) & (folds.horizon == h)].set_index("origin").abs_err
        err = g.set_index("origin").abs_err
        diff = (err - base.reindex(err.index)).to_numpy()  # misma lista de orígenes: comparación pareada
        boot = block_bootstrap_means(diff, block=max(1, -(-h // STEP)), draws=draws, rng=rng)
        out.append({
            "model": model, "horizon": h, "n": len(err), "mae": err.mean(),
            "mae_vs_naive": err.mean() / base.mean() - 1 if model != baseline else 0.0,
            "diff_lo": np.percentile(boot, 2.5), "diff_hi": np.percentile(boot, 97.5),
            "direction": g.hit.mean() if model != baseline else float("nan"),  # el naive nunca predice un cambio
        })
    return pd.DataFrame(out)


def main() -> None:
    series = load_series(Path("data/usd_dop.csv"))
    folds = run(series.to_numpy())
    table = summarize(folds)
    pd.set_option("display.width", 160)
    print(f"{series.index[0].date()} a {series.index[-1].date()}, {len(series)} días, {folds.origin.nunique()} orígenes\n")
    print(table.round(4).to_string(index=False))
    Path("results").mkdir(exist_ok=True)
    table.round(4).to_csv("results/summary.csv", index=False)


if __name__ == "__main__":
    main()
