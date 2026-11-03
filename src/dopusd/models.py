"""Pronosticadores. Todos reciben solo el historial hasta el origen y devuelven un valor por horizonte."""

import warnings
from collections.abc import Callable, Sequence

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

Forecaster = Callable[[np.ndarray, Sequence[int]], dict[int, float]]

WINDOW = 365  # los modelos ajustados miran el último año, no todo el historial


def naive(history: np.ndarray, horizons: Sequence[int]) -> dict[int, float]:
    """El último valor observado. Es la vara de medir: un modelo que no lo supera no sirve."""
    return {h: float(history[-1]) for h in horizons}


def median5(history: np.ndarray, horizons: Sequence[int]) -> dict[int, float]:
    """Mediana de los últimos 5 días. Aguanta un día suelto con un valor raro en el origen."""
    return {h: float(np.median(history[-5:])) for h in horizons}


def drift(history: np.ndarray, horizons: Sequence[int]) -> dict[int, float]:
    """El último valor más la pendiente media del último año por cada día de horizonte."""
    recent = history[-WINDOW:]
    slope = (recent[-1] - recent[0]) / (len(recent) - 1)
    return {h: float(history[-1] + slope * h) for h in horizons}


def ets(history: np.ndarray, horizons: Sequence[int]) -> dict[int, float]:
    """Suavizado exponencial con tendencia amortiguada."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(history[-WINDOW:], trend="add", damped_trend=True, initialization_method="estimated").fit()
    path = fit.forecast(max(horizons))
    return {h: float(path[h - 1]) for h in horizons}


def arima(history: np.ndarray, horizons: Sequence[int]) -> dict[int, float]:
    """ARIMA(1,1,1) sobre el último año."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(history[-WINDOW:], order=(1, 1, 1)).fit()
    path = fit.forecast(max(horizons))
    return {h: float(path[h - 1]) for h in horizons}


MODELS: dict[str, Forecaster] = {
    "naive": naive,
    "median5": median5,
    "drift": drift,
    "ets": ets,
    "arima": arima,
}
