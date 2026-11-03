import numpy as np
import pandas as pd
import pytest

from dopusd.backtest import block_bootstrap_means, load_series, origins, run, summarize
from dopusd.models import drift, median5, naive


def series(n=500, slope=0.0, base=60.0):
    return base + slope * np.arange(n)


def test_el_modelo_solo_ve_hasta_el_origen():
    values = series(500) + np.random.default_rng(1).normal(0, 0.1, 500)
    seen = []

    def spy(history, horizons):
        seen.append(history.copy())
        return {h: float(history[-1]) for h in horizons}

    folds = run(values, {"spy": spy}, horizons=(1, 7), min_train=365, step=7)
    for origin, history in zip(sorted(folds.origin.unique()), seen):
        assert len(history) == origin + 1
        np.testing.assert_array_equal(history, values[: origin + 1])


def test_cambiar_el_futuro_no_cambia_los_pronosticos():
    a = series(500) + np.random.default_rng(2).normal(0, 0.1, 500)
    b = a.copy()
    cut = 364 + 7 * 4  # un origen de verdad: 364, 371, ..., 392 (el quinto)
    assert cut in origins(len(a), 1)
    b[cut + 1 :] += 100  # todo lo posterior al origen cambia
    fa = run(a, {"drift": drift, "median5": median5}, horizons=(1,), min_train=365, step=7)
    fb = run(b, {"drift": drift, "median5": median5}, horizons=(1,), min_train=365, step=7)
    early = fa.origin <= cut
    np.testing.assert_allclose(fa[early].forecast.to_numpy(), fb[early].forecast.to_numpy())


def test_el_objetivo_es_el_valor_h_dias_despues_del_origen():
    values = np.arange(500, dtype=float)
    folds = run(values, {"naive": naive}, horizons=(1, 7, 30), min_train=365, step=7)
    assert ((folds.actual - folds["last"]) == folds.horizon).all()


def test_ultimo_origen_deja_espacio_para_el_horizonte_mas_largo():
    r = origins(500, 30)
    assert r[0] == 364
    assert r[-1] + 30 <= 499


def test_serie_constante_ningun_modelo_falla_ni_se_aleja_del_valor():
    from dopusd.models import MODELS

    values = series(500)
    folds = run(values, MODELS, horizons=(1, 7), min_train=365, step=7)
    assert folds.forecast.notna().all()
    assert (folds.forecast - 60.0).abs().max() < 1e-3


def test_con_tendencia_lineal_drift_acierta_y_naive_se_atrasa_h_pasos():
    values = series(500, slope=0.01)
    folds = run(values, {"naive": naive, "drift": drift}, horizons=(1, 30), min_train=365, step=7)
    folds["err"] = (folds.forecast - folds.actual).abs()
    assert folds[folds.model == "drift"].err.max() < 1e-9
    n30 = folds[(folds.model == "naive") & (folds.horizon == 30)].err
    np.testing.assert_allclose(n30, 0.3)


def test_dos_modelos_identicos_dan_intervalo_que_incluye_cero():
    values = series(600) + np.random.default_rng(3).normal(0, 0.2, 600)
    folds = run(values, {"naive": naive, "copy": naive}, horizons=(1,), min_train=365, step=7)
    row = summarize(folds).query("model == 'copy'").iloc[0]
    assert row.diff_lo <= 0 <= row.diff_hi
    assert row.mae_vs_naive == 0


def test_un_modelo_peor_sale_con_intervalo_positivo():
    values = series(600) + np.random.default_rng(4).normal(0, 0.2, 600)

    def bad(history, horizons):
        return {h: float(history[-1]) + 5 for h in horizons}

    folds = run(values, {"naive": naive, "bad": bad}, horizons=(1,), min_train=365, step=7)
    row = summarize(folds).query("model == 'bad'").iloc[0]
    assert row.diff_lo > 0 and row.mae_vs_naive > 0


def test_un_bloque_del_tamano_de_la_serie_es_una_rotacion_y_siempre_da_la_media():
    # Con bloques circulares contiguos del largo completo, cada remuestreo es una rotación de la serie y su media no cambia.
    # Un remuestreo uno a uno, o uno que no cierre el círculo, daría medias distintas.
    diff = np.arange(10, dtype=float)
    means = block_bootstrap_means(diff, block=10, draws=300, rng=np.random.default_rng(0))
    np.testing.assert_allclose(means, 4.5)


def test_bootstrap_por_bloques_respeta_la_forma():
    means = block_bootstrap_means(np.arange(10, dtype=float), block=3, draws=500, rng=np.random.default_rng(0))
    assert means.shape == (500,)
    assert 0 <= means.min() and means.max() <= 9


def test_load_series_repite_el_dia_anterior_y_no_mira_el_siguiente(tmp_path):
    p = tmp_path / "s.csv"
    pd.DataFrame({"date": ["2025-01-01", "2025-01-02", "2025-01-04"], "usd_dop": [60.0, 61.0, 63.0]}).to_csv(p, index=False)
    s = load_series(p)
    assert len(s) == 4
    assert s["2025-01-03"] == pytest.approx(61.0)


def test_un_empate_de_direccion_no_cuenta_como_fallo():
    values = series(600) + np.random.default_rng(5).normal(0, 0.2, 600)
    folds = run(values, {"naive": naive, "median5": median5}, horizons=(1,), min_train=365, step=7)
    row = summarize(folds).query("model == 'median5'").iloc[0]
    ties = (folds[folds.model == "median5"].forecast == folds[folds.model == "median5"]["last"]).sum()
    assert row.direction_n == row.n - ties


def test_los_datos_reales_no_tienen_huecos_ni_valores_absurdos():
    from pathlib import Path

    s = load_series(Path(__file__).parent.parent / "data" / "usd_dop.csv")
    assert s.index.is_monotonic_increasing and s.index.to_series().diff().dropna().eq(pd.Timedelta("1D")).all()
    assert s.between(50, 70).all()
