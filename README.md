Español: [README.es.md](README.es.md)

# dop-usd-backtest

An honest backtest of USD to DOP exchange rate forecasts: five models against the simplest one of all (repeat the last value), with a rolling origin and no peeking at the future. DOP is the Dominican peso.

I built it to practice the part of forecasting that is almost always done badly: measuring. It is easy to fit an ARIMA and show a nice curve. The hard part is checking that it beats doing nothing.

**Stack:** Python 3.13 · pandas · statsmodels · pytest · uv

## Result

No model beats the last value. Data: 972 days (2024-03-06 to 2026-11-02), 83 origins (one every 7 days, with at least a year of history). The table shows mean absolute error in pesos per dollar, and how much more (+) or less (-) error each model has than the naive one. The interval is 95% for that difference, using a block bootstrap.

| Model | h=1 day | h=7 days | h=30 days |
|---|---|---|---|
| naive (last value) | 0.177 | 0.468 | 1.101 |
| drift (trend of the last year) | -0.2 % (includes 0) | +1.5 % (includes 0) | **+7.0 %, worse** |
| ETS with damped trend | **+10.9 %, worse** | +2.2 % (includes 0) | +5.6 % (includes 0) |
| ARIMA(1,1,1) | **+10.3 %, worse** | **+2.8 %, worse** (barely) | +0.2 % (includes 0) |
| 5-day median | **+57.8 %, worse** | **+11.4 %, worse** | +2.4 % (includes 0) |

"Worse" means the interval of the difference sits entirely above zero. "Includes 0" means that with 83 origins I cannot tell it apart from the naive model in either direction. ARIMA's "worse" at 7 days is the fairest call: its lower bound is 0.0001 pesos. I tried 10 bootstrap seeds and it always stayed above zero, but only just. I also do not correct for having made 12 comparisons, which only weakens the "worse" calls.

Directional accuracy (whether the model guesses if the rate goes up or down) never gets past 57 % and 9 of 12 cases fall below 50 %. When a model predicts "same as today" (it happens to the 5-day median in 9 of 83 origins) I count that origin as neither a hit nor a miss.

![Each model's error relative to naive, with a 95% interval](results/error_vs_naive.png)

The full table, with the intervals, comes from `uv run python -m dopusd.backtest` and ends up in `results/summary.csv`.

## How it is built

- `fetch.py` downloads the rate day by day into `data/usd_dop.csv`.
- `backtest.py` walks through the origins. At each one it hands the model only `values[: origen + 1]` and compares with the value `h` days later.
- `models.py` has the five forecasters. They all share the same signature, so adding one takes ten lines.
- The intervals use a circular block bootstrap. With one origin per week and a 30-day horizon, neighboring origins share part of the path; resampling them one by one would make the intervals narrower than they really are.

## How to run it

```bash
uv sync
uv run python -m dopusd.backtest      # usa data/usd_dop.csv, que ya está en el repo
uv run pytest
```

To download the data again: `uv run python -m dopusd.fetch` (takes a minute, about a thousand requests).

## What to know before believing it

- **The data is not the official rate.** It comes from `fawazahmed0/currency-api`, a free aggregate published on jsDelivr. It is not the Central Bank's rate or any bank's. It is noisy: on 2026-09-29 it shows 57.72 when the previous and next days are around 59.5, an isolated value that does not repeat. The autocorrelation of daily returns is -0.23, more negative than I would expect from a currency that moves little, and I think it is partly that noise. I did not clean the series: a centered filter would look at each origin's future.
- **It is one series, two years and 83 origins.** It is enough to see that ARIMA and ETS add nothing here. It says nothing about other currencies or other periods.
- **One day is missing** (2025-12-10) and I repeat the previous day's value. I used to interpolate it, and since that day is exactly an origin of the backtest, the model saw the next day's value. A review found it; it changed the 1-day naive from 0.1757 to 0.1767 and no conclusion.
- **There are no models with external variables** (inflation, policy rate, reserves). A model like that might beat naive, but I would need that data with the same publication date a real user would have, and I could not get it.
- **It is not a recommendation of anything.**

The decisions and what I discarded are in [docs/decisions.md](docs/decisions.md).
