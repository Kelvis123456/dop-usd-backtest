"""Baja la tasa USD -> DOP día por día y la guarda en data/usd_dop.csv.

La fuente es fawazahmed0/currency-api, que publica una foto por fecha en jsDelivr. No es la tasa
oficial del Banco Central: es un agregado gratuito. Ver el README para lo que eso implica.
"""

import asyncio
import csv
import sys
from datetime import date, timedelta
from pathlib import Path

import httpx

URLS = (
    "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{d}/v1/currencies/usd.json",
    "https://{d}.currency-api.pages.dev/v1/currencies/usd.json",
)


async def fetch_day(client: httpx.AsyncClient, day: date, sem: asyncio.Semaphore) -> tuple[date, float | None]:
    async with sem:
        for url in URLS:
            for attempt in range(3):
                try:
                    r = await client.get(url.format(d=day.isoformat()), timeout=20)
                    if r.status_code == 404:
                        break  # esa fecha no existe en este espejo, se prueba el otro
                    r.raise_for_status()
                    rate = r.json()["usd"]["dop"]
                    return day, float(rate)
                except (httpx.HTTPError, KeyError, ValueError):
                    await asyncio.sleep(1 + attempt)
        return day, None


async def main(start: date, end: date, out: Path) -> None:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(fetch_day(client, d, sem) for d in days))
    ok = [(d, r) for d, r in results if r is not None]
    missing = [d for d, r in results if r is None]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "usd_dop"])
        w.writerows((d.isoformat(), f"{r:.6f}") for d, r in ok)
    print(f"{len(ok)} días guardados en {out}; faltan {len(missing)}")
    if missing:
        print("primeros faltantes:", [d.isoformat() for d in missing[:10]])


if __name__ == "__main__":
    start = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2024, 3, 6)
    end = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2026, 11, 2)
    asyncio.run(main(start, end, Path("data/usd_dop.csv")))
