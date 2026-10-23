"""Write data/sample_large.csv. Seed 42. Neutral calendar starting 2010-03-01."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from pravah.ingest.constants import RANDOM_SEED

OUTPUT = Path(__file__).resolve().parent / "sample_large.csv"
N_CASES = 400
ORIGIN = pd.Timestamp("2010-03-01T09:00:00")
FIRST = [
    "Asha", "Dev", "Meera", "Kabir", "Nisha", "Omar", "Riya", "Vikram", "Neha", "Arjun",
    "Pooja", "Kiran", "Farah", "Gita", "Lata", "Rahul", "Anita", "Ravi", "Omkar", "Isha",
]
LAST = [
    "Rao", "Shah", "Iyer", "Khan", "Paul", "Seth", "Das", "Nair", "Jain", "Mehta",
    "Sen", "Ali", "Bose", "Menon", "Dixit", "Roy", "Lal", "Kapoor", "Bhatt", "Gill",
]


def _stamp(days: float) -> str:
    return (ORIGIN + pd.Timedelta(float(days), unit="D")).strftime("%Y-%m-%dT%H:%M:%S")


def generate(n_cases: int = N_CASES, seed: int = RANDOM_SEED) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    n_open = 30
    n_closed = n_cases - n_open
    arrivals = np.sort(rng.uniform(0.0, 24 * 7, n_closed))
    rows: list[dict[str, str]] = []
    latest = 0.0
    for index in range(n_closed):
        start = float(arrivals[index])
        north = bool(rng.random() < 0.5)
        centre = "North" if north else "South"
        fast_tutor = bool(rng.random() < 0.5)
        tutor = "Asha Kulkarni" if fast_tutor else "Vikram Joshi"
        name = f"{FIRST[index % len(FIRST)]} {LAST[(index // len(FIRST)) % len(LAST)]}"
        case = f"S{index + 1:03d}"
        phone = str(9100000100 + index)
        base = {
            "student_id": case,
            "student_name": name,
            "phone": phone,
            "tutor": tutor,
            "centre": centre,
        }
        rows.append({**base, "step": "enquiry", "event_time": _stamp(start)})
        latest = max(latest, start)
        if rng.random() > 0.88:
            continue
        early = bool(rng.random() < (0.62 if north else 0.38))
        demo_at = start + (1.0 if early else 6.0) + float(rng.uniform(-0.2, 0.2))
        rows.append({**base, "step": "demo", "event_time": _stamp(demo_at)})
        latest = max(latest, demo_at)
        last_demo = demo_at
        if rng.random() < 0.14:
            last_demo = demo_at + 2.4 + float(rng.uniform(-0.2, 0.2))
            rows.append({**base, "step": "demo", "event_time": _stamp(last_demo)})
            latest = max(latest, last_demo)
        join_p = float(np.clip(0.30 + 0.28 * int(early) + 0.18 * int(north), 0.08, 0.95))
        if rng.random() > join_p:
            continue
        join_at = last_demo + 12.0 + float(rng.uniform(-1.0, 3.0))
        rows.append({**base, "step": "join", "event_time": _stamp(join_at)})
        latest = max(latest, join_at)
        if rng.random() > 0.90:
            continue
        attend_gap = (2.0 if fast_tutor else 9.0) + float(rng.normal(0.0, 0.35))
        attend_at = join_at + max(0.5, attend_gap)
        rows.append({**base, "step": "attend", "event_time": _stamp(attend_at)})
        latest = max(latest, attend_at)
    for offset in range(n_open):
        index = n_closed + offset
        start = latest - float(rng.uniform(1.0, 6.0))
        north = bool(rng.random() < 0.5)
        centre = "North" if north else "South"
        tutor = "Asha Kulkarni" if rng.random() < 0.5 else "Vikram Joshi"
        name = f"{FIRST[index % len(FIRST)]} {LAST[(index // len(FIRST)) % len(LAST)]}"
        base = {
            "student_id": f"S{index + 1:03d}",
            "student_name": name,
            "phone": str(9100000100 + index),
            "tutor": tutor,
            "centre": centre,
        }
        rows.append({**base, "step": "enquiry", "event_time": _stamp(start)})
    return rows


def main() -> None:
    rows = generate()
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["student_id", "student_name", "phone", "step", "event_time", "tutor", "centre"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
