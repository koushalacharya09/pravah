# PRAVAH

**Hidden process intelligence.** Upload a spreadsheet. See whether the table is clean, where work gets stuck, and what is safe to treat as a cause.

[![Live](https://img.shields.io/badge/Live-pravah--lemon.vercel.app-0f5f5a?style=flat-square)](https://pravah-lemon.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-required-yellow?style=flat-square)](tests/)

**Live:** [https://pravah-lemon.vercel.app](https://pravah-lemon.vercel.app)

Designed & built by [Koushal Acharya](https://github.com/koushalacharya09).

---

## What it does

A founder uploads a CSV, Excel, or XES file (up to 3 MB). The engine reads it in memory and never stores it. Personal columns are hashed before roles are scored.

Two scores come back:

- **Data quality** on every table
- **Process readiness** only when a case, an activity, and a time exist

From a process log it then computes bottlenecks, Kaplan–Meier survival, open-case risk, causal checks (IPW + refutation), a discrete-event what-if, and an action plan whose gains come from the simulator.

There is no language model in the core. Numbers are computed. Correlation is not treated as causation.

## Try it

On the live page:

1. **Try a realistic dataset** — 400-case coaching log with drop-offs, a join bottleneck, rework, a tutor effect, and a planted early-demo cause
2. **Try an event log** — 8 students, roles already labelled
3. **Try a wide sheet** — one row per student, stage dates
4. **Try a plain table** — quality only; process checks stay blank

How the large sample was generated: [`docs/DATA.md`](docs/DATA.md)

## Run locally

```bash
python -m pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

```bash
python -m pytest tests
```

Python 3.11 or newer. Runtime dependencies are pinned in `requirements.txt` (Flask, NumPy, pandas, openpyxl, lifelines, scikit-learn).

## Stack

| Layer | Choice |
| --- | --- |
| Engine | Python, NumPy, pandas — algorithms in-tree except Cox (lifelines) and the risk arena (scikit-learn) |
| App | Flask (`app.py`), static UI in `public/` |
| Hosting | Vercel, production URL `https://pravah-lemon.vercel.app` |
| Privacy | Uploads stay in memory (`BytesIO`). Names, emails, phones, and profile URLs are hashed before schema scoring |

## Repo map

```
pravah/     ingestion, bottlenecks, survival, risk, causes, simulation, action plan
public/     live page, About, Learn
data/       built-in samples
docs/       theory, decisions, data generation, project report
tests/      pytest
```

Method and terms: [About](https://pravah-lemon.vercel.app/about) · [Learn](https://pravah-lemon.vercel.app/learn)

## License

Private project source, published for portfolio. Ask before reuse.
