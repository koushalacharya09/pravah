# Architecture

PRAVAH reads an ordinary spreadsheet and rebuilds the process hidden in it.
Phase 1 stops at a validated event table. Later phases consume that table.
They are not implemented, and this package does not import them.

## Phase 1 flow

1. **Load.** CSV and Excel through pandas. XES through pm4py when it is installed, otherwise a small XML reader that copies trace and event attributes. The extension selects the reader.
2. **Dates and near-duplicates, on the raw sheet.** Slash dates are resolved before any month-first guess. Fuzzy name pairs are recorded here, because the next step destroys edit distance.
3. **Anonymise.** Phone numbers, emails, and personal names are replaced with a stable SHA-256 prefix before any scoring. A hashed column cannot be chosen as the case, the activity, or the time. It can still be the resource, so a tutor column remains groupable.
3. **Shape.** Two scores, long and wide. The winner must clear 0.55 or the sheet is a generic table. Confidence shrinks when the two scores are within 0.20 of each other.
4. **Rebuild, wide sheets only.** Pairwise precedence, a Wilson lower bound, then Kahn's algorithm. A cycle loses its weakest edge and the removal is reported. Events are emitted in stage order so a backdated cell shows up as an inversion.
5. **Schema.** Scores, not column names, in `schema_scores.py`. `schema.py` assigns the winners. Roles under 0.45 stay empty.
6. **Health.** Five weighted rates, each with a 95% Wilson interval. The headline score is the weighted mean and does not pretend to have its own interval.
7. **Deep profile, still on the unmelted sheet.** Missingness, modified-z outliers, the typed relationship matrix, Benford, and the readiness table. Blank stage dates only exist before the melt.
8. **Narrative.** Template sentences filled with those computed values, plus the safe-to-analyse line.

A public page is `app.py`. It serves `public/index.html` and `POST /api/ingest`. Vercel detects the Flask `app` object and routes the deployment through it. The page does not analyse anything itself.

The object a caller keeps is `IngestResult`. `frame` is the working table: a long log in its original row order, or a rebuilt wide sheet sorted by case and time. `validation` was computed on the pre-sort wide sheet, so the health score still sees the backdated cell.

## What runs on the event table

Bottlenecks, survival, risk, causes, simulation, and the action plan run after the event table exists. Each is imported inside `ingest`, not at package import, because the ingest package is still initialising. A sheet without a case, an activity, and a time gets an explicit refusal from each of them. Risk also refuses when finished prefixes are under 10 per feature. Simulation refuses scenario numbers when the baseline misses the observed median or the stage shares. The action plan copies gains only from a validated scenario.

Discovery and drift monitoring are not built. NetworkX and SciPy appear only in tests. Shapley values are computed in this package. Lifelines supplies the Cox fit and the proportional-hazards check. scikit-learn supplies the classifiers, the propensity model, and the isotonic calibration.

## Where a new column enters

No module contains the string `"case_id"` as a required input name. Reconstruction mints `activity` and `timestamp` when a wide sheet has no such columns, then schema detection scores those minted columns the same way it scores any other column.
