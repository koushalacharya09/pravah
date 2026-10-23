# Sample datasets

Checked-in files under `data/` are synthetic. They are not an export from a live coaching centre. Dates sit in a neutral 2010 range so the files do not look like a product build stamp.

## How to regenerate the 400-case log

```text
python data/generate_sample_large.py
```

The generator uses `RANDOM_SEED = 42` from `pravah/ingest/constants.py`. The first enquiry is on 2010-03-01. Arrivals are spread over 24 weeks. The output is `data/sample_large.csv`.

## What was planted in `sample_large.csv`

**Process.** Four activities: enquiry → demo → join → attend. Every case starts with enquiry.

**Drop-offs.** About 12% never reach demo. Of those who see a demo, joining chance depends on the early-demo flag and the centre, so a large share stop after demo. About 10% of joiners never attend.

**Bottleneck.** The wait from the last demo to join is about 11–15 days. That is longer than the enquiry-to-demo wait and longer than the join-to-attend wait, so join is the intended bottleneck.

**Rework.** About 14% of demo cases get a second demo two and a half days later before they are allowed to join.

**Tutor effect.** `Asha Kulkarni` takes about two days from join to attend. `Vikram Joshi` takes about nine. The engine hashes tutor names; the time gap remains.

**Known cause.** Reaching demo within 48 hours of enquiry raises the probability of joining by 0.28, on top of a centre confounder (North is both more likely to get an early demo and more likely to join). The product treatment `Reached demo within the median time` matches that early-demo split because late demos are set at six days.

**Open cases.** The last 30 arrivals have only an enquiry, close to the end of the calendar, so they stay open rather than counting as dropouts.

The smaller samples (`sample_event_log.csv`, `sample_wide_sheet.csv`, `sample_generic.csv`, `sample_identifiers.csv`, `sample_log.xes`) are pedagogical. Event-log and wide-sheet calendars also use 2010.
