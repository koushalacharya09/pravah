"""Turn a wide sheet of stage dates into a long event log."""

from __future__ import annotations

import pandas as pd

from pravah.ingest.constants import (
    DATETIME_PARSE_RATE,
    MIN_PAIR_SUPPORT,
    PRECEDENCE_LOWER_BOUND,
    PRECEDENCE_ROWS_FOR_PM10,
    PRECEDENCE_THRESHOLD,
)
from pravah.ingest.datetime_parse import parse_datetimes
from pravah.ingest.models import PairPrecedence, PrecedenceEdge, ReconstructionResult
from pravah.ingest.profile import profile_frame
from pravah.ingest.proportion import wilson_interval
from pravah.ingest.topo import resolve_precedence_order


def _fresh_name(existing: set[str], base: str) -> str:
    if base not in existing:
        return base
    suffix = 2
    while f"{base}_{suffix}" in existing:
        suffix += 1
    return f"{base}_{suffix}"


def pairwise_precedence(
    frame: pd.DataFrame,
    date_columns: list[str],
    *,
    threshold: float = PRECEDENCE_THRESHOLD,
    min_support: int = MIN_PAIR_SUPPORT,
    lower_bound: float = PRECEDENCE_LOWER_BOUND,
) -> tuple[list[PairPrecedence], list[PrecedenceEdge]]:
    """Estimate P(date_A < date_B) for every pair of stage columns.

    WHAT: For columns A and B, ``support`` is the number of rows where both
    dates parse. ``left_before`` counts rows with A strictly before B, and
    ties count rows with equal timestamps. The point estimate is
    ``left_before / support``. Each direction gets a 95% Wilson interval.
    An edge A -> B is kept when ``support >= min_support``, the point estimate
    is at least ``threshold`` (default 0.75), and the Wilson lower bound is
    strictly above ``lower_bound`` (default 0.5). The pair is provisional when
    support is below 73, the planning size for a ±10 point interval at p=0.75.

    WHY: A stage order has to be one order for the whole sheet, not a sort of
    each row. The pairwise rate uses only rows where both dates exist, so a
    stage that is often blank does not pull the mean. The lower-bound rule
    refuses a 4-out-of-5 lead, which is still compatible with a coin flip.
    The 0.75 rule refuses a precisely estimated but weak 60/40 split: that is
    not a stage order a founder should automate against.

    WHY NOT: Ranking columns by their mean date is the obvious alternative.
    It is rejected because one very late date moves the mean, and a stage that
    is missing on early cases looks artificially early. A binomial significance
    test is rejected because with thousands of rows a 52/48 split is
    "significant" and still operationally a tie. Kemeny-Young would search for
    the ranking that minimises pairwise disagreements; it is NP-hard, and the
    majority graph plus a topological sort is the standard tractable stand-in.

    ASSUMPTIONS: Among rows where both dates are present, each row is an
    independent draw of the same precedence probability. Ties are evidence for
    neither direction. The threshold is an effect size, not a p-value.

    COMPLEXITY: ``O(n c^2)`` time for ``n`` rows and ``c`` date columns, and
    ``O(c^2)`` space for the pairs. ``c`` is the number of stage columns.

    FAILS WHEN: The rows where both dates are filled are a biased subset (only
    successful cases have an enrol date, and those cases run in a different
    order). The estimate is then the order conditional on co-occurrence, not
    the order of all cases. Dependent rows from the same family also make the
    Wilson interval too narrow.
    """
    parsed = {name: parse_datetimes(frame[name]) for name in date_columns}
    pairs: list[PairPrecedence] = []
    edges: list[PrecedenceEdge] = []
    for i, left in enumerate(date_columns):
        for right in date_columns[i + 1 :]:
            both = pd.DataFrame({"left": parsed[left], "right": parsed[right]}).dropna()
            support = int(len(both))
            if support == 0:
                continue
            left_before = int((both["left"] < both["right"]).sum())
            right_before = int((both["right"] < both["left"]).sum())
            ties = support - left_before - right_before
            p_left, left_lo, left_hi = wilson_interval(left_before, support)
            p_right, right_lo, right_hi = wilson_interval(right_before, support)
            provisional = support < PRECEDENCE_ROWS_FOR_PM10
            kept: str | None = None
            reject = ""
            if support < min_support:
                reject = (
                    f"Only {support} rows have both dates; {min_support} are required "
                    "before a stage order is even proposed."
                )
            elif p_left >= threshold and p_left >= p_right and left_lo > lower_bound:
                kept = "left"
            elif p_right >= threshold and p_right > p_left and right_lo > lower_bound:
                kept = "right"
            elif max(p_left, p_right) < threshold:
                reject = (
                    f"The stronger direction is only {max(p_left, p_right):.0%}, "
                    f"below the {threshold:.0%} bar for a stage order."
                )
            else:
                reject = (
                    "The point estimate clears the bar, but the 95% interval still "
                    "reaches a coin-flip, so this direction is not used."
                )
            pair = PairPrecedence(
                left=left,
                right=right,
                support=support,
                left_before=left_before,
                right_before=right_before,
                ties=ties,
                p_left_before=p_left,
                p_right_before=p_right,
                left_lower=left_lo,
                left_upper=left_hi,
                right_lower=right_lo,
                right_upper=right_hi,
                kept_direction=kept,
                provisional=provisional,
                reject_reason=reject,
            )
            pairs.append(pair)
            if kept == "left":
                edges.append(
                    PrecedenceEdge(left, right, p_left, support, left_lo, left_hi)
                )
            elif kept == "right":
                edges.append(
                    PrecedenceEdge(right, left, p_right, support, right_lo, right_hi)
                )
    return pairs, edges


def reconstruct_wide(
    frame: pd.DataFrame,
    *,
    case_id: str,
    date_columns: list[str] | None = None,
    threshold: float = PRECEDENCE_THRESHOLD,
    forced_order: list[str] | None = None,
) -> ReconstructionResult:
    """Melt stage-date columns into events ordered by the precedence graph.

    WHAT: Date columns are the columns whose parse rate is at least 0.80,
    unless the caller passes them. Pairwise precedence builds a directed graph.
    Cycles are broken by ``resolve_precedence_order``. Each source row becomes
    one event per non-missing stage date, emitted in topological order. Other
    columns are copied onto every event of that case. Minted column names are
    ``activity`` and ``timestamp`` unless those names are already taken.

    WHY: Discovery, in a later phase, consumes a long event log. A wide sheet
    does not have an activity column; the column header is the activity. One
    global order, rather than a per-row sort, keeps the activity catalogue
    stable when a single row has a backdated cell.

    WHY NOT: Melting in left-to-right sheet order is simpler and is rejected
    because the sheet order is a formatting accident. Sorting each row by its
    own dates is rejected because two rows can imply two catalogues. Using the
    mean date per column is rejected for the same reason as in
    ``pairwise_precedence``.

    ASSUMPTIONS: ``case_id`` identifies the entity. A missing stage date means
    the stage did not happen, not that the timestamp is unknown. The graph
    built here is the one returned by ``pairwise_precedence``.

    COMPLEXITY: Precedence is ``O(n c^2)``. Emitting events is ``O(n c)``.
    Space is ``O(n c)`` for the long table.

    FAILS WHEN: ``case_id`` is missing, or fewer than two date columns parse.
    Those are raised as ``ValueError``. It also fails quietly when co-occurrence
    is biased: the order is then the order of the rows that filled both dates.
    A real loop in the business is reported as a dropped edge, not as a loop.
    """
    if case_id not in frame.columns:
        raise ValueError(f"Case column {case_id!r} is not on this sheet.")
    if date_columns is None:
        date_columns = [
            profile.name
            for profile in profile_frame(frame)
            if profile.name != case_id and profile.parse_rate >= DATETIME_PARSE_RATE
        ]
    if len(date_columns) < 2:
        raise ValueError(
            "A wide sheet needs at least two date columns before it can be rebuilt "
            f"as an event log. Found {len(date_columns)}."
        )

    pairs, edges = pairwise_precedence(frame, date_columns, threshold=threshold)
    order, dropped = resolve_precedence_order(date_columns, edges)
    if forced_order and set(forced_order) == set(date_columns):
        order = list(forced_order)
    notes: list[str] = []
    if not edges:
        notes.append(
            "No pair of dates was ordered strongly enough. Stages stay in the "
            "order of the columns on the sheet."
        )
    if any(pair.provisional for pair in pairs if pair.kept_direction):
        notes.append(
            f"At least one kept stage link rests on fewer than {PRECEDENCE_ROWS_FOR_PM10} "
            "rows with both dates, so the order is provisional."
        )
    for edge in dropped:
        notes.append(
            f"Dropped {edge.source} -> {edge.target} "
            f"({edge.probability:.0%} of {edge.support} rows). {edge.reason}"
        )

    existing = set(map(str, frame.columns))
    activity_column = _fresh_name(existing, "activity")
    timestamp_column = _fresh_name(existing, "timestamp")
    attribute_columns = [name for name in frame.columns if name not in date_columns]
    parsed = {name: parse_datetimes(frame[name]) for name in date_columns}
    rank = {name: position for position, name in enumerate(order)}

    records: list[dict[str, object]] = []
    for idx in frame.index:
        attributes = {column: frame.at[idx, column] for column in attribute_columns}
        stages = sorted(date_columns, key=lambda name: rank.get(name, 0))
        for stage in stages:
            timestamp = parsed[stage].loc[idx]
            if pd.isna(timestamp):
                continue
            record = dict(attributes)
            record[activity_column] = stage
            record[timestamp_column] = timestamp
            records.append(record)

    columns = attribute_columns + [activity_column, timestamp_column]
    long = pd.DataFrame.from_records(records, columns=columns)
    return ReconstructionResult(
        frame=long,
        activity_column=activity_column,
        timestamp_column=timestamp_column,
        order=order,
        edges=edges,
        dropped_edges=dropped,
        pairs=pairs,
        notes=notes,
    )
