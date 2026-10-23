"""Choose DD/MM or MM/DD as a hypothesis, then parse the column."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pravah.ingest.datetime_parse import (
    _ISO_DATE,
    _SLASH_DATE,
    infer_dayfirst,
    is_numeric,
    is_stringy,
    parse_datetimes,
)
from pravah.ingest.datetime_parse import _parse_with_dayfirst


@dataclass(frozen=True)
class DateFormatDecision:
    column: str
    choice: str
    confidence: float
    method: str
    sentence: str


def _ambiguous_slash_count(series: pd.Series) -> int:
    count = 0
    for raw in series.dropna():
        text = str(raw).strip()
        if _ISO_DATE.match(text):
            continue
        match = _SLASH_DATE.match(text)
        if match is None or len(match.group(1)) == 4:
            continue
        first, second = int(match.group(1)), int(match.group(2))
        if first <= 12 and second <= 12:
            count += 1
    return count


def _order_consistency(parsed: dict[str, pd.Series]) -> float:
    """Share of pairs that respect left-to-right column order.

    A fully reversed calendar still has a perfect undirected order. The useful
    question is whether the columns, as written, go earlier to later.
    """
    names = list(parsed)
    scores: list[float] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            both = pd.DataFrame({"left": parsed[left], "right": parsed[right]}).dropna()
            if len(both) < 5:
                continue
            scores.append(float((both["left"] < both["right"]).mean()))
    if not scores:
        return 0.5
    return sum(scores) / len(scores)


def _parse_hypothesis(frame: pd.DataFrame, columns: list[str], *, dayfirst: bool) -> dict[str, pd.Series]:
    parsed: dict[str, pd.Series] = {}
    for name in columns:
        series = frame[name]
        if pd.api.types.is_datetime64_any_dtype(series):
            parsed[name] = pd.to_datetime(series, errors="coerce")
        else:
            parsed[name] = _parse_with_dayfirst(series.astype("string"), dayfirst=dayfirst)
    return parsed


def resolve_date_formats(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[DateFormatDecision]]:
    """Parse date columns, using stage-order consistency when the day and month are both plausible.

    WHAT: A value with a part above 12 is a vote for DD/MM or MM/DD. If a column has only one kind of vote, that format is used and the confidence is ``decisive / (decisive + ambiguous)``. If every slash date is ambiguous, both formats are parsed and the one with the higher left-to-right consistency wins. Consistency is the average, over column pairs in sheet order, of ``P(left < right)``. The confidence is ``min(1, |score_dmy - score_mdy| / 0.3)``. A gap under 0.05 is a tie, and the Indian prior DD/MM is used at confidence 0.35.
    WHY: ``03/04/2026`` is 3 April or 4 March. Indian operational sheets are usually DD/MM, but a silent prior would misread an export that is MM/DD. Impossible days are proof. When there is no proof, the format that makes the stage dates agree with one order is the better hypothesis.
    WHY NOT: Always using ``dayfirst=True`` matches India and fails a US export with no warning. Always using month-first is what the parser did before this rule, and it scrambles a DD/MM sheet whose days never exceed 12. Asking the user is clearer and is not available on a shared link.
    ASSUMPTIONS: A sheet uses one slash convention. Stage columns are the slash-date columns plus columns that already parsed. Pairs with fewer than 5 co-observed rows do not vote.
    COMPLEXITY: O(n c + c^2 n) time for ``n`` rows and ``c`` date columns. Space O(n c) for the two candidate parses.
    FAILS WHEN: The columns are not written earlier-to-later, so the left-to-right score prefers the wrong calendar. A tie then uses day/month at low confidence. A single ambiguous date column cannot use stage order either.
    """
    resolved = frame.copy()
    decisions: list[DateFormatDecision] = []
    ambiguous: list[str] = []
    known: list[str] = []
    for name in frame.columns:
        series = frame[name]
        if is_numeric(series) or (
            not is_stringy(series) and not pd.api.types.is_datetime64_any_dtype(series)
        ):
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            known.append(str(name))
            continue
        mode = infer_dayfirst(series)
        ambiguous_rows = _ambiguous_slash_count(series)
        if mode in {"dmy", "mdy"}:
            votes = 0
            for raw in series.dropna().astype(str):
                match = _SLASH_DATE.match(raw.strip())
                if match is None or len(match.group(1)) == 4:
                    continue
                first, second = int(match.group(1)), int(match.group(2))
                if first > 12 or second > 12:
                    votes += 1
            confidence = 1.0
            resolved[name] = _parse_with_dayfirst(series.astype("string"), dayfirst=(mode == "dmy"))
            label = "day/month" if mode == "dmy" else "month/day"
            decisions.append(
                DateFormatDecision(
                    column=str(name),
                    choice=mode,
                    confidence=confidence,
                    method="impossible_day",
                    sentence=(
                        f"{name} is {label} because {votes} value(s) have a part above 12, "
                        f"which the other order cannot be. Confidence {confidence:.0%}."
                    ),
                )
            )
            known.append(str(name))
        elif mode == "conflict":
            resolved[name] = parse_datetimes(series)
            decisions.append(
                DateFormatDecision(
                    column=str(name),
                    choice="conflict",
                    confidence=0.2,
                    method="conflict",
                    sentence=(
                        f"{name} mixes day/month and month/day. Ambiguous values were left blank "
                        "rather than guessed."
                    ),
                )
            )
        elif ambiguous_rows >= 5:
            ambiguous.append(str(name))
        elif parse_datetimes(series).notna().mean() >= 0.8:
            resolved[name] = parse_datetimes(series)
            decisions.append(
                DateFormatDecision(
                    column=str(name),
                    choice="iso",
                    confidence=1.0,
                    method="iso",
                    sentence=f"{name} is an unambiguous calendar date.",
                )
            )
            known.append(str(name))
    if not ambiguous:
        return resolved, decisions
    pool = known + ambiguous
    dmy_score = _order_consistency(_parse_hypothesis(frame, pool, dayfirst=True))
    mdy_score = _order_consistency(_parse_hypothesis(frame, pool, dayfirst=False))
    gap = abs(dmy_score - mdy_score)
    if gap < 0.05:
        choice = "dmy"
        method = "prior"
        confidence = 0.35
        reason = (
            f"day/month and month/day are equally consistent "
            f"({dmy_score:.0%} vs {mdy_score:.0%}), so the Indian day/month prior is used"
        )
    elif dmy_score > mdy_score:
        choice = "dmy"
        method = "stage_order"
        confidence = min(1.0, gap / 0.3)
        reason = (
            f"day/month keeps a cleaner stage order ({dmy_score:.0%} consistent) "
            f"than month/day ({mdy_score:.0%})"
        )
    else:
        choice = "mdy"
        method = "stage_order"
        confidence = min(1.0, gap / 0.3)
        reason = (
            f"month/day keeps a cleaner stage order ({mdy_score:.0%} consistent) "
            f"than day/month ({dmy_score:.0%})"
        )
    covered = ", ".join(ambiguous)
    for name in ambiguous:
        resolved[name] = _parse_with_dayfirst(frame[name].astype("string"), dayfirst=(choice == "dmy"))
    decisions.append(
        DateFormatDecision(
            column=covered,
            choice=choice,
            confidence=round(confidence, 3),
            method=method,
            sentence=f"{reason}. Confidence {confidence:.0%}. One decision for {covered}.",
        )
    )
    return resolved, _collapse_decisions(decisions)


def _collapse_decisions(decisions: list[DateFormatDecision]) -> list[DateFormatDecision]:
    """One sentence when every date column was read the same way."""
    if len(decisions) <= 1:
        return decisions
    signatures = {(item.choice, item.method, item.confidence) for item in decisions}
    if len(signatures) != 1:
        return decisions
    first = decisions[0]
    names = ", ".join(item.column for item in decisions)
    return [
        DateFormatDecision(
            column=names,
            choice=first.choice,
            confidence=first.confidence,
            method=first.method,
            sentence=(
                f"Every date column was read as {first.choice} ({first.method}). "
                f"Confidence {first.confidence:.0%}. Columns: {names}."
            ),
        )
    ]
