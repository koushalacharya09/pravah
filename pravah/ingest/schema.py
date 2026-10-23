"""Assign case, activity, timestamp, and resource. Names are not evidence."""

from __future__ import annotations

import pandas as pd

from pravah.ingest.constants import DATETIME_PARSE_RATE, ROLE_MIN_SCORE
from pravah.ingest.models import SchemaResult
from pravah.ingest.profile import profile_frame
from pravah.ingest.schema_scores import (
    _activity_score,
    _best,
    _case_score,
    _resource_score,
    _timestamp_score,
    _wide_case_scores,
)

def detect_schema(
    frame: pd.DataFrame,
    *,
    shape: str,
    anonymised_columns: list[str] | None = None,
) -> SchemaResult:
    """Assign case, activity, timestamp, and resource by score.

    WHAT: Each column receives a score in ``[0, 1]`` for each role. The winner
    of a role is the highest score that is still at least 0.45; ties go to the
    left-most column. Anonymised columns cannot be the case, the activity, or
    the timestamp. They can be the resource, because a tutor column is hashed
    on purpose and grouping must still work.

    The case score on a long log is ``repeat * fineness * (1 - nulls) *
    (0.55 + 0.45 * nunique / max_nunique)``. ``repeat`` is 1 when the average
    value appears on 2 to 30 rows. ``fineness`` is 1 when the column's
    cardinality ratio sits between 0.08 and 0.60. The activity score is
    ``(0.30 * card + 0.30 * coverage + 0.25 * within_case_variation + 0.15 * reuse)``
    times 1 for text and 0.6 for codes. ``coverage`` is the median share of the
    global label set visited by one case. The timestamp score is the parse rate
    times ``0.70 + 0.30 * within_case_variation``. The resource score rewards a
    value that appears in several cases and changes inside some of them; a
    column that never changes inside a case is halved, because that is a case
    attribute. On a wide sheet only the case is scored, as uniqueness times
    ``(1 - 0.5 * name_match_rate)``.

    WHY: Process-mining schema detection has to work for ``student_id`` /
    ``step`` / ``event_time`` and for columns in another language. Repeating
    finest grain is what a case id is. Coverage separates a stage label (one
    student walks through most of the catalogue) from a tutor (one student
    meets one or two people out of the staff). The cutoff 0.45 leaves the role
    empty instead of forcing a bad column into it.

    WHY NOT: A dictionary of column names is the usual production shortcut and
    is rejected because it fails the spec and fails the first non-English
    file. Picking the lowest-cardinality string as the activity is rejected
    because a region column is often smaller than the activity catalogue and
    does not vary inside the case. Mutual information with no role model is
    rejected because it still needs a definition of the target.

    ASSUMPTIONS: The sheet is already anonymised, so person-name columns are
    listed in ``anonymised_columns``. On a long log the case repeats. On a wide
    sheet the case is one row. Scores are comparable only inside one role.

    COMPLEXITY: ``O(n m)`` time for ``n`` rows and ``m`` columns, dominated by
    the grouped uniqueness counts. Space ``O(n)`` per grouped column.

    FAILS WHEN: One tutor owns every step of a case and no other column varies.
    The resource penalty then fires on the real resource; if it is the only
    candidate its halved score may still clear 0.45 and it is kept, but a
    noisier column can outrank it. A process where each case does one activity
    out of many has low coverage, so the activity can lose to a code. A wide
    sheet whose only identifier is a full name loses that name to anonymisation
    and may be left with no case id.
    """
    anonymised = set(anonymised_columns or [])
    profiles = profile_frame(frame)
    order = [profile.name for profile in profiles]
    notes: list[str] = []
    if shape == "wide_sheet":
        scores = _wide_case_scores(frame, profiles, anonymised)
        case_id, case_score = _best(scores, set(), order)
        if case_id is None or case_score < ROLE_MIN_SCORE:
            case_id = None
            notes.append("No column is unique enough to be the case on this wide sheet.")
        else:
            notes.append(
                "Stage names and event times are created when the date columns are unpivoted."
            )
        column_scores = {name: {"case_id": score} for name, score in scores.items()}
        return SchemaResult(
            case_id=case_id,
            activity=None,
            timestamp=None,
            resource=None,
            column_scores=column_scores,
            confidence=0.0 if case_id is None else case_score,
            notes=notes,
        )

    max_nunique = max(
        (
            profile.nunique
            for profile in profiles
            if profile.name not in anonymised and profile.parse_rate < DATETIME_PARSE_RATE
        ),
        default=1,
    )
    case_scores = {
        profile.name: _case_score(profile, max_nunique=max_nunique, anonymised=profile.name in anonymised)
        for profile in profiles
    }
    case_id, case_score = _best(case_scores, set(), order)
    if case_score < ROLE_MIN_SCORE:
        case_id = None
        case_score = 0.0

    timestamp_scores = {
        profile.name: 0.0
        if profile.name in anonymised
        else _timestamp_score(frame, profile, case_id)
        for profile in profiles
    }
    blocked = {case_id} if case_id else set()
    timestamp, timestamp_score = _best(timestamp_scores, blocked, order)
    if timestamp_score < ROLE_MIN_SCORE:
        timestamp = None
        timestamp_score = 0.0

    activity_scores = {
        profile.name: 0.0
        if profile.name in anonymised
        else _activity_score(frame, profile, case_id)
        for profile in profiles
    }
    blocked = {name for name in (case_id, timestamp) if name}
    activity, activity_score = _best(activity_scores, blocked, order)
    if activity_score < ROLE_MIN_SCORE:
        activity = None
        activity_score = 0.0

    n_cases = int(frame[case_id].nunique(dropna=True)) if case_id else len(frame)
    resource_scores = {
        profile.name: _resource_score(frame, profile, case_id, n_cases) for profile in profiles
    }
    blocked = {name for name in (case_id, timestamp, activity) if name}
    resource, resource_score = _best(resource_scores, blocked, order)
    if resource_score < ROLE_MIN_SCORE:
        resource = None

    column_scores = {
        profile.name: {
            "case_id": case_scores[profile.name],
            "activity": activity_scores[profile.name],
            "timestamp": timestamp_scores[profile.name],
            "resource": resource_scores[profile.name],
        }
        for profile in profiles
    }
    present = [score for score in (case_score, activity_score, timestamp_score)]
    confidence = sum(present) / 3.0
    if case_id is None:
        notes.append("No column scored as a case id. Health will treat that as a failed file.")
    if activity is None:
        notes.append("No column scored as the activity.")
    if timestamp is None:
        notes.append("No column scored as the event time.")
    if resource is None:
        notes.append("No resource column cleared the cutoff. That is allowed.")
    return SchemaResult(
        case_id=case_id,
        activity=activity,
        timestamp=timestamp,
        resource=resource,
        column_scores=column_scores,
        confidence=confidence,
        notes=notes,
    )
