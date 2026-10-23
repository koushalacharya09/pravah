"""Next-activity Markov chain and absorption into complete or drop."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pravah.ingest.datetime_parse import parse_datetimes
from pravah.ingest.models import SchemaResult

DROP = "drop"
COMPLETE = "complete"


def transition_counts(frame: pd.DataFrame, schema: SchemaResult, completion: str | None) -> dict[tuple[str, str], int]:
    """Count each step that actually followed another inside a case.

    WHAT: Sort each case by time. Each consecutive pair is one transition. A case that reaches ``completion`` stops there and that state is absorbing. A case that ends anywhere else steps into ``drop``.
    WHY: The next-step question is a transition count. Absorption turns those counts into the chance a case now in a step eventually finishes or disappears.
    WHY NOT: A transition for every pair of activities the case ever touched. That would count a step as next when other steps sat in between. Treating the last observed activity of an open case as drop would call a student who enquired yesterday a dropout.
    ASSUMPTIONS: The completion activity is absorbing. Every other ending in this finished file is treated as drop for the chain. The caller decides which activity is completion.
    COMPLEXITY: O(n log n) time, O(a^2) space for ``a`` activities.
    FAILS WHEN: The roles are missing. The count dictionary is empty. A one-event case only contributes a transition into complete or drop.
    """
    if not (schema.case_id and schema.activity and schema.timestamp):
        return {}
    if schema.case_id not in frame.columns or schema.activity not in frame.columns or schema.timestamp not in frame.columns:
        return {}
    working = frame.dropna(subset=[schema.case_id, schema.activity, schema.timestamp]).copy()
    working[schema.timestamp] = parse_datetimes(working[schema.timestamp])
    working = working.dropna(subset=[schema.timestamp])
    counts: dict[tuple[str, str], int] = {}
    ordered = working.sort_values([schema.case_id, schema.timestamp], kind="mergesort")
    for _case, group in ordered.groupby(schema.case_id, sort=False):
        labels = [str(value) for value in group[schema.activity]]
        if completion and completion in labels:
            labels = labels[: labels.index(completion) + 1]
            sink = COMPLETE
        else:
            sink = DROP
        for left, right in zip(labels, labels[1:]):
            counts[(left, right)] = counts.get((left, right), 0) + 1
        if labels:
            counts[(labels[-1], sink)] = counts.get((labels[-1], sink), 0) + 1
    return counts


def absorption_probabilities(counts: dict[tuple[str, str], int]) -> list[dict[str, object]]:
    """Chance of being absorbed in complete or drop, starting from each activity.

    WHAT: Transient activities have a transition matrix ``Q`` and a matrix ``R`` into the two absorbing states. The absorption matrix is ``(I - Q)^{-1} R``. Each row is the chance of complete and of drop.
    WHY: Solving the linear system is the definition of absorption for a finite Markov chain. Simulation would only approximate a number the system already determines.
    WHY NOT: Multiplying the transition matrix for a fixed number of steps and stopping. That truncates long paths. A next-activity accuracy score is a different question and is not an absorption probability.
    ASSUMPTIONS: An activity that also appears as a successor of itself stays transient unless it is the completion label, which is already absorbing and is not in ``Q``. The two absorbing states are ``complete`` and ``drop``.
    COMPLEXITY: O(a^3) for the linear solve. Space O(a^2).
    FAILS WHEN: ``I - Q`` is singular because some activities cannot leave a closed loop. Those activities are reported with a null probability and the sentence says the chain did not have a unique absorption.
    """
    activities = sorted({left for left, _right in counts if left not in {COMPLETE, DROP}})
    if not activities:
        return []
    index = {name: position for position, name in enumerate(activities)}
    size = len(activities)
    q = np.zeros((size, size))
    r = np.zeros((size, 2))
    for (left, right), count in counts.items():
        if left not in index:
            continue
        if right in index:
            q[index[left], index[right]] += count
        elif right == COMPLETE:
            r[index[left], 0] += count
        elif right == DROP:
            r[index[left], 1] += count
    totals = q.sum(axis=1) + r.sum(axis=1)
    for row in range(size):
        if totals[row] > 0:
            q[row, :] /= totals[row]
            r[row, :] /= totals[row]
    eye = np.eye(size)
    try:
        absorbed = np.linalg.solve(eye - q, r)
    except np.linalg.LinAlgError:
        return [
            {"activity": name, "complete": None, "drop": None, "sentence": f"{name} sits in a loop the chain cannot leave."}
            for name in activities
        ]
    rows = []
    for position, name in enumerate(activities):
        complete = float(absorbed[position, 0])
        drop = float(absorbed[position, 1])
        rows.append(
            {
                "activity": name,
                "complete": round(complete, 4),
                "drop": round(drop, 4),
                "sentence": f"From {name}, the chain puts {complete:.0%} on complete and {drop:.0%} on drop.",
            }
        )
    return rows
