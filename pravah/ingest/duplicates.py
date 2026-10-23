"""Near-duplicate people, compared only inside a block."""

from __future__ import annotations

import pandas as pd

from pravah.ingest.anonymise import name_match_rate, phone_match_rate

_SIMILARITY = 0.84
_MAX_BLOCK = 40


def levenshtein_distance(left: str, right: str) -> int:
    """Edit distance by the standard dynamic program.

    WHAT: The minimum insertions, deletions, and substitutions that turn ``left`` into ``right``. ``D(i,j) = D(i-1,j-1)`` when the characters match, otherwise ``1 + min(insert, delete, substitute)``.
    WHY: "Rahul Sharma" and "Rahul Sharmaa" are the same person to a reader and different strings to an exact duplicate check. Process logs care about the person, so the comparison has to allow a small edit.
    WHY NOT: A token-set comparison would treat the two names as a match too, and would also treat "Sharma Rahul" as the same, but it ignores how many letters changed inside a token. Soundex collapses too many Indian names into one bucket.
    ASSUMPTIONS: Both strings are already stripped and lower-cased. The relevant edits are single characters, not a reordering of whole words.
    COMPLEXITY: O(len(left) * len(right)) time and O(len(right)) space with one rolling row.
    FAILS WHEN: Two different people have short names inside the cutoff, such as "Ram" and "Raj". Blocking and the 0.84 similarity cutoff are what limit that, and they do not remove it.
    """
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1]


def _similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    distance = levenshtein_distance(left, right)
    return 1.0 - distance / max(len(left), len(right))


def _normalise(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def _phone_key(value: object) -> str:
    digits = "".join(character for character in str(value) if character.isdigit())
    return digits if len(digits) >= 10 else ""


def find_fuzzy_duplicates(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Candidate duplicate people, without writing the raw name into the result.

    Rows are compared only when they share a phone number or, failing that, the same first letter, and only inside blocks of at most 40 rows.
    """
    if frame.empty:
        return []
    name_columns = [
        str(name)
        for name in frame.columns
        if name_match_rate(frame[name]) >= 0.5
    ]
    if not name_columns:
        return [
            {
                "similarity": None,
                "sentence": "No name column was found, so near-duplicate people were not compared.",
            }
        ]
    name_column = name_columns[0]
    phone_columns = [str(name) for name in frame.columns if phone_match_rate(frame[name]) >= 0.5]
    phone_column = phone_columns[0] if phone_columns else None
    records: list[tuple[int, str, str]] = []
    for index, value in frame[name_column].items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = _normalise(value)
        if not text:
            continue
        phone = ""
        if phone_column is not None and not pd.isna(frame.at[index, phone_column]):
            phone = _phone_key(frame.at[index, phone_column])
        records.append((int(index) if str(index).isdigit() else len(records), text, phone))
    blocks: dict[str, list[int]] = {}
    for position, (_row, text, phone) in enumerate(records):
        key = f"phone:{phone}" if phone else f"letter:{text[0]}"
        blocks.setdefault(key, []).append(position)
    seen: set[tuple[int, int]] = set()
    pairs: list[dict[str, object]] = []
    for key, members in blocks.items():
        if len(members) > _MAX_BLOCK:
            continue
        for i, left_pos in enumerate(members):
            for right_pos in members[i + 1 :]:
                ordered = (min(left_pos, right_pos), max(left_pos, right_pos))
                if ordered in seen:
                    continue
                seen.add(ordered)
                similarity = _similarity(records[left_pos][1], records[right_pos][1])
                if similarity < _SIMILARITY:
                    continue
                block = "the same phone" if key.startswith("phone") else "the same first letter"
                left_row = records[left_pos][0] + 1
                right_row = records[right_pos][0] + 1
                pairs.append(
                    {
                        "similarity": round(similarity, 3),
                        "block": block,
                        "sentence": (
                            f"Data rows {left_row} and {right_row} look like the same person "
                            f"(name similarity {similarity:.0%}, blocked by {block}). The names are not shown."
                        ),
                    }
                )
    pairs.sort(key=lambda item: float(item["similarity"]), reverse=True)
    if not pairs:
        return [
            {
                "similarity": None,
                "sentence": "No near-duplicate names were found inside a phone or first-letter block.",
            }
        ]
    return pairs[:20]
