"""Result objects for ingestion. Frames are not frozen: callers may sort them."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class LoadResult:
    """A table as read from disk, before anonymisation or typing."""

    source_format: str
    source_name: str
    n_rows: int
    n_columns: int
    reader: str


@dataclass(frozen=True)
class ShapeResult:
    """Which kind of table this is, and how sure the heuristic is."""

    shape: str
    confidence: float
    long_score: float
    wide_score: float
    signals: dict[str, float]
    reasons: list[str]


@dataclass(frozen=True)
class SchemaResult:
    """Column roles. Missing roles stay None; they are not guessed past the cutoff."""

    case_id: str | None
    activity: str | None
    timestamp: str | None
    resource: str | None
    column_scores: dict[str, dict[str, float]]
    confidence: float
    notes: list[str]


@dataclass(frozen=True)
class PrecedenceEdge:
    """A directed claim that ``source`` happens before ``target``."""

    source: str
    target: str
    probability: float
    support: int
    lower: float
    upper: float


@dataclass(frozen=True)
class DroppedEdge:
    """An edge removed because it was the weakest link in a cycle."""

    source: str
    target: str
    probability: float
    support: int
    lower: float
    upper: float
    reason: str


@dataclass(frozen=True)
class PairPrecedence:
    """One unordered pair of date columns and the vote each direction received."""

    left: str
    right: str
    support: int
    left_before: int
    right_before: int
    ties: int
    p_left_before: float
    p_right_before: float
    left_lower: float
    left_upper: float
    right_lower: float
    right_upper: float
    kept_direction: str | None
    provisional: bool
    reject_reason: str


@dataclass
class ReconstructionResult:
    """Wide sheet melted into events, in sheet order then stage order.

    ``frame`` is intentionally not sorted by time. Validation counts inversions
    against this stage order. The pipeline sorts a copy for downstream use.
    """

    frame: pd.DataFrame
    activity_column: str
    timestamp_column: str
    order: list[str]
    edges: list[PrecedenceEdge]
    dropped_edges: list[DroppedEdge]
    pairs: list[PairPrecedence]
    notes: list[str]


@dataclass(frozen=True)
class RateReport:
    """One ingredient of the health score, with a Wilson interval on the rate."""

    name: str
    bad_count: int
    total: int
    bad_fraction: float
    score: float
    lower: float
    upper: float
    required_total: int
    sufficient: bool
    unit: str
    detail: str
    applicable: bool = True
    rough: bool = False
    technical: str = ""


@dataclass
class ValidationResult:
    """0-100 health score. The score is a weighted sum, not a probability."""

    quality_score: float
    weights: dict[str, float]
    components: dict[str, RateReport]
    n_rows: int
    n_cases: int | None
    sufficient_for_inference: bool
    minimum_rows_recommended: int
    sample_sentence: str
    summary: str
    data_quality_score: float = 0.0
    process_readiness_score: float | None = None
    data_components: dict[str, RateReport] = field(default_factory=dict)


@dataclass
class IngestResult:
    """Everything phase 1 knows about one upload."""

    frame: pd.DataFrame
    load: LoadResult
    shape: ShapeResult
    schema: SchemaResult
    reconstruction: ReconstructionResult | None
    validation: ValidationResult
    anonymised_columns: list[str]
    narrative: str
    warnings: list[str] = field(default_factory=list)
    profile: dict[str, object] | None = None
    stages: list[dict[str, str]] = field(default_factory=list)
    bottlenecks: dict[str, object] | None = None
    survival: dict[str, object] | None = None
    risk: dict[str, object] | None = None
    causes: dict[str, object] | None = None
    simulation: dict[str, object] | None = None
    plan: dict[str, object] | None = None
