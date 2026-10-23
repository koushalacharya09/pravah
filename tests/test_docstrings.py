"""Every algorithm docstring carries the interview headings."""

from __future__ import annotations

import pravah.ingest.anonymise as anonymise
import pravah.ingest.date_format as date_format
import pravah.ingest.datetime_parse as datetime_parse
import pravah.ingest.duplicates as duplicates
import pravah.ingest.loader as loader
import pravah.ingest.proportion as proportion
import pravah.ingest.readiness as readiness
import pravah.ingest.reconstruct as reconstruct
import pravah.ingest.schema as schema
import pravah.ingest.shape as shape
import pravah.ingest.topo as topo
import pravah.ingest.validate as validate
import pravah.perf.analyze as analyze
import pravah.perf.littles as littles
import pravah.perf.resources as resources
import pravah.perf.rework as rework
import pravah.perf.score as score
import pravah.perf.times as times
import pravah.perf.trend as trend
import pravah.causes.analyze as causes_analyze
import pravah.causes.effect as effect
import pravah.causes.refute as refute
import pravah.plan.analyze as plan_analyze
import pravah.plan.knapsack as knapsack
import pravah.risk.analyze as risk_analyze
import pravah.risk.arena as arena
import pravah.risk.markov as markov
import pravah.risk.prefix as prefix
import pravah.risk.shapley as shapley
import pravah.sim.analyze as sim_analyze
import pravah.sim.engine as engine
import pravah.sim.forecast as forecast
import pravah.stats.benford as benford
import pravah.stats.inference as inference
import pravah.stats.missingness as missingness
import pravah.stats.outliers as outliers
import pravah.stats.relationships as relationships
import pravah.survival.analyze as survival_analyze
import pravah.survival.cox as cox
import pravah.survival.events as events
import pravah.survival.km as km
import pravah.survival.logrank as logrank

_HEADINGS = ("WHAT:", "WHY:", "WHY NOT:", "ASSUMPTIONS:", "COMPLEXITY:", "FAILS WHEN:")

_FUNCTIONS = [
    proportion.wilson_interval,
    proportion.required_n_for_half_width,
    datetime_parse.parse_datetimes,
    topo.kahn_topological_sort,
    topo.resolve_precedence_order,
    reconstruct.pairwise_precedence,
    reconstruct.reconstruct_wide,
    shape.detect_shape,
    schema.detect_schema,
    validate.validate_event_log,
    anonymise.anonymise_frame,
    loader.load_table,
    date_format.resolve_date_formats,
    duplicates.levenshtein_distance,
    readiness.readiness_matrix,
    inference.chi2_sf,
    inference.benjamini_hochberg,
    inference.mann_whitney_p,
    missingness.missingness_report,
    outliers.modified_z_scores,
    relationships.spearman,
    relationships.cramers_v,
    relationships.correlation_ratio,
    relationships.mutual_information,
    benford.benford_report,
    times.find_end_column,
    times.build_visits,
    score.bootstrap_shares,
    littles.littles_check,
    resources.shrink_resources,
    trend.slowdown_index,
    trend.weekly_trend,
    rework.rework_table,
    analyze.analyze_bottlenecks,
    km.kaplan_meier,
    logrank.logrank,
    logrank.compare_segments,
    cox.fit_cox,
    events.prepare_cases,
    survival_analyze.analyze_survival,
    prefix.features_at,
    prefix.build_prefixes,
    markov.transition_counts,
    markov.absorption_probabilities,
    shapley.exact_shapley,
    arena.run_arena,
    risk_analyze.analyze_risk,
    effect.standardised_mean_difference,
    effect.fit_effect,
    refute.e_value,
    refute.refute,
    causes_analyze.candidate_treatments,
    causes_analyze.analyze_causes,
    engine.run_replication,
    forecast.seasonal_forecast,
    sim_analyze.analyze_simulation,
    knapsack.knapsack,
    plan_analyze.build_actions,
    plan_analyze.choose_plan,
]


def test_algorithm_docstrings_carry_the_required_headings() -> None:
    missing: list[str] = []
    for function in _FUNCTIONS:
        text = function.__doc__ or ""
        for heading in _HEADINGS:
            if heading not in text:
                missing.append(f"{function.__name__} missing {heading}")
    assert missing == []
