"""Plain-language glossary for every statistical term PRAVAH shows.

The page underlines a term only when it is in this dictionary. Later phases
add terms here instead of inventing a second definition in the interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GlossaryTerm:
    """One term: a founder sentence, a precise definition, why it is here, and an example."""

    plain: str
    technical: str
    why_used: str
    example: str
    module: str
    aliases: tuple[str, ...] = ()


def _term(
    plain: str,
    technical: str,
    why_used: str,
    example: str,
    module: str,
    aliases: tuple[str, ...] = (),
) -> GlossaryTerm:
    return GlossaryTerm(plain, technical, why_used, example, module, aliases)


# Insertion order is the Learn page order within each module.
GLOSSARY: dict[str, GlossaryTerm] = {
    "case": _term(
        "One person or file moving through the work, not one row of a spreadsheet.",
        "The entity id that groups events. Distinct case ids are the sample size for a rate.",
        "A rate about students is a rate about cases. Counting melted rows instead inflates the sample.",
        "At Diksha Coaching, student S14 is one case from enquiry through the fee.",
        "process",
        ("cases",),
    ),
    "event": _term(
        "One thing that happened to a case at one time.",
        "A row in the event log: case id, activity, timestamp, and optional resource.",
        "The process map is built from events. A wide sheet stores several events on one row.",
        "S14's demo on 12 March is one event. The fee paid on 2 April is another.",
        "process",
        ("events",),
    ),
    "activity": _term(
        "The name of a step, such as enquiry, demo, or fee.",
        "The label of what happened, shared by every case that took that step.",
        "Activities are the nodes of the process map. A date column on a wide sheet is an activity.",
        "enquiry, demo, joining, and fee are the four activities on the Diksha sheet.",
        "process",
        ("activities",),
    ),
    "variant": _term(
        "A path some cases take that is different from the usual order of steps.",
        "A distinct sequence of activities. Cases with the same sequence share a variant.",
        "A centre that skips the demo is a variant, not a data error, until the dates say otherwise.",
        "Most Diksha students go enquiry, demo, joining, fee. A few pay before joining.",
        "process",
        ("variants",),
    ),
    "directly-follows graph": _term(
        "A map of which step comes immediately after which other step.",
        "A directed graph. An arc A to B exists when B is the next activity after A in at least one case.",
        "It is the process map. It shows the paths people actually take, not the path in a policy document.",
        "If demo is the next step after enquiry for 20 of 24 students, that arc is the thick one.",
        "process",
    ),
    "throughput time": _term(
        "How long a case takes from the first step to the last.",
        "Timestamp of the last event minus timestamp of the first event, for one case.",
        "It is the number a founder means by 'how long does admission take'.",
        "S14 enquired on 1 March and paid on 2 April, so throughput time is 32 days.",
        "process",
    ),
    "waiting time": _term(
        "The idle gap between two steps, when nobody is doing that case's work.",
        "Start of the next event minus the end of the previous one. A single timestamp has no separate service time.",
        "A long wait between demo and joining is a different problem from a long demo.",
        "The demo ended on 12 March and joining was 28 March, so the student waited 16 days.",
        "process",
    ),
    "bottleneck": _term(
        "The step where work piles up and the rest of the process slows down.",
        "The activity with the largest queue or the longest wait in front of it.",
        "Fixing a step that is not the bottleneck does not shorten the admission.",
        "If 18 students are waiting on the demo and only 2 are waiting on the fee, the demo is the bottleneck.",
        "process",
        ("bottlenecks",),
    ),
    "Little's Law": _term(
        "The queue length equals the arrival rate times the time spent in the queue.",
        "L = λ W, for a stable system. L is the average number in the step, λ the arrival rate, W the average time there.",
        "It turns a count of waiting students into a waiting time, and the other way around, without a simulation.",
        "If 6 students sit in the demo queue and 2 arrive per week, the average wait is 3 weeks.",
        "process",
    ),
    "rework loop": _term(
        "A case goes back to a step it had already finished.",
        "A cycle in the directly-follows graph, or a repeated activity inside one case.",
        "Rework is lost time. A loop in the dates can also be a swapped cell, so both are reported.",
        "A student who attends a second demo after joining has a rework loop on demo.",
        "process",
    ),
    "edit distance": _term(
        "How many single-character edits turn one name into another.",
        "Levenshtein distance: insertions, deletions, and substitutions. A small distance is a likely typo.",
        "Rahul Sharma and Rahul Sharmaa should count as one person, not two resources.",
        "Sharma and Sharmaa differ by one added letter, so the distance is 1.",
        "process",
    ),
    "censoring": _term(
        "The case has not finished yet, so its total time is only known to be at least this long.",
        "Right-censoring: the observed time is a lower bound. The case is kept in the risk set and is not called a failure.",
        "Dropping students who have not paid yet makes the admission look faster than it is.",
        "S20 enquired 10 days ago and has no fee. The time is at least 10 days, not a completed 10-day admission.",
        "process",
    ),
    "Kaplan-Meier": _term(
        "A curve of how many cases are still unfinished as time passes, keeping the ones that have not ended.",
        "The product-limit estimator. At each observed finish time, survival drops by the finishes divided by the cases still at risk.",
        "It answers 'what share of students are still not enrolled after 30 days' without pretending open cases failed.",
        "Of 24 Diksha students, 20 have paid by day 30 and 4 are still open. Those 4 stay in the curve.",
        "survival",
    ),
    "hazard": _term(
        "The chance a case finishes in the next small slice of time, given it has not finished yet.",
        "h(t) = f(t) / S(t), the event rate among cases still at risk at time t.",
        "A high hazard just after the demo means that is when families decide, not a month later.",
        "If most fees are paid within a week of joining, the hazard peaks there.",
        "survival",
    ),
    "Markov chain": _term(
        "A model that predicts the next step using only the step the case is on now.",
        "P(next | history) = P(next | current activity). The rows of the transition matrix sum to 1.",
        "It is the simplest forecast of where a student goes next. If it fits badly, memory matters and the map must keep more history.",
        "Given a student is on the demo, the chain gives the chance the next step is joining rather than a second demo.",
        "prediction",
    ),
    "calibration": _term(
        "When the model says 70 percent, about 70 percent of those cases should actually happen.",
        "A reliability check: bin predicted probabilities and compare each bin's average prediction with its observed rate.",
        "A sharp score that is over-confident is worse than a dull one a founder can trust.",
        "If 10 students are each given a 0.8 chance of paying and only 4 pay, the model is not calibrated.",
        "prediction",
    ),
    "conformal interval": _term(
        "A range that covers the true outcome for a chosen share of future cases, without assuming a bell curve.",
        "Split conformal prediction builds the interval from the residuals of a held-out set so that coverage is at least 1 − α.",
        "It tells a founder the uncertainty of one student's forecast, not only the average error.",
        "A 90 percent conformal interval for days-to-fee of 12 to 40 days should contain the truth for about 9 of 10 new students.",
        "prediction",
    ),
    "SHAP value": _term(
        "How much one fact pushed this case's prediction up or down, averaged over the order of the facts.",
        "The Shapley value of a feature for one prediction. Features share the difference between this prediction and the average prediction.",
        "It answers 'why this student' without treating a correlated column as the cause.",
        "For S14, a long gap after the demo may account for most of a high dropout score. That is an attribution, not a cause.",
        "prediction",
        ("SHAP",),
    ),
    "confounder": _term(
        "A third fact that influences both the supposed cause and the outcome, so the raw comparison misleads.",
        "A variable that is a common cause of the treatment and the outcome. Conditioning on it blocks that back-door path.",
        "Centres differ in both fee discounts and follow-up speed. The centre can imitate an effect of the discount.",
        "Mansarovar gives more discounts and also calls faster. The discount is not shown to cause the faster joining until the centre is held fixed.",
        "causes",
        ("confounders",),
    ),
    "propensity score": _term(
        "The chance a case received the treatment, estimated from the facts known beforehand.",
        "e(x) = P(treatment = 1 | covariates x). Cases with similar scores are compared with each other.",
        "It lets a small sheet compare like with like when a randomised trial was never run.",
        "Two students with the same chance of being offered a discount, one offered it and one not, are a fairer pair than the raw groups.",
        "causes",
    ),
    "IPW": _term(
        "Inverse probability weighting: give more voice to cases who rarely received the treatment they got.",
        "Each case is weighted by 1 / e(x) if treated and 1 / (1 − e(x)) if not. The weighted difference estimates the treatment contrast.",
        "It is the estimator PRAVAH will show beside the raw gap, so a founder sees both.",
        "A student who was unlikely to get a discount, and did, counts more than one who was always going to get it.",
        "causes",
        ("inverse probability weighting",),
    ),
    "refutation test": _term(
        "A check that a claimed cause should fail if the claim is only a coincidence.",
        "A placebo treatment, a random subset, or an added noise confounder. The estimate should collapse toward zero.",
        "A cause that survives only the flattering cut of the data is not ready for an action plan.",
        "If shuffling the discount column still 'explains' joining, the original claim was a coincidence.",
        "causes",
    ),
    "discrete-event simulation": _term(
        "A replay of the process one event at a time, so a change can be tried before it is real.",
        "Cases arrive, wait, and move according to the measured times. The clock jumps from event to event.",
        "It is the what-if. Adding a demo slot on Saturday is tried on the replay, not on next month's students.",
        "Replaying Diksha with one extra demo seat shows whether the joining queue shrinks or the fee step becomes the new wait.",
        "planning",
    ),
    "knapsack": _term(
        "Choose a limited set of actions that buys the most improvement for the time and money you have.",
        "Maximise the sum of values subject to a weight budget. Here the weight is staff hours and the value is cases unblocked.",
        "The action plan cannot recommend everything. The knapsack keeps the few actions that fit the week.",
        "Calling the 11 stalled demos may beat rewriting the fee policy if the counsellor has only six hours.",
        "planning",
    ),
    "MCAR": _term(
        "Missing completely at random: the gaps do not line up with any column you can see.",
        "P(missing | observed columns) = P(missing). A rejected test means the gaps track some other column.",
        "An average of the filled fees is fair only when the blanks are not concentrated in one centre.",
        "If Sanganer leaves the demo date blank and Mansarovar does not, the demo date is not MCAR.",
        "quality",
        ("not MCAR", "likely MCAR"),
    ),
    "Cramér's V": _term(
        "A 0 to 1 measure of how strongly two categories move together.",
        "V = sqrt(χ² / (n · min(r − 1, c − 1))), from the chi-square statistic of the contingency table.",
        "Centre and batch are categories. A correlation of ranks would invent an order they do not have.",
        "If every Morning batch is in Mansarovar, Cramér's V between centre and batch is near 1.",
        "quality",
        ("Cramer's V",),
    ),
    "Spearman": _term(
        "A correlation of ranks. It sees a rise even when the relationship is not a straight line.",
        "Pearson correlation of the midranks. Ties take the average rank. The coefficient is between −1 and 1.",
        "Fee and delay can rise together without a straight line. Spearman still sees that.",
        "Higher fees sitting next to longer waits, on only five products, is a Spearman number with very low confidence.",
        "quality",
    ),
    "correlation ratio": _term(
        "How much of a number's spread is explained by which category it sits in.",
        "η² = SS_between / SS_total. It does not rank the categories.",
        "A fee that depends on the centre is a number split by a category, not two lists of ranks.",
        "If Mansarovar fees sit far above Sanganer fees, the correlation ratio of fee by centre is high.",
        "quality",
    ),
    "mutual information": _term(
        "How many bits one column tells you about another, with no straight-line assumption.",
        "The discrete mutual information of the joint histogram, in bits. Zero means the columns are independent.",
        "It is the fallback when Spearman, Cramér's V, or the correlation ratio cannot be computed.",
        "If knowing the centre tells you nothing about the batch, the mutual information is about zero.",
        "quality",
    ),
    "Benford's Law": _term(
        "In many naturally varying amounts, the first digits 1, 2, and 3 are more common than 8 or 9.",
        "P(first digit = d) = log10(1 + 1/d) for d from 1 to 9. A chi-square compares the observed counts.",
        "It is a screen for invented amounts. It is skipped when there are too few values, too few distinct amounts, or too little range.",
        "A column of real course fees should have more 1s than 9s. A column of three prices should not be tested.",
        "quality",
        ("Benford", "Benford's law"),
    ),
    "modified z-score": _term(
        "How far a number sits from the middle, using the median so one huge fee does not move the yardstick.",
        "The Iglewicz–Hoaglin score: 0.6745 · (x − median) / MAD. A value beyond 3.5 is flagged.",
        "The fee of 250000 should not redefine what a normal fee is. The median stays with the ordinary fees.",
        "Fees of 5000, 6500, 8000 and one fee of 250000: only 250000 is far from the median.",
        "quality",
    ),
    "Wilson interval": _term(
        "A confidence interval for a share that stays honest when the sample is small or perfect.",
        "The 95% Wilson score interval for a binomial proportion, using z = 1.96.",
        "Zero bad rows in five rows is not proof the rate is zero. Wilson still leaves room above zero.",
        "0 bad dates out of 5 still has a Wilson interval that reaches about 43 percent, so the sheet stays a preview.",
        "quality",
        ("Wilson",),
    ),
    "confidence interval": _term(
        "A range that would cover the true rate in about 95 of 100 repeated samples of this size.",
        "PRAVAH uses the Wilson score interval at z = 1.96 for proportions.",
        "A single percentage with no range looks more certain than the sheet can support.",
        "1 backwards pair out of 60 is a small share, and the interval says how small it might really be.",
        "quality",
    ),
    "credible interval": _term(
        "A range that holds the true value with a chosen probability, after seeing this sheet.",
        "A Bayesian interval from a stated prior and the likelihood. It is not a long-run coverage claim.",
        "Later phases can state uncertainty as a belief about this process, not only as a repeated-sample range.",
        "A 90 percent credible interval for the demo-to-joining wait is a claim about Diksha's wait, given this sheet and the prior.",
        "quality",
    ),
    "p-value": _term(
        "How often a result at least this extreme would appear if there were really no pattern.",
        "The tail probability of the test statistic under the null hypothesis. It is not the chance the null is true.",
        "A tiny p-value on a huge sheet can crown a difference too small to act on. The size of the gap is reported beside it.",
        "A p-value of 0.04 on a fee gap of 50 rupees is a detectable gap and a useless one.",
        "quality",
    ),
    "Benjamini-Hochberg": _term(
        "A rule for many tests at once, so a few lucky 'finds' are not treated as real.",
        "Sort the p-values and keep those with p_(k) ≤ (k/m) · q. It controls the false discovery rate at level q.",
        "Checking every column against every other column would otherwise declare noise as a pattern.",
        "Twenty missingness tests at 0.05 would expect one false alarm. Benjamini-Hochberg keeps the ones that still stand.",
        "quality",
        ("Benjamini–Hochberg",),
    ),
    "PSI": _term(
        "Population stability index: how far a column's mix has moved from the mix you trained on.",
        "Sum over bins of (p − q) · ln(p / q), where p is the new share and q the reference share.",
        "A model fit on last term's centres should be questioned when this term's mix has shifted.",
        "If Sanganer was 20 percent of students and is now 60 percent, PSI says the old model is off its base.",
        "quality",
        ("population stability index",),
    ),
    "sample adequacy": _term(
        "Whether there are enough cases for a rate to sit within about ±10 points.",
        "n = z² · p(1 − p) / E² with z = 1.96, worst-case p = 0.5, and E = 0.10, which gives n = 97.",
        "Readiness scores the structure of the log. Sample adequacy is a separate question and does not change that score.",
        "Diksha's 24 cases against 97 required cases is a preview. The readiness score can still be high.",
        "quality",
    ),
}

MODULES: tuple[tuple[str, str], ...] = (
    ("process", "Process"),
    ("survival", "Survival"),
    ("prediction", "Prediction"),
    ("causes", "Causes"),
    ("planning", "Planning"),
    ("quality", "Data quality"),
)


def glossary_payload() -> list[dict[str, object]]:
    """JSON rows for the page and the Learn list, in module order."""
    order = {name: index for index, (name, _label) in enumerate(MODULES)}
    labels = dict(MODULES)
    rows = []
    for phrase, term in GLOSSARY.items():
        rows.append(
            {
                "phrase": phrase,
                "plain": term.plain,
                "technical": term.technical,
                "why_used": term.why_used,
                "example": term.example,
                "module": term.module,
                "module_label": labels.get(term.module, term.module),
                "aliases": list(term.aliases),
            }
        )
    rows.sort(key=lambda row: (order.get(str(row["module"]), 99), str(row["phrase"]).lower()))
    return rows


def known_phrases() -> set[str]:
    """Every spelling the page is allowed to treat as a defined term."""
    phrases = set(GLOSSARY)
    for term in GLOSSARY.values():
        phrases.update(term.aliases)
    return phrases


def unglossed_mentions(text: str, watch: tuple[str, ...] | list[str]) -> list[str]:
    """Watch phrases that appear in ``text`` and are not defined in the glossary."""
    known = {phrase.lower() for phrase in known_phrases()}
    missing: list[str] = []
    for phrase in watch:
        pattern = re.compile(r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)")
        if pattern.search(text.lower()) and phrase.lower() not in known:
            missing.append(phrase)
    return missing
