# Theory

Derivations for the algorithms that are actually in the code. Little's Law,
the resource shrinkage, Kaplan-Meier, inverse probability weights, and the
knapsack are below.

## Topological order

A directed graph has a topological order when its nodes can be listed so that
every edge `u -> v` has `u` before `v`. Such an order exists if and only if the
graph has no directed cycle. Proof sketch: if there is a cycle, some node on
the cycle would have to precede itself. If there is no cycle, some node has
in-degree 0 (otherwise a walk that always enters a positive-in-degree node
would eventually repeat and form a cycle). Delete it, and repeat.

Kahn's algorithm is that proof turned into a procedure. Maintain the in-degree
of every remaining node. Remove a node of in-degree 0, append it to the order,
and decrement its successors. Stop when no such node remains. The removed nodes
are a topological order of the acyclic part. The nodes left over are exactly
the ones that still participate in a cycle.

This implementation breaks ties by the original column order, one node at a
time. Taking a whole "ready" wave at once is a different order: after `A` is
removed, a newly unlocked node can be earlier in the column list than a node
that was already ready, and it should come next. One-at-a-time removal is what
makes "left to right on the sheet" a real rule.

Each of `V` removals scans `V` nodes and the edges are walked once, so the time
is `O(V^2 + E)`. `V` here is the number of date columns.

## Pairwise precedence

For date columns `A` and `B`, restrict to the `n` rows where both values parse.
Let `k` be the number of those rows with `A` strictly before `B`. The estimate
is `p = k / n`. Ties are in `n` and in neither direction, so
`P(A < B) + P(B < A) + P(equal) = 1`.

A mean date is the rejected alternative. One cell a year late moves the mean,
and a stage that is blank on the early cases looks early because the missing
rows never enter the average. The pairwise rate uses only rows that observed
both stages. The cost is bias of a different kind: the estimate is the order
conditional on co-occurrence. If only successful cases have an enrol date, the
order is the order of successful cases.

The edge is kept only when three things hold: `n >= 5`, `p >= 0.75`, and the
Wilson lower bound is above `0.5`. The first is a minimum vote count. The
second is an effect size: a stable 60/40 split is not a stage order. The third
is a certainty check: 4 out of 5 is a point estimate of 0.80 whose 95% interval
still contains a coin flip, so it does not become an edge. A significance test
is the wrong third check. With a large `n`, 52% is "significant" and still
operationally a tie.

## Wilson interval

For `k` successes in `n` trials, `p = k/n`, and `z = 1.96`,

```
centre = (p + z^2 / (2n)) / (1 + z^2 / n)
margin = z * sqrt( p(1-p)/n + z^2 / (4n^2) ) / (1 + z^2 / n)
```

The interval is `centre ± margin`, clipped to `[0, 1]`. Clipping only removes a
rounding overshoot at 0 and 1; a proportion cannot leave that range.

The centre is not `p`. It is pulled toward 1/2 by `z^2 / (2n)`, which is why
5 out of 5 does not get a lower bound of 1. With `z = 1.96` that lower bound is
about 0.57, just above a coin flip. 4 out of 5 lands near 0.38.

Wald's `p ± z sqrt(p(1-p)/n)` has width 0 when `p` is 0 or 1. That is the
small-sheet case we see most often, so Wald is not used for the reported
interval. It is used only as a planning formula, below, with `p` clipped away
from 0 and 1 so the plan does not come out as `n = 0`.

The interval treats rows as independent Bernoulli trials. Several events from
one case violate that. The interval is then too narrow. Phase 1 says so in the
failure note rather than pretending a cluster-robust interval it does not compute.

## How many rows

Worst-case proportion, half-width `d = 0.10`, `z = 1.96`:

```
n = z^2 * p * (1-p) / d^2
```

At `p = 0.5`, `n = 3.8416 * 0.25 / 0.01 = 96.04`. The spec's small-data floor
is 97, the next whole number of cases. A headline rate on fewer cases
than that is reported as a preview.

At the precedence boundary `p = 0.75`, `n = 3.8416 * 0.1875 / 0.01 = 72.03`,
rounded up to 73. A kept stage link with fewer co-observed rows than that is
labelled provisional. Variance inside the "keep" region is largest at 0.75,
so this is the cautious planning size for a link we are willing to draw, not
the size for a coin flip.

## Health score

Each component is `100 * (1 - bad_fraction)`. The score is

```
(25*missing + 25*unreadable_time + 20*backwards + 15*duplicates + 15*single_event) / 100
```

That is a convex combination: the weights are positive and sum to 100, so the
score stays on 0-100 and moves one point when a fully weighted component moves
one point. It is not a probability, and the five rates share rows, so summing
their Wilson variances would claim more precision than the file has. Each rate
carries its own interval. The score does not.

## Date format as a hypothesis

For a slash token `a/b/y` with `a <= 12` and `b <= 12`, both calendars are legal.
A token with `a > 12` is possible only as day/month, and `b > 12` only as month/day.
Those votes are evidence, not a prior.

When a sheet has no such vote, parse every date column both ways. For a candidate
parse, the consistency of columns in sheet order is the mean of `P(left < right)`
over pairs with at least 5 rows where both parsed. The winner is the calendar
with the higher mean. Confidence is `min(1, |dmy - mdy| / 0.3)`. A gap under
0.05 stores day/month at confidence 0.35. Using `max(P(A<B), P(B<A))` would
score a fully reversed calendar as perfect, so the direction is required.

## Modified z-score

Let `m` be the median and `MAD` the median of `|x - m|`. The modified z-score is
`0.6745 (x - m) / MAD`. The constant makes MAD comparable to a standard deviation
for a normal sample, because the MAD of a standard normal is about 0.6745.
The ordinary z-score uses the mean and the standard deviation, both of which
move when `x` itself is extreme. Flag `|z| > 3.5`. If `MAD = 0`, more than half
the points share one value and the score is undefined; the fence
`[Q1 - 1.5 IQR, Q3 + 1.5 IQR]` is used instead.

## Spearman, Cramér's V, eta

Spearman is the Pearson correlation of average ranks. A strictly increasing
curve therefore scores 1 even when the raw plot is not a line.

Cramér's V is `sqrt(chi^2 / (n * min(r - 1, c - 1)))`. Dividing by `n` is what
makes two sheets of different sizes comparable. The chi-square is withheld when
any expected count is under 5, and then V is withheld with it.

The correlation ratio is `sqrt(SS_between / SS_total)`. It is the share of
variance of a number explained by category means. The category is not ranked.

## Benford

The significand of a number that spans several orders of magnitude spends a
share `log10(1 + 1/d)` of the time with first digit `d`. The chi-square
statistic compares the nine observed counts with `n` times those shares, on
8 degrees of freedom. The upper tail uses the Wilson–Hilferty cube-root
normal approximation rather than the regularised gamma.

## Missingness and false discoveries

A missingness flag is dependent on an observed column when a valid chi-square
or Mann–Whitney test survives Benjamini–Hochberg at 0.05. Sort p-values
`p_(1) <= ... <= p_(m)` and reject through the largest `k` with
`p_(k) <= (k/m) * 0.05`. That controls the expected share of false leads.
It does not identify whether the missingness is MAR or MNAR.

## Bottleneck share

For case `c`, throughput time `T_c` is the last clock minus the first. With a finish column that is the last finish minus the first start. Waiting time `W_{c,a}` is the idle gap in front of activity `a`, summed over the visits in that case. The score is

`S_a = (sum_c W_{c,a}) / (sum_c T_c)`.

A single timestamp cannot separate idle time from work, so the gap between events is stored as waiting time and processing time is left empty. The shares then add to 1, because those gaps partition `T_c`.

The interval resamples cases with replacement, `B = 1000`, seed 42. `S_a` is recomputed on each sample. The reported interval is the 2.5 and 97.5 percentiles. It is not a normal approximation: identical cases produce a zero-width interval, which is the right answer.

## Little's Law

For one activity, the window runs from the first completed arrival to the last completed departure. `λ = N / T` and `W` is the mean completed time in the activity. `L = λ W`. The observed `L` is the time-average of the number of cases inside the activity.

On a window that contains every arrival and its departure, `sum of sojourns = integral of the level`, so `λ W` equals the time-average exactly. That identity cannot detect a queue that is still filling. The window is therefore split in half. If the two time-averages differ by more than 25% of the larger one, the activity is marked not in steady state.

In transition mode the time in the activity is the gap until the next event. That is the sojourn of the activity that has started. The bottleneck score charges the same gap to the next activity, as waiting time in front of it. The two numbers answer different questions and are not added together.

## Resource shrinkage

A resource median `m` from `n` events is replaced by

`n / (n + 5) * m + 5 / (n + 5) * m_all`

before it is ranked. `m_all` is the median of the event durations, not the median of the group medians. Five is the prior strength: a resource with one event keeps one sixth of its own median. Resources with fewer than 5 events are placed after every resource with at least 5, so a single slow case cannot head the list.

## One slowdown

Weekly median throughput time, dated on the Monday of the week the case ended. Binary segmentation tries one split with at least two weeks on each side. The cost is the residual sum of squares around a constant mean. The split is kept only when the later mean is higher and the cost falls by at least 20% against one mean for the whole series. The date named is the Monday of the first slower week.

## Kaplan-Meier

At a time with `d` events and `n` cases still at risk, survival is multiplied by `1 - d/n`. A censored case is in `n` until its time and is not in `d`. Greenwood's variance is the running sum of `d / (n (n - d))`. The 95% band uses the log-log transform with `z = 1.96`, so the band stays inside 0 and 1. The hazard window is the event time with the largest `d/n`, named from the previous event time up to that time.

## Cox sample floor

A Cox model is not fit below `MIN_COX_EVENTS` = 40 observed events (`pravah/ingest/constants.py`). The usual events-per-variable rule of thumb is about 10 observed events per covariate. With a handful of segment indicators that is 40 events before lifelines is asked to invert a Hessian. Below that floor the page names the minimum and shows stratified Kaplan–Meier curves instead of a hazard ratio.

## Prefixes and events per feature

A training row is a prefix that stops before the outcome. Its features are the clock, the count of events so far, the gap since the previous event, repeats already seen, the resource's past load, the day of week, and the latest activity. The label is whether the case later drops or runs past the training-fold 75th percentile of duration. A model is refused when finished prefix rows are fewer than 10 times the feature count.

Shapley values are the average marginal contribution of each feature over every subset, with the other features set to the training mean. For a linear score they equal the weight times the distance from that mean.

Split conformal remaining time takes the absolute residuals on a later time slice and the quantile `ceil((n+1) * 0.9) / n`. The interval is the prediction plus or minus that quantile.

## Inverse probability weights

The propensity score is a logistic regression of the treatment on the confounders the user kept. Rows outside 0.05 to 0.95 are trimmed. Stabilised weights are `p / e` for the treated and `(1-p) / (1-e)` for the rest, where `e` is the score and `p` is the treated share. The effect is the difference of weighted outcome means. Matching pairs on the logit score inside 0.2 standard deviations, without reuse. Balance is the standardised mean difference, and it passes under 0.1. The verdict is likely causal only when that balance holds, the interval excludes 0, a shuffled treatment's interval covers 0, a random extra cause keeps the sign, a 75% subset keeps the sign, and the E-value is at least 1.2.

## Discrete-event simulation

Arrivals are resampled inter-arrival times. Service times are resampled observed gaps. The next activity is drawn from the transition row. A heap is the clock. The baseline must come within 35% of the observed median throughput time and within 0.20 of every stage share. Otherwise no scenario number is shown. A passing scenario is 16 replications; the reported interval is the 5th and 95th percentiles.

Weekly arrivals are compared with a 4-week seasonal naive. Decomposition is used only when its holdout mean absolute error is lower.

## Knapsack

`best[c]` is the best value with budget `c`. Items are considered once, from the top of the budget down. Greedy by value/cost is computed beside it. Values 60, 100, 120 with costs 10, 20, 30 and budget 50 give greedy 160 and the program 220. The risk-adjusted option substitutes the lower end of each gain interval for the value.

