# Interview prep

Phase 1, ingestion. Each answer is what the code does, not a sketch of a
different design.

## 1. Why Kahn's algorithm, and why drop the weakest edge instead of solving the feedback arc set?

A wide sheet has no activity column. The date columns are the stages, and they
need one order before the sheet can become an event log. A topological order
exists exactly when the precedence graph has no cycle. Kahn's algorithm builds
it by repeatedly removing a node of in-degree 0. We remove the one that sits
furthest left on the sheet, one node at a time. Doing a whole ready-set at
once is a different order, because a node unlocked by the removal can outrank
a node that was already waiting. A depth-first sort is also linear, but the
tie-break hides in the recursion. A heap is the same rule with a log factor
we do not need for thirty columns.

Cycles happen because a few cells are backdated. Refusing the file throws away
the rest of the order. The minimum feedback arc set would delete the smallest
number of edges, and that can be a strong edge. We delete the edge with the
lowest precedence, then the lowest support, inside the cyclic component, and
we print it. The founder can put it back. This is wrong when the true link is
the weak one: several moderate errors can outvote one real reversal. The
report is the mitigation, not a claim that the greedy choice is optimal.

## 2. Why is stage order a pairwise rate with a 75% bar and a Wilson lower bound, not the mean date and not a p-value?

For columns A and B, `n` is the number of rows where both dates parse and
`k` is the number with A strictly before B. The estimate is `p = k / n`.
Ties sit in the denominator and vote for neither direction. An edge is kept
only when `n >= 5`, `p >= 0.75`, and the Wilson lower bound is above 0.5.

The mean date loses for two reasons. One late cell moves it. A stage that is
blank on the early rows looks early, because those rows never enter its
average. The pairwise rate ignores a row that did not observe both stages.
What it cannot fix is selection: if only converted students have an enrol
date, the order is the order of converted students. That conditional reading
is stated on the pair, not hidden.

A binomial test is the wrong gate. At a few thousand rows, 52% against 48% is
significant and still not a stage order. The 75% bar is an effect size. The
Wilson bound is the small-sample check the test does not provide: 4 out of 5
is a point estimate of 0.80 whose 95% interval still contains a coin flip
(lower bound about 0.38), so it does not become an edge. 5 out of 5 has a
lower bound near 0.57 and does. The Wald interval is not used for that
decision because its width is 0 when every row agrees, which is exactly the
five-row sheet. Wald is used only to plan the sample: at `p = 0.75` and a
±10 point half-width, `n = 1.96^2 * 0.75 * 0.25 / 0.10^2`, which rounds up to
73. A kept link below that is called provisional.

## 3. Why is data health a weighted sum with no interval of its own, and why is 100 the cutoff?

Five rates. Each component score is `100 * (1 - bad_fraction)`. The weights
are 25 for a missing case, activity, or time, 25 for a time that will not
parse, 20 for a consecutive pair whose time goes backwards, 15 for a repeated
case-activity-time, and 15 for a case with one event. They sum to 100, so the
headline number is already on a 0-100 scale. Time is half the score because
every later phase orders events by it. A one-event file can be perfectly clean
and still have no process to discover, so that failure has its own term, but
not a large one: some businesses really are one step, and the score says so
by dropping 15 points rather than failing the file.

The score is not given a confidence interval. The five rates share rows, so
adding their Wilson variances would be too tight. A bootstrap of the score is
the statistics phase, and inventing one here would be a second algorithm
smuggled into ingestion. Each rate does carry a 95% Wilson interval, and the
narrative says when the denominator is under 100.

The 100 is not a round habit. The worst-case Wald size for a ±10 point
interval is at `p = 0.5`: `1.96^2 * 0.25 / 0.10^2 = 96.04`, and the spec's
small-data floor is 100. A rarer rate is precise sooner. We still refuse to
call the file sufficient, because the next upload may not be rare. Backwards
time is counted in the order the rows are in. A file someone sorted for
display can hide a bad capture, and an unsorted export is penalised even when
the timestamps are true. On a rebuilt wide sheet the scored order is the stage
order, before the working copy is sorted by time, so a backdated cell remains
visible in the health score.

## 4. Why is `03/04/2026` a hypothesis test?

Both 3 April and 4 March are legal. I do not pick a country and move on.
If any value in the column has a first part above 12, that value cannot be a month, so the column is day/month. The mirror image is month/day. Confidence is the share of dates that actually cast that vote, so one decisive row and fifty still-ambiguous rows is a weak proof.

If nothing is above 12, I parse the whole sheet both ways and score each parse by how often a date column is earlier than the date column to its right. The format that produces more "enquiry before demo" rows wins. The confidence is the gap between those two scores. If they tie, I use day/month and I say the confidence is low. Treating the format as a hypothesis is what stops a silent month-first read from scrambling an Indian sheet whose days never reach 13.

## 5. Why not a z-score for fees?

A z-score is `(x - mean) / standard deviation`. The fee you are trying to catch is inside the mean and inside the standard deviation, so it pulls the yardstick toward itself and the next strange fee looks normal. The median does not move when one fee is huge, and the median absolute deviation does not either. The modified z-score is `0.6745 * (x - median) / MAD`, flagged past 3.5. The 0.6745 lines MAD up with a normal standard deviation. When more than half the fees are the same number, MAD is zero and there is nothing to divide by, so the fallback is the 1.5 IQR fence.

## 6. Why is "12% missing" not enough?

Twelve percent can be a random hole or it can be every Sanganer lead. I build a missing/present flag and test it against the other columns: chi-square for a category, Mann–Whitney for a number, and only when the counts are large enough for that test. Benjamini–Hochberg at 0.05 is there because I look at many columns and an uncorrected 0.05 would invent a pattern. If the flag lines up with an observed column, the data is not missing completely at random, and the average of the filled rows is biased. I do not then claim the mechanism is MAR rather than MNAR. That would need a reason that is not in the file.

## 7. Why rank a bottleneck by its share of throughput time, not by the mean wait?

Throughput time is the last clock on a case minus the first. The score for an activity is the sum of the waiting time in front of it, divided by the sum of throughput times. A mean wait of three days on a step two students reach is smaller, as a business problem, than two days on a step every student reaches. The share says which one eats the days. With a start and an end, waiting time is the idle gap and processing time is finish minus start, and only the waiting time enters the score. With one timestamp there is no service time in the file, so the gap is labelled transition time and the page says so. The interval resamples cases, not rows, 1000 times from seed 42, and takes the 2.5 and 97.5 percentiles. Rows in one case are not a sample.

## 8. Why does Little's Law need the two halves of the window?

`L = λ W`. `λ` is completed arrivals over the window, `W` is the mean time in the activity, and the observed `L` is the time-average number of cases inside it. If every arrival in the window also departs in the window, the sum of the sojourns equals the area under the work-in-progress, so the two sides match even when the queue doubled halfway through. That match is reported, and it is not used as the flag. The flag splits the window in half. A relative gap above 25% means the step is not in steady state, so the bottleneck share is a description of this file and not a claim about a stable capacity. In a one-timestamp log the gap until the next event is the sojourn of the activity that has started, while the bottleneck score charges that same gap to the next activity as waiting time. They are not added.

## 9. Why can't a tutor with one slow case top the resource list?

The displayed time is `n / (n + 5)` times the tutor's median plus `5 / (n + 5)` times the median of every event. Five is a prior strength, the same floor used before a proportion is trusted. A single observation keeps one sixth of its own number. Estimating a prior variance from the tutors would let that one slow case inflate the variance and then barely shrink, which is the failure the rule is there to stop. After shrinkage, any tutor with fewer than 5 events is placed below every tutor with at least 5. A genuinely slow new tutor waits for a fifth event before they can head the list.

## 10. Why is an open case censored instead of dropped?

The product-limit multiplies survival by `1 - deaths / at risk`. An open case stays in the risk set until its last observed day and is not a death. Dropping it pretends the file only contains people who already finished. Calling its last day a failure pretends the silence is the event. Greenwood's sum is `deaths / (at risk * (at risk - deaths))`, and the band is the log-log transform with 1.96, because `S ± 1.96 se` leaves the unit interval. Completion and dropout are different events: dropout is silence longer than the threshold the reader sets, and reaching the completion activity censors that dropout.

## 11. Why refuse one hazard ratio when the curves cross?

A Cox model is one number per covariate only while that number stays flat in time. Schoenfeld residuals, ranked in time, test that. If any covariate's p-value is under 0.05, the ratios are withheld and the page shows the stratified Kaplan-Meier curves. Reporting the ratio anyway would describe a gap that changed direction as if it had not. The log-rank comparisons against the largest level are corrected with Benjamini-Hochberg once more than one comparison is made.

## 12. Where is the hazard window?

At each event time the discrete hazard is deaths over the number still at risk. The window runs from the previous event time to the time with the largest hazard. The sentence is "Most drop-offs happen between day X and day Y" for a dropout definition, and the same shape for completions. It is a description of this curve, not a forecast.

## 13. Why can a prefix not include the finishing event?

The label is the future: the case later drops, or its duration exceeds the 75th percentile computed on the training cases only. The features are whatever a clerk could see at that prefix: time since start, events so far, time since the previous event, repeats already seen, that resource's load up to this clock, the day of week, and the latest activity. Putting the completion event in the row would make "last activity" the answer. The leakage test builds a case whose second step is named secret and asserts the training row never carries that name.

## 14. Why PR-AUC, and why refuse the event-log sample?

When the rarer class is under 40% of the training rows, ROC-AUC can look healthy while every dropout is missed. PR-AUC ranks the models in that case. Otherwise the winner is the higher ROC-AUC. The sentence on the page says which rule fired. The sample event log has 8 cases. Ten finished events are required per feature, and the feature count includes the one-hot activities, so the arena returns no model. A leaderboard on eight students would be a costume.

## 15. Why exact Shapley values, and what does the conformal interval claim?

With at most 12 features the sum over subsets is the definition, not an approximation. The background is the training mean. For a linear score the value of feature `i` is the weight times the distance from that mean, which the test checks. Permutation importance is the global question: how much the Brier score rises when a column is shuffled. Split conformal fits remaining time on the earliest cases, takes absolute residuals on the middle slice, and uses the quantile `ceil((n+1) * 0.9) / n`. The held-out coverage is reported next to the 90% target. It is not a confidence interval for a coefficient.

## 16. Why is "likely causal" a product of checks?

The propensity score is logistic in the confounders the reader left in the diagram. Scores outside 0.05 to 0.95 are trimmed. IPW and 1:1 matching are two estimators of the same contrast. Balance is the standardised mean difference after weighting, and it has to be under 0.1. A placebo shuffles the treatment and its interval must cover 0. A standard-normal extra column must not flip the sign or pull the interval over 0. A 75% subset must keep the sign. The E-value has to be at least 1.2, which is the strength an unseen confounder would need. Any failure, including an interval that already includes 0, is "association only" plus the name of the check.

## 17. What does the synthetic causal test actually know?

The confounder changes both the treatment probability and the outcome probability. The true risk difference is the average, over the same confounders, of the treated probability minus the untreated probability. The test asks the IPW estimate to land within 0.12 of that number, with an interval that contains the estimate, and with every confounder under 0.1 after weighting. The coaching sample is not used for this, because eight cases cannot identify an effect.

## 18. Why can a what-if number be refused?

The baseline resamples the observed inter-arrival times, the observed service gaps, and the transition matrix, with enough servers that the queue does not invent extra wait. It has to land within 35% of the observed median throughput time and within 0.20 of every stage share. If it does not, the scenario is not run and the action plan has no gain to price. A precise saving from a simulator that cannot reproduce the file is worse than a blank. Sixteen replications give the median and the 5th and 95th percentiles, which is a 90% interval across seeds, not a confidence interval for a mean.

## 19. When does the arrival forecast get to drive the simulation?

Weekly case starts are held out for the last four weeks. The seasonal naive repeats the count from four weeks earlier. The decomposition is a rolling mean plus the average leftover at each position in the cycle. It is shipped only when its mean absolute error is lower. Otherwise the page says the naive won and uses that forecast. Fewer than eight weeks skips both models and uses the mean, and says so.

## 20. Why is the knapsack not a ratio sort?

Values 60, 100 and 120 with costs 10, 20 and 30 inside a budget of 50: greedy by value per cost takes the first two and scores 160. The dynamic program takes the last two and scores 220. The program considers each item once, updating the budget from the top down so an item cannot be picked twice. The page shows both totals when they differ. The risk-adjusted switch replaces each gain with the lower end of its simulated interval before the program runs. An action whose simulator did not validate has a null gain and is not eligible.

## 21. Where do action sentences come from?

A bottleneck action exists only when a stage was ranked. A causal action exists only when the verdict was likely causal. A risk action quotes the open case with the highest probability. A rework action quotes the extra days already computed. The gain interval is the baseline median minus the scenario median, copied from the simulation. The PDF lines are those same sentences. There is no sentence for a saving the simulator did not produce.

