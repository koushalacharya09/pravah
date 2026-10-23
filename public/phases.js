/* Later-phase panels. Helpers come from the page so this file does not own the DOM toolkit. */
function renderPhases(data, api) {
  const saved = window.phaseSettings || { survival: {}, modules: { causes: {}, simulation: {}, plan: {} } };
  document.getElementById("panel-survival").append(survivalCard(data.survival || {}, saved, api));
  document.getElementById("panel-risk").append(riskCard(data.risk || {}, api));
  document.getElementById("panel-causes").append(causesCard(data, saved, api));
  document.getElementById("panel-whatif").append(whatIfCard(data.simulation || {}, saved, api));
  document.getElementById("panel-plan").append(planCard(data.plan || {}, saved, api));
}

function onPhaseChange(node) {
  node.addEventListener("change", () => {
    if (!window.pravahRendering && window.pravahRerun) window.pravahRerun();
  });
}

function kmSvg(curve) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 320 150");
  svg.setAttribute("role", "img");
  svg.style.width = "100%";
  svg.style.maxWidth = "420px";
  svg.style.height = "160px";
  if (!curve || !curve.length) return svg;
  const maxT = Math.max(...curve.map((point) => point.time), 1);
  const x = (time) => 28 + (time / maxT) * 270;
  const y = (value) => 128 - value * 108;
  const upper = curve.map((point) => x(point.time) + "," + y(point.high)).join(" ");
  const lower = curve.slice().reverse().map((point) => x(point.time) + "," + y(point.low)).join(" ");
  const band = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
  band.setAttribute("points", upper + " " + lower);
  band.setAttribute("fill", "currentColor");
  band.setAttribute("opacity", "0.15");
  const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", "currentColor");
  line.setAttribute("stroke-width", "2");
  line.setAttribute("points", curve.map((point) => x(point.time) + "," + y(point.survival)).join(" "));
  svg.append(band, line);
  return svg;
}

function survivalCard(report, saved, api) {
  const el = api.el;
  const card = el("article", "", "card");
  card.append(el("h2", "Survival"));
  if (!report.applicable) {
    card.append(el("p", report.verdict || "Survival needs a case, an activity, and a time."));
    return card;
  }
  const event = document.createElement("select");
  event.dataset.survivalEvent = "1";
  event.append(new Option("Reaching completion", "completion"));
  event.append(new Option("Dropping out", "dropout"));
  event.value = (saved.survival && saved.survival.event) || report.event || "completion";
  onPhaseChange(event);
  const activity = document.createElement("select");
  activity.dataset.survivalActivity = "1";
  (report.activities || []).forEach((name) => activity.append(new Option(name, name)));
  activity.value = (saved.survival && saved.survival.completion_activity) || report.completion_activity || "";
  onPhaseChange(activity);
  const days = document.createElement("input");
  days.type = "number";
  days.min = "1";
  days.max = "365";
  days.dataset.dropoutDays = "1";
  days.value = String((saved.survival && saved.survival.dropout_days) || report.dropout_days || 14);
  onPhaseChange(days);
  const controls = el("div", "", "actions");
  controls.append(event, activity, days, el("span", "dropout days", "status"));
  card.append(controls);
  card.append(el("p", report.verdict, "band caution"));
  card.append(el("p", report.definition, "status"));
  card.append(el("p", report.censor_note, "status"));
  card.append(el("h2", "Kaplan-Meier curve"));
  card.append(kmSvg(report.curve || []));
  card.append(el("p", "The band is the 95% Greenwood confidence interval.", "status"));
  card.append(el("p", report.segment_note));
  (report.segments || []).forEach((segment) => {
    card.append(el("h2", segment.column));
    (segment.comparisons || []).forEach((row) => {
      const kept = row.rejected ? "kept after Benjamini-Hochberg" : "not kept";
      card.append(el("p", row.level + " versus " + row.reference + ": p=" + Number(row.p).toFixed(3) + ", " + kept));
    });
    (segment.curves || []).forEach((level) => {
      card.append(el("p", level.level, "status"));
      card.append(kmSvg(level.curve || []));
    });
  });
  card.append(el("h2", "Cox check"));
  card.append(el("p", (report.cox && report.cox.sentence) || ""));
  return card;
}

function riskCard(report, api) {
  const el = api.el;
  const card = el("article", "", "card");
  card.append(el("h2", "Risk"));
  card.append(el("p", report.verdict || "Risk prediction needs a case, an activity, and a time."));
  (report.markov || []).forEach((row) => card.append(el("p", row.sentence, "status")));
  if (!report.trained) return card;
  const arena = report.arena || {};
  card.append(el("p", arena.selection || ""));
  const table = document.createElement("table");
  const head = document.createElement("tr");
  ["Model", "ROC-AUC", "PR-AUC", "Brier"].forEach((label) => head.append(el("th", label)));
  table.append(head);
  (arena.models || []).forEach((row) => {
    const tr = document.createElement("tr");
    [row.model, row.roc_auc, row.pr_auc, row.brier].forEach((value) => tr.append(el("td", value === null || value === undefined ? "" : String(value))));
    table.append(tr);
  });
  card.append(table);
  card.append(el("h2", "Calibration"));
  card.append(calibrationSvg(arena.calibration || []));
  if (arena.conformal) card.append(el("p", arena.conformal.sentence));
  (report.cases || []).forEach((row) => {
    const block = el("div", "", "strip");
    block.append(el("p", row.case + "  probability " + row.probability));
    block.append(el("p", "Remaining time " + row.remaining_low + " to " + row.remaining_high + " days. The conformal interval is the held-out residual band.", "status"));
    (row.reasons || []).forEach((reason) => block.append(el("p", reason)));
    card.append(block);
  });
  return card;
}

function calibrationSvg(points) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 220 140");
  svg.style.width = "100%";
  svg.style.maxWidth = "280px";
  const guide = document.createElementNS("http://www.w3.org/2000/svg", "line");
  guide.setAttribute("x1", "20");
  guide.setAttribute("y1", "120");
  guide.setAttribute("x2", "200");
  guide.setAttribute("y2", "10");
  guide.setAttribute("stroke", "currentColor");
  guide.setAttribute("opacity", "0.35");
  svg.append(guide);
  points.forEach((point) => {
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("cx", String(20 + point.predicted * 180));
    dot.setAttribute("cy", String(120 - point.observed * 110));
    dot.setAttribute("r", "4");
    dot.setAttribute("fill", "currentColor");
    svg.append(dot);
  });
  return svg;
}

function causesCard(data, saved, api) {
  const el = api.el;
  const report = data.causes || {};
  const card = el("article", "", "card");
  card.append(el("h2", "Causes"));
  card.append(el("p", report.verdict || "Causes need a case, an activity, and a time."));
  const causes = (saved.modules && saved.modules.causes) || {};
  const treatment = document.createElement("select");
  treatment.dataset.causeTreatment = "1";
  treatment.append(new Option("Treatment", ""));
  const outcome = document.createElement("select");
  outcome.dataset.causeOutcome = "1";
  outcome.append(new Option("Outcome", ""));
  (report.candidates || []).forEach((item) => {
    const target = item.kind === "outcome" ? outcome : treatment;
    target.append(new Option(item.label, item.id));
  });
  if (causes.treatment) treatment.value = causes.treatment;
  else if (report.treatment) treatment.value = report.treatment;
  if (causes.outcome) outcome.value = causes.outcome;
  else if (report.outcome) outcome.value = report.outcome;
  onPhaseChange(treatment);
  onPhaseChange(outcome);
  const row = el("div", "", "actions");
  row.append(treatment, outcome);
  card.append(row);
  card.append(el("p", "Confounders assumed in the diagram. Untick one to take it out.", "status"));
  const chosen = new Set(causes.confounders || report.confounders || []);
  (data.columns || []).forEach((column) => {
    if (["case_id", "activity", "timestamp"].includes(column)) return;
    const label = el("label", " " + column);
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = column;
    box.dataset.confounder = "1";
    box.checked = chosen.has(column);
    onPhaseChange(box);
    label.prepend(box);
    card.append(label);
  });
  const estimate = report.estimate;
  if (estimate && estimate.estimated) {
    card.append(el("p", "IPW " + estimate.ipw + " (confidence interval " + estimate.ipw_low + " to " + estimate.ipw_high + "). Matching " + estimate.matching + "."));
    card.append(el("p", "Propensity score trimmed to 0.05–0.95. " + estimate.kept + " rows kept.", "status"));
    (estimate.balance || []).forEach((item) => {
      card.append(el("p", item.name + " standardised difference " + item.before + " before, " + item.after + " after."));
    });
  }
  const refutation = report.refutation;
  if (refutation) {
    (refutation.checks || []).forEach((check) => {
      card.append(el("p", (check.passed ? "Passed" : "Failed") + " refutation test: " + check.name + ". " + check.detail));
    });
  }
  return card;
}

function whatIfCard(report, saved, api) {
  const el = api.el;
  const card = el("article", "", "card");
  card.append(el("h2", "What-if"));
  card.append(el("p", report.verdict || "What-if needs a case, an activity, and a time."));
  if (!report.applicable) return card;
  const comparison = report.comparison || {};
  const badge = el("p", report.validated ? "Baseline validated" : "Baseline blocked", report.validated ? "band" : "band no");
  card.append(badge);
  card.append(el("p", "Observed median " + comparison.observed_median + " days. Simulated median " + comparison.simulated_median + ". Relative error " + comparison.relative_error + ". Stage-share gap " + comparison.share_error + ".", "status"));
  if (report.forecast) card.append(el("p", report.forecast.sentence));
  const sim = (saved.modules && saved.modules.simulation) || {};
  const activity = document.createElement("select");
  activity.dataset.simActivity = "1";
  (report.activities || []).forEach((name) => activity.append(new Option(name, name)));
  const illustrated = report.scenario || {};
  if (sim.activity) activity.value = sim.activity;
  else if (illustrated.activity) activity.value = illustrated.activity;
  const capacity = document.createElement("input");
  capacity.type = "number";
  capacity.min = "1";
  capacity.max = "32";
  capacity.dataset.simCapacity = "1";
  capacity.value = String(sim.capacity || 32);
  const wait = document.createElement("input");
  wait.type = "number";
  wait.min = "0";
  wait.max = "1";
  wait.step = "0.1";
  wait.dataset.simWait = "1";
  const waitDefault = illustrated.wait_scale === undefined ? 0.5 : illustrated.wait_scale;
  wait.value = String(sim.wait_scale === undefined ? waitDefault : sim.wait_scale);
  const volume = document.createElement("input");
  volume.type = "number";
  volume.min = "0.2";
  volume.max = "3";
  volume.step = "0.1";
  volume.dataset.simVolume = "1";
  volume.value = String(sim.volume || 1);
  const skip = document.createElement("select");
  skip.dataset.simSkip = "1";
  skip.append(new Option("Do not skip a step", ""));
  (report.activities || []).forEach((name) => skip.append(new Option("Skip " + name, name)));
  if (sim.skip) skip.value = sim.skip;
  [activity, capacity, wait, volume, skip].forEach(onPhaseChange);
  const controls = el("div", "", "actions");
  controls.append(activity, el("span", "capacity", "status"), capacity, el("span", "wait scale", "status"), wait, el("span", "arrival volume", "status"), volume, skip);
  card.append(controls);
  if (report.validated && report.scenario) {
    const scenario = report.scenario;
    card.append(el("p", "Scenario median " + scenario.median + " days. 90% interval " + scenario.low + " to " + scenario.high + ". Baseline " + scenario.baseline + ".", "band caution"));
  }
  return card;
}

function planCard(report, saved, api) {
  const el = api.el;
  const card = el("article", "", "card");
  card.append(el("h2", "Action plan"));
  const plan = (saved.modules && saved.modules.plan) || {};
  const budget = document.createElement("input");
  budget.type = "number";
  budget.min = "0";
  budget.dataset.planBudget = "1";
  budget.value = String(plan.budget || 3);
  const adjusted = document.createElement("input");
  adjusted.type = "checkbox";
  adjusted.dataset.planAdjusted = "1";
  adjusted.checked = Boolean(plan.risk_adjusted);
  onPhaseChange(budget);
  onPhaseChange(adjusted);
  const controls = el("div", "", "actions");
  controls.append(el("span", "Budget", "status"), budget, adjusted, el("span", "use the lower bound", "status"));
  card.append(controls);
  (report.actions || []).forEach((action) => {
    const cost = document.createElement("input");
    cost.type = "number";
    cost.min = "1";
    cost.dataset.planCost = action.id;
    const previous = plan.costs || {};
    cost.value = String(previous[action.id] || 1);
    onPhaseChange(cost);
    const row = el("div", "", "strip");
    row.append(el("p", action.text));
    row.append(el("span", "Cost", "status"), cost);
    const why = document.createElement("details");
    why.append(el("summary", "Why this action"));
    why.append(el("p", String(action.evidence || "")));
    why.append(el("p", "Gain " + action.gain + " (interval " + action.low + " to " + action.high + "). Confidence: " + action.confidence));
    row.append(why);
    card.append(row);
  });
  if (report.value !== null && report.value !== undefined) {
    card.append(el("p", "Knapsack value " + report.value + ". Interval of the chosen gains " + report.low + " to " + report.high + ".", "band caution"));
  }
  (report.lines || []).forEach((line) => card.append(el("p", line)));
  if (!(report.actions || []).length) card.append(el("p", report.lines && report.lines[0] ? report.lines[0] : "No action has a computed gain."));
  return card;
}
