/* Coastline dashboard — Recommend (with embedded Queue/Admin) and Playground.
   Persistent Activity log mirrors every toast so failures are never lost. */
"use strict";

const $ = (id) => document.getElementById(id);
const val = (id) => $(id).value;

// fetch with a hard client-side timeout: a stalled backend (e.g. a slow ML
// predictor) must surface as an error toast, never an infinite spinner.
async function fetchT(url, opts = {}, timeoutMs = 60000) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    return await fetch(url, { ...opts, signal: ctl.signal });
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s — the selected predictor may be too slow for interactive use.`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/* ── formatting ───────────────────────────────────── */
const fmtThroughput = (v) => v == null ? "—" : Math.round(v).toLocaleString() + " tok/s";
function fmtRuntime(s) {
  if (s == null) return "—";
  if (s < 60) return s.toFixed(0) + " s";
  if (s < 3600) return (s / 60).toFixed(1) + " min";
  if (s < 86400) return (s / 3600).toFixed(1) + " h";
  return (s / 86400).toFixed(1) + " d";
}
function fmtEnergy(kwh) {
  if (kwh == null) return "—";
  return kwh < 1 ? (kwh * 1000).toFixed(0) + " Wh" : kwh.toFixed(2) + " kWh";
}
const fmtPower = (w) => w == null ? "—" : Math.round(w) + " W";

/* Arrival time: render epoch-style (> ~Sep 2001) as YYYY-MM-DD, HH:MM:SS;
   relative seconds (e.g. CSV-imported 0, 5, 10) stay as "N.N s". */
function fmtArrival(t) {
  if (t == null) return "—";
  if (t > 1e9) {
    const d = new Date(t * 1000);
    const p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}, ` +
           `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  }
  return Number(t).toFixed(1) + " s";
}
function fmtLayout(g, n) {
  n = n || 1; g = g || 1;
  return `${n} node${n > 1 ? "s" : ""} × ${g} GPU${g > 1 ? "s" : ""}`;
}
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ── Activity log (persistent) ────────────────────── */
function logActivity(level, message) {
  const list = $("activityList");
  if (!list) return;
  const empty = list.querySelector(".activity-empty");
  if (empty) empty.remove();
  const entry = document.createElement("div");
  entry.className = `activity-entry ${level}`;
  const time = new Date().toLocaleTimeString();
  entry.innerHTML =
    `<span class="time">${esc(time)}</span>` +
    `<span class="lvl">${esc(level)}</span>` +
    `<span class="msg">${esc(message)}</span>`;
  list.insertBefore(entry, list.firstChild);
  while (list.children.length > 50) list.lastChild?.remove();
}
function clearActivity() {
  const list = $("activityList");
  if (list) list.innerHTML = '<div class="activity-empty">No activity yet.</div>';
}

/* Transient banner (5s) + permanent record in the Activity log. */
function toast(message, type = "err") {
  const box = document.createElement("div");
  box.className = `toast ${type}`;
  box.textContent = message;
  $("toast").appendChild(box);
  setTimeout(() => { box.classList.add("out"); setTimeout(() => box.remove(), 220); }, 5000);
  logActivity(type, message);
}

/* state: prefix ∈ {rec, pg}; name ∈ {Empty, Loading, Data} */
function setState(prefix, name) {
  for (const s of ["Empty", "Loading", "Data"]) {
    const el = $(prefix + s);
    if (!el) continue;
    el.classList.toggle("show", s === name);
    if (s === name) el.removeAttribute("hidden"); else el.setAttribute("hidden", "");
  }
}

/* ── tabs ─────────────────────────────────────────── */
function initTabs() {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    const tab = t.dataset.tab;
    document.querySelectorAll(".tab-panel").forEach((p) => {
      p.toggleAttribute("hidden", p.id !== `panel-${tab}`);
    });
  }));
}

/* ── Recommend — hardware-mode segmented control ── */
function initHwToggle() {
  document.querySelectorAll("#hwMode .seg").forEach((seg) => seg.addEventListener("click", () => {
    document.querySelectorAll("#hwMode .seg").forEach((s) => s.classList.remove("active"));
    seg.classList.add("active");
    const mode = seg.dataset.mode;
    document.querySelectorAll('[data-hw="total"]').forEach((e) => e.toggleAttribute("hidden", mode !== "total"));
    document.querySelectorAll('[data-hw="nodes"]').forEach((e) => e.toggleAttribute("hidden", mode !== "nodes"));
  }));
}
const hwMode = () => document.querySelector("#hwMode .seg.active").dataset.mode;

const PRESET_NOTE = {
  balanced: "Balanced — weights runtime and power equally (α/β = 0.5 / 0.5).",
  performance: "Performance — favours throughput over power (α/β = 0.2 / 0.8).",
  energy: "Energy-saver — favours lower power over throughput (α/β = 0.8 / 0.2).",
};
function syncPolicy() {
  const isMultiObjective = val("rec_strategy") === "multi_objective";
  $("presetField").style.display = isMultiObjective ? "" : "none";
  $("policyNote").textContent = isMultiObjective
    ? PRESET_NOTE[val("rec_preset")]
    : "Minimum GPUs — fewest total GPUs that remain feasible for the workload.";
}

function recPayload() {
  return {
    llm_model: val("rec_llm_model"),
    fine_tuning_method: val("rec_method"),
    gpu_model: val("rec_gpu_model"),
    tokens_per_sample: parseInt(val("rec_tokens"), 10),
    batch_size: parseInt(val("rec_batch"), 10),
    training_epochs: parseInt(val("rec_epochs"), 10),
    dataset_size: parseInt(val("rec_dataset"), 10),
    hardware_mode: hwMode(),
    total_gpus: parseInt(val("rec_total_gpus"), 10),
    num_nodes: parseInt(val("rec_num_nodes"), 10),
    gpus_per_node: parseInt(val("rec_gpus_per_node"), 10),
    prediction_model: val("rec_model"),
    strategy: val("rec_strategy"),
    preset: val("rec_preset"),
  };
}

let _lastRec = null;  // cached so the Schedule button on each row can build a queue payload

async function submitRec(ev) {
  ev.preventDefault();

  const _totalGpus = parseInt(val("rec_total_gpus"), 10);
  const _numNodes = parseInt(val("rec_num_nodes"), 10);
  const _gpn = parseInt(val("rec_gpus_per_node"), 10);
  const _maxTotal = parseInt($("rec_total_gpus").max, 10) || Infinity;
  const _maxNodes = parseInt($("rec_num_nodes").max, 10) || Infinity;
  const _maxGpn = parseInt($("rec_gpus_per_node").max, 10) || Infinity;
  if (hwMode() === "total" && _totalGpus > _maxTotal) {
    return toast(`Requested ${_totalGpus} GPUs exceeds the cluster cap of ${_maxTotal}.`);
  }
  if (hwMode() === "nodes") {
    if (_gpn > _maxGpn) return toast(`Requested ${_gpn} GPUs/node exceeds cluster max ${_maxGpn}.`);
    if (_numNodes > _maxNodes) return toast(`Requested ${_numNodes} nodes exceeds cluster max ${_maxNodes}.`);
    if (_gpn * _numNodes > _maxTotal) return toast(`Requested ${_gpn * _numNodes} GPUs exceeds the cluster cap of ${_maxTotal}.`);
  }

  const epochs = parseInt(val("rec_epochs"), 10), dataset = parseInt(val("rec_dataset"), 10);
  if (!epochs || epochs < 1) return toast("Training epochs must be at least 1.");
  if (!dataset || dataset < 1) return toast("Dataset size must be at least 1.");

  setState("rec", "Loading");
  $("recBtn").disabled = true;
  logActivity("info", `POST /api/recommend (${val("rec_llm_model")} · ${val("rec_method")} · ${val("rec_model")})`);
  try {
    const resp = await fetchT("/api/recommend", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(recPayload()),
    }, 60000);
    const data = await resp.json();
    if (resp.status === 404) {
      setState("rec", "Empty");
      toast(`No feasible configuration — the selected model (${val("rec_model")}) may not cover this workload yet.`, "info");
      return;
    }
    if (!resp.ok || !data.success) throw new Error(data.detail || data.error || "Recommendation failed");
    _lastRec = data;
    renderRec(data);
    setState("rec", "Data");
    logActivity("ok", `Recommendation OK — ${(data.candidates || []).length} candidates`);
  } catch (err) {
    setState("rec", "Empty");
    toast(err.message || "Something went wrong.");
  } finally {
    $("recBtn").disabled = false;
  }
}

function renderRec(data) {
  const ws = data.workload_summary || {};
  const label = data.strategy === "min_gpu"
    ? "Minimum GPUs"
    : `Multi-objective · ${(data.preset || "balanced").replace(/^\w/, (c) => c.toUpperCase())}`;
  const cands = data.candidates || [];
  $("recSub").textContent = `${ws.llm_model} · ${ws.fine_tuning_method} · ${ws.gpu_model}`;
  $("recSummary").innerHTML = `
    <div class="item"><span class="k">Policy</span><span class="v">${esc(label)}</span></div>
    <div class="item"><span class="k">Dataset</span><span class="v mono">${Number(ws.dataset_size || 0).toLocaleString()} × ${esc(String(ws.training_epochs))} ep</span></div>
    <div class="item"><span class="k">Simulated</span><span class="v mono">${cands.length} configs</span></div>`;
  const tbody = $("recRows");
  if (!cands.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="dim" style="text-align:center;padding:28px">No configurations returned.</td></tr>`;
    return;
  }
  tbody.innerHTML = cands.map((c, i) => {
    const best = c.rank === 1;
    const nodes = c.number_of_nodes ?? c.workers;
    return `<tr class="${best ? "best" : ""}" style="animation-delay:${i * 45}ms">
      <td><span class="rank-num">${c.rank}</span>${best ? '<span class="tag-best">Best</span>' : ""}</td>
      <td><div class="layout-main">${esc(fmtLayout(c.gpus_per_node, nodes))}</div><div class="layout-sub">${c.total_gpus} GPUs total</div></td>
      <td class="r num">${c.batch_size ?? "—"}</td>
      <td class="r num">${fmtThroughput(c.predicted_throughput)}</td>
      <td class="r num">${fmtRuntime(c.predicted_runtime_seconds)}</td>
      <td class="r num">${fmtEnergy(c.energy_kwh)}</td>
      <td class="action"><button class="btn-mini schedule-btn" type="button" data-idx="${i}" title="Add this configuration to the workload queue">Schedule</button></td>
    </tr>`;
  }).join("");
  tbody.querySelectorAll(".schedule-btn").forEach((b) =>
    b.addEventListener("click", () => scheduleCandidate(parseInt(b.dataset.idx, 10)))
  );
}

async function scheduleCandidate(idx) {
  if (!_lastRec) return toast("Run the recommender first.");
  const cand = (_lastRec.candidates || [])[idx];
  if (!cand) return toast("Recommendation index not found.");
  const ws = _lastRec.workload_summary || {};
  if (!cand.predicted_runtime_seconds) {
    return toast("Cannot schedule: this candidate has no predicted runtime.");
  }
  const payload = {
    num_gpus: cand.total_gpus,
    predicted_duration_s: cand.predicted_runtime_seconds,
    llm_model: ws.llm_model,
    fine_tuning_method: ws.fine_tuning_method,
    gpu_model: ws.gpu_model,
    tokens_per_sample: ws.tokens_per_sample,
    batch_size: cand.batch_size,
    training_epochs: ws.training_epochs,
    gpus_per_node: cand.gpus_per_node,
    number_of_nodes: cand.number_of_nodes ?? cand.workers ?? 1,
  };
  try {
    const resp = await fetch("/api/queue", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) return toast(data.detail || "Could not schedule.");
    toast(`Scheduled #${cand.rank} as job ${data.job.request_id}`, "ok");
    refreshQueue();
  } catch (e) { toast("Schedule failed: " + e); }
}

/* ── Playground ───────────────────────────────────── */
function pgPayload() {
  const models = Array.from(document.querySelectorAll("#pgModels input:checked")).map((c) => c.value);
  return {
    llm_model: val("pg_llm_model"),
    fine_tuning_method: val("pg_method"),
    gpu_model: val("pg_gpu_model"),
    tokens_per_sample: parseInt(val("pg_tokens"), 10),
    batch_size: parseInt(val("pg_batch"), 10),
    gpus_per_node: parseInt(val("pg_gpus_per_node"), 10),
    number_of_nodes: parseInt(val("pg_nodes"), 10),
    dataset_size: parseInt(val("pg_dataset"), 10),
    training_epochs: parseInt(val("pg_epochs"), 10),
    models,
  };
}

async function submitPg(ev) {
  ev.preventDefault();
  const payload = pgPayload();
  if (!payload.models.length) return toast("Select at least one model to compare.");

  setState("pg", "Loading");
  $("pgBtn").disabled = true;
  logActivity("info", `POST /api/predict (${payload.models.length} model(s): ${payload.models.join(", ")})`);
  try {
    const resp = await fetchT("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }, 300000);
    const data = await resp.json();
    if (!resp.ok || !data.success) throw new Error(data.detail || data.error || "Prediction failed");
    renderPg(data);
    setState("pg", "Data");
    const avail = (data.results || []).filter((r) => r.available).length;
    logActivity("ok", `Playground OK — ${avail}/${(data.results || []).length} model(s) returned a prediction`);
  } catch (err) {
    setState("pg", "Empty");
    toast(err.message || "Something went wrong.");
  } finally {
    $("pgBtn").disabled = false;
  }
}

// Predictor family for colour-coding the Playground rows:
//   kavier → analytical (blue) · cache → retrieval/ground-truth (green) · everything else → data-driven ML (orange)
function pgFamily(modelId) {
  if (modelId === "kavier") return "kavier";
  if (modelId === "cache") return "cache";
  return "ml";
}

function renderPg(data) {
  const cfg = data.config || {};
  const results = data.results || [];
  $("pgSub").textContent = `${cfg.llm_model} · ${cfg.fine_tuning_method} · ${cfg.gpu_model}`;
  $("pgSummary").innerHTML = `
    <div class="item"><span class="k">Layout</span><span class="v">${esc(fmtLayout(cfg.gpus_per_node, cfg.number_of_nodes))}</span></div>
    <div class="item"><span class="k">Batch / Seq</span><span class="v mono">${cfg.batch_size} / ${cfg.tokens_per_sample}</span></div>
    <div class="item"><span class="k">Dataset</span><span class="v mono">${(cfg.dataset_size ?? 0).toLocaleString()} × ${cfg.training_epochs ?? "?"} ep</span></div>
    <div class="item"><span class="k">Models</span><span class="v mono">${results.length}</span></div>`;
  const tbody = $("pgRows");
  tbody.innerHTML = results.map((r, i) => {
    const fam = pgFamily(r.model);  // kavier → blue, cache → green, data-driven ML → orange
    if (!r.available) {
      return `<tr class="dim-row fam-${fam}" style="animation-delay:${i * 45}ms">
        <td>${esc(r.label)} <span class="muted">· not available yet</span></td>
        <td class="r num dim">—</td><td class="r num dim">—</td><td class="r num dim">—</td><td class="r num dim">—</td></tr>`;
    }
    return `<tr class="fam-${fam}" style="animation-delay:${i * 45}ms">
      <td class="config-cell">${esc(r.label)}</td>
      <td class="r num">${fmtThroughput(r.predicted_throughput)}</td>
      <td class="r num">${fmtRuntime(r.predicted_runtime_seconds)}</td>
      <td class="r num">${fmtPower(r.power_watts)}</td>
      <td class="r num">${fmtEnergy(r.energy_kwh)}</td>
    </tr>`;
  }).join("");
}

/* ── Queue & admin (embedded inside the Recommend tab) ─ */
async function refreshQueue() {
  try {
    const resp = await fetch("/api/queue");
    const data = await resp.json();
    renderQueue(data.jobs || []);
  } catch (e) { toast("Could not load queue: " + e); }
}

function renderQueue(jobs) {
  const n = jobs.length;
  $("queueCount").textContent = n === 0 ? "empty" : (n === 1 ? "1 job" : `${n} jobs`);
  const tbody = $("queueRows");
  if (!n) {
    tbody.innerHTML = `<tr class="queue-empty"><td colspan="9">Queue is empty — add a job above, or import a CSV in admin mode.</td></tr>`;
    return;
  }
  tbody.innerHTML = jobs.map((j) => `
    <tr>
      <td><code>${esc(j.request_id)}</code></td>
      <td>${esc(j.llm_model || "—")}</td>
      <td class="r num">${j.num_gpus}</td>
      <td class="r num">${j.batch_size ?? '<span class="dim">—</span>'}</td>
      <td class="r num">${j.training_epochs ?? '<span class="dim">—</span>'}</td>
      <td class="r num">${fmtRuntime(j.predicted_duration_s)}</td>
      <td class="r num">${j.predicted_power_watts_per_gpu ? fmtPower(j.predicted_power_watts_per_gpu) + "/GPU" : '<span class="dim">—</span>'}</td>
      <td class="r num">${fmtArrival(j.arrival_time)}</td>
      <td class="action"><button class="btn-mini danger icon queue-remove" data-id="${esc(j.request_id)}" type="button" title="Remove">×</button></td>
    </tr>`).join("");
  tbody.querySelectorAll(".queue-remove").forEach((b) =>
    b.addEventListener("click", () => removeQueueJob(b.dataset.id))
  );
}

async function submitQueueJob(ev) {
  ev.preventDefault();
  const num_gpus = parseInt(val("q_num_gpus"), 10);
  const epochs = parseInt(val("q_epochs"), 10);
  const dataset_size = parseInt(val("q_dataset"), 10);
  if (!num_gpus || num_gpus < 1) return toast("GPUs must be at least 1.");
  if (!epochs || epochs < 1) return toast("Epochs is required and must be at least 1.");
  if (!dataset_size || dataset_size < 1) return toast("Dataset (samples) is required and must be at least 1.");

  const payload = { num_gpus, training_epochs: epochs, dataset_size };
  const llm_model = val("q_model");           if (llm_model) payload.llm_model = llm_model;
  const fine_tuning_method = val("q_method"); if (fine_tuning_method) payload.fine_tuning_method = fine_tuning_method;
  const gpu_model = val("q_gpu_model");       if (gpu_model) payload.gpu_model = gpu_model;
  const tokens = parseInt(val("q_tokens"), 10);  if (tokens) payload.tokens_per_sample = tokens;
  const batch = parseInt(val("q_batch"), 10);    if (batch) payload.batch_size = batch;
  // Duration is an optional override: backend will Kavier-predict when the
  // workload config is complete, and fall back to this user value otherwise.
  const dur_raw = val("q_duration");
  const predicted_duration_s = dur_raw ? parseFloat(dur_raw) : null;
  if (predicted_duration_s !== null && predicted_duration_s > 0) {
    payload.predicted_duration_s = predicted_duration_s;
  }
  try {
    const resp = await fetch("/api/queue", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) return toast(data.detail || "Could not add job.");
    const src = data.duration_source === "kavier" ? " · Kavier-predicted duration" : " · user duration";
    toast(`Added job ${data.job.request_id}${src}`, "ok");
    refreshQueue();
  } catch (e) { toast("Add failed: " + e); }
}

async function removeQueueJob(id) {
  try {
    const resp = await fetch(`/api/queue/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (!resp.ok) return toast("Could not remove job.");
    refreshQueue();
  } catch (e) { toast("Remove failed: " + e); }
}

async function clearQueue() {
  if (!confirm("Clear ALL queued jobs?")) return;
  try {
    await fetch("/api/admin/clear", { method: "POST" });
    $("adminResults").innerHTML = "";
    refreshQueue();
    logActivity("info", "Queue cleared");
  } catch (e) { toast("Clear failed: " + e); }
}

function toggleAdmin() {
  $("adminPanel").toggleAttribute("hidden");
  $("adminToggle").classList.toggle("active");
}

/* ── Cluster timeline figure (GPUs allocated + queue depth over time) ────────
   The Exp2/Exp4 cluster plot, drawn from the FIFO run's step-series
   (data.timeline). Two stacked SVG strip-charts on a shared time axis with a
   crosshair scrubber (pointer + arrow keys). Presentation only — it reads the
   timeline payload and never re-fetches. */
const SVGNS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs) {
  const n = document.createElementNS(SVGNS, tag);
  if (attrs) for (const k in attrs) n.setAttribute(k, String(attrs[k]));
  return n;
}
/* Step-after staircase: value v[i] holds from t[i] until t[i+1]. */
function ctStepPath(ts, vs, xf, yf) {
  let d = `M ${xf(ts[0])} ${yf(vs[0])}`;
  for (let i = 1; i < ts.length; i++) {
    d += ` L ${xf(ts[i])} ${yf(vs[i - 1])} L ${xf(ts[i])} ${yf(vs[i])}`;
  }
  return d;
}
/* Largest index i with ts[i] <= t (the active step). */
function ctStepIndex(ts, t) {
  let lo = 0, hi = ts.length - 1, ans = 0;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (ts[m] <= t) { ans = m; lo = m + 1; } else hi = m - 1;
  }
  return ans;
}

function renderClusterTimeline(mount, tl) {
  if (!mount) return;
  const ts = (tl && tl.t) || [];
  // Need at least two breakpoints to draw a staircase; otherwise leave it blank.
  if (ts.length < 2) { mount.innerHTML = ""; return; }
  const gpus = tl.gpus_used, queue = tl.queue_depth;
  const cap = Math.max(tl.cluster_gpus || 1, 1);
  const span = tl.makespan_s || ts[ts.length - 1] || 1;
  const peakG = tl.peak_gpus || 0, peakQ = tl.peak_queue || 0;

  // viewBox geometry (CSS scales the SVG to the container width). Margins are
  // sized for the larger tick / axis-title fonts (see .ct-tick / .ct-axis-title
  // in coastline.css) so the bigger labels never clip — matching the CLI plot.
  const W = 820, ML = 60, MR = 16, MT = 22;
  const gpuH = 150, gap = 40, queueH = 78, xlabH = 48;
  const H = MT + gpuH + gap + queueH + xlabH;
  const plotW = W - ML - MR;
  const gT = MT, gB = MT + gpuH;                 // GPU chart top / bottom
  const qT = MT + gpuH + gap, qB = qT + queueH;  // queue chart top / bottom
  const yMaxQ = Math.max(peakQ, 1);

  const xf = (t) => ML + (span > 0 ? t / span : 0) * plotW;
  const ygf = (v) => gB - (v / cap) * gpuH;
  const yqf = (v) => qB - (v / yMaxQ) * queueH;

  const summary =
    `Peak ${peakG} of ${cap} GPUs in use · peak queue ${peakQ} ${peakQ === 1 ? "job" : "jobs"} · span ${fmtRuntime(span)}`;
  const valueText = (t, g, q) =>
    `At ${fmtRuntime(t)}: ${g} of ${cap} GPUs in use, ${q} ${q === 1 ? "job" : "jobs"} queued`;

  mount.innerHTML = `
    <figure class="cluster-fig">
      <div class="cluster-fig-head">
        <h4>Cluster timeline</h4>
        <div class="ct-legend" aria-hidden="true">
          <span class="it"><span class="ct-sw ct-sw-gpu"></span>GPUs in use</span>
          <span class="it"><span class="ct-sw ct-sw-q"></span>Jobs queued</span>
          <span class="it"><span class="ct-sw ct-sw-cap"></span>Capacity</span>
        </div>
      </div>
      <div class="cluster-plot">
        <div class="ct-readout" aria-hidden="true">
          <span class="ct-ro-t"></span>
          <span class="ct-ro-row"><span class="ct-sw ct-sw-gpu"></span><span class="ct-ro-g"></span></span>
          <span class="ct-ro-row"><span class="ct-sw ct-sw-q"></span><span class="ct-ro-q"></span></span>
        </div>
        <div class="ct-scrub" tabindex="0" role="slider"
             aria-label="Scrub the cluster timeline by time"
             aria-valuemin="0" aria-valuemax="${span}" aria-valuenow="0"
             aria-valuetext="${esc(valueText(0, gpus[0], queue[0]))}"></div>
      </div>
      <figcaption class="cluster-cap"><b>${esc(summary)}</b>. Hover or focus the chart and use the arrow keys to read GPUs in use and queue depth at any moment.</figcaption>
    </figure>`;

  const plot = mount.querySelector(".cluster-plot");
  const svg = svgEl("svg", {
    viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": summary, preserveAspectRatio: "xMidYMid meet",
  });

  // Vertical time gridlines + x tick labels (shared by both charts).
  for (const f of [0, 0.25, 0.5, 0.75, 1]) {
    const tv = f * span, x = xf(tv);
    svg.appendChild(svgEl("line", { class: "ct-grid", x1: x, y1: gT, x2: x, y2: qB }));
    const lbl = svgEl("text", { class: "ct-tick x", x, y: qB + 18 });
    lbl.textContent = fmtRuntime(tv);
    svg.appendChild(lbl);
  }
  // Horizontal gridlines + y ticks — GPU chart (0 / mid / capacity).
  for (const v of [...new Set([0, Math.round(cap / 2), cap])]) {
    const y = ygf(v);
    svg.appendChild(svgEl("line", { class: "ct-grid", x1: ML, y1: y, x2: W - MR, y2: y }));
    const lbl = svgEl("text", { class: "ct-tick y", x: ML - 8, y: y + 4 });
    lbl.textContent = v;
    svg.appendChild(lbl);
  }
  // Horizontal gridlines + y ticks — queue chart. Jobs are integers, so keep the
  // queue axis on whole numbers (no 2.5, 5.5, ...), mirroring the CLI plot's
  // MaxNLocator(integer=True). peak_queue is already an integer count; Math.round
  // + de-dupe guards any mid value (qMax / 2) that would otherwise land on a half.
  const qMax = Math.max(peakQ, 1);
  const qRaw = qMax <= 1 ? [0, 1] : qMax <= 4 ? [0, qMax] : [0, qMax / 2, qMax];
  const qticks = [...new Set(qRaw.map((v) => Math.round(v)))];
  for (const v of qticks) {
    const y = yqf(v);
    svg.appendChild(svgEl("line", { class: "ct-grid", x1: ML, y1: y, x2: W - MR, y2: y }));
    const lbl = svgEl("text", { class: "ct-tick y", x: ML - 8, y: y + 4 });
    lbl.textContent = v;
    svg.appendChild(lbl);
  }

  // Areas + step lines.
  const x0 = xf(ts[0]), xN = xf(ts[ts.length - 1]);
  const gpuStep = ctStepPath(ts, gpus, xf, ygf);
  svg.appendChild(svgEl("path", { class: "ct-area-gpu", d: `${gpuStep} L ${xN} ${gB} L ${x0} ${gB} Z` }));
  svg.appendChild(svgEl("path", { class: "ct-line-gpu", d: gpuStep }));
  const qStep = ctStepPath(ts, queue, xf, yqf);
  svg.appendChild(svgEl("path", { class: "ct-area-q", d: `${qStep} L ${xN} ${qB} L ${x0} ${qB} Z` }));
  svg.appendChild(svgEl("path", { class: "ct-line-q", d: qStep }));

  // Capacity ceiling (dashed) on the GPU chart.
  svg.appendChild(svgEl("line", { class: "ct-cap-rule", x1: ML, y1: ygf(cap), x2: W - MR, y2: ygf(cap) }));

  // Axis frame (left edges + baselines for both charts).
  for (const [x1, y1, x2, y2] of [
    [ML, gT, ML, gB], [ML, gB, W - MR, gB],
    [ML, qT, ML, qB], [ML, qB, W - MR, qB],
  ]) svg.appendChild(svgEl("line", { class: "ct-axis", x1, y1, x2, y2 }));

  // Axis titles.
  const title = (x, y, rot, text) => {
    const t = svgEl("text", { class: "ct-axis-title", x, y, "text-anchor": "middle" });
    if (rot) t.setAttribute("transform", `rotate(-90 ${x} ${y})`);
    t.textContent = text;
    return t;
  };
  svg.appendChild(title(ML - 46, (gT + gB) / 2, true, "GPUs"));
  svg.appendChild(title(ML - 46, (qT + qB) / 2, true, "Queue"));
  svg.appendChild(title(ML + plotW / 2, H - 8, false, "Time"));

  // Crosshair group (hidden until hover/focus via the .cursor-on class).
  const cursor = svgEl("g", { class: "ct-cursor" });
  const cline = svgEl("line", { class: "ct-cursor-line", x1: ML, y1: gT, x2: ML, y2: qB });
  const dotG = svgEl("circle", { class: "ct-dot ct-dot-gpu", r: 3.2, cx: ML, cy: ygf(gpus[0]) });
  const dotQ = svgEl("circle", { class: "ct-dot ct-dot-q", r: 3, cx: ML, cy: yqf(queue[0]) });
  cursor.appendChild(cline); cursor.appendChild(dotG); cursor.appendChild(dotQ);
  svg.appendChild(cursor);

  plot.insertBefore(svg, plot.firstChild);  // SVG behind the overlays

  // ── crosshair wiring ──
  const scrub = plot.querySelector(".ct-scrub");
  const readout = plot.querySelector(".ct-readout");
  const roT = readout.querySelector(".ct-ro-t");
  const roG = readout.querySelector(".ct-ro-g");
  const roQ = readout.querySelector(".ct-ro-q");

  // Place the scrub overlay over the plot columns only (exclude margins/labels).
  scrub.style.left = (ML / W * 100) + "%";
  scrub.style.width = (plotW / W * 100) + "%";
  scrub.style.top = (gT / H * 100) + "%";
  scrub.style.height = ((qB - gT) / H * 100) + "%";

  let curIdx = 0;
  function apply(t) {
    t = Math.max(0, Math.min(span, t));
    const i = ctStepIndex(ts, t); curIdx = i;
    const g = gpus[i], q = queue[i], x = xf(t);
    cline.setAttribute("x1", x); cline.setAttribute("x2", x);
    dotG.setAttribute("cx", x); dotG.setAttribute("cy", ygf(g));
    dotQ.setAttribute("cx", x); dotQ.setAttribute("cy", yqf(q));
    roT.textContent = fmtRuntime(t);
    roG.textContent = `${g} / ${cap} GPU`;
    roQ.textContent = `${q} queued`;
    const leftPct = x / W * 100;
    readout.style.left = leftPct + "%";
    readout.classList.toggle("flip", leftPct > 62);
    scrub.setAttribute("aria-valuenow", t.toFixed(3));
    scrub.setAttribute("aria-valuetext", valueText(t, g, q));
  }
  const showCursor = (on) => plot.classList.toggle("cursor-on", on);

  scrub.addEventListener("pointermove", (e) => {
    const r = scrub.getBoundingClientRect();
    const frac = r.width ? (e.clientX - r.left) / r.width : 0;
    apply(Math.max(0, Math.min(1, frac)) * span);
    showCursor(true);
  });
  scrub.addEventListener("pointerleave", () => { if (document.activeElement !== scrub) showCursor(false); });
  scrub.addEventListener("focus", () => { apply(ts[curIdx]); showCursor(true); });
  scrub.addEventListener("blur", () => showCursor(false));
  scrub.addEventListener("keydown", (e) => {
    let i = curIdx;
    if (e.key === "ArrowRight" || e.key === "ArrowUp") i = Math.min(ts.length - 1, i + 1);
    else if (e.key === "ArrowLeft" || e.key === "ArrowDown") i = Math.max(0, i - 1);
    else if (e.key === "Home") i = 0;
    else if (e.key === "End") i = ts.length - 1;
    else return;
    e.preventDefault();
    apply(ts[i]); showCursor(true);
  });

  apply(0);  // seed the readout values (cursor stays hidden until hover/focus)
}

async function runFifo() {
  $("adminResults").innerHTML = '<p class="note">Running…</p>';
  try {
    const resp = await fetchT("/api/admin/run", { method: "POST" }, 120000);
    const data = await resp.json();
    if (!resp.ok) return toast(data.detail || "Run failed.");
    if (!data.totals) {
      $("adminResults").innerHTML = '<p class="note">Queue is empty — add a job or import a CSV first.</p>';
      return;
    }
    const t = data.totals;
    const jobsHtml = (data.jobs || []).map((j) => `
      <tr>
        <td><code>${esc(j.request_id)}</code></td>
        <td>${esc(j.llm_model || "—")}</td>
        <td class="r num">${j.num_gpus}</td>
        <td class="r num">${j.batch_size ?? '<span class="dim">—</span>'}</td>
        <td class="r num">${j.training_epochs ?? '<span class="dim">—</span>'}</td>
        <td class="r num">${fmtRuntime(j.predicted_duration_s)}</td>
        <td class="r num">${fmtRuntime(j.wait_time_s)}</td>
        <td class="r num">${fmtRuntime(j.completion_time_s)}</td>
        <td class="r num">${fmtEnergy(j.energy_kwh)}</td>
      </tr>`).join("");
    $("adminResults").innerHTML = `
      <div class="admin-totals">
        <div class="stat"><span class="k">Cluster</span><span class="v">${t.cluster_gpus} GPUs</span></div>
        <div class="stat"><span class="k">Jobs</span><span class="v">${t.n_jobs}</span></div>
        <div class="stat"><span class="k">Total runtime</span><span class="v">${fmtRuntime(t.makespan_s)}</span></div>
        <div class="stat"><span class="k">Avg wait</span><span class="v">${fmtRuntime(t.avg_waiting_time_s)}</span></div>
        <div class="stat"><span class="k">Avg JCT</span><span class="v">${fmtRuntime(t.avg_job_completion_time_s)}</span></div>
        <div class="stat"><span class="k">Total energy</span><span class="v">${fmtEnergy(t.total_energy_kwh)}</span></div>
      </div>
      <div id="clusterFigMount"></div>
      <div class="table-wrap">
        <table class="rec-table">
          <thead><tr><th>ID</th><th>Model</th><th class="r">GPUs</th><th class="r">Batch</th><th class="r">Epochs</th><th class="r">Duration</th><th class="r">Wait</th><th class="r">JCT</th><th class="r">Energy</th></tr></thead>
          <tbody>${jobsHtml}</tbody>
        </table>
      </div>`;
    // Draw the cluster figure (GPUs allocated + queue depth over time) from the
    // run's step-series; it sits between the totals and the per-job table.
    renderClusterTimeline($("clusterFigMount"), data.timeline);
    logActivity("ok",
      `Admin · FIFO run OK — ${t.n_jobs} jobs · makespan ${t.makespan_s.toFixed(1)}s · energy ${t.total_energy_kwh.toFixed(3)} kWh`);
  } catch (e) { toast("Run failed: " + e); }
}

function importCsv(ev) {
  const file = ev.target.files && ev.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async () => {
    try {
      const resp = await fetch("/api/admin/import", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ csv: reader.result }),
      });
      const data = await resp.json();
      if (!resp.ok) return toast(data.detail || "Import failed.");
      toast(`Imported ${data.imported} job(s).`, "ok");
      refreshQueue();
    } catch (e) { toast("Import failed: " + e); }
  };
  reader.readAsText(file);
  ev.target.value = "";  // reset so re-picking the same file still fires "change"
}

/* ── Inline help hints ─────────────────────────────────────────────────────
   Build an accessible "i" trigger + tooltip from each [data-hint] label/legend.
   The CSS handles showing it (:hover / :focus-within, so mouse, keyboard, and
   tap all work); this only wires the markup + ARIA. A click on the dot must
   never submit the form or open the field it sits inside. Defensive throughout:
   a failure here must not break the rest of init. */
function initHints() {
  let seq = 0;
  document.querySelectorAll("[data-hint]").forEach((host) => {
    const text = host.getAttribute("data-hint");
    if (!text) return;
    const id = `hint-tip-${++seq}`;

    // Accessible name for the trigger: the field's own label text, minus the
    // unit chip and any markup we add.
    const probe = host.cloneNode(true);
    probe.querySelectorAll(".unit, .hint").forEach((x) => x.remove());
    const name = probe.textContent.trim() || "this field";

    const wrap = document.createElement("span");
    wrap.className = "hint";

    const dot = document.createElement("button");
    dot.type = "button";          // never submit the enclosing form
    dot.className = "hint-dot";
    dot.textContent = "i";
    dot.setAttribute("aria-label", `What to enter for ${name}`);
    dot.setAttribute("aria-describedby", id);
    // Clicking the dot must not also activate the label's control or submit.
    dot.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); });

    const pop = document.createElement("span");
    pop.className = "hint-pop";
    pop.id = id;
    pop.setAttribute("role", "tooltip");
    pop.textContent = text;

    wrap.appendChild(dot);
    wrap.appendChild(pop);
    // Place the hint as a SIBLING right after the label (still within the same
    // .field / .group), never inside the <label for=…>: nesting the "i" button
    // there would fold it into the field's accessible name and into the label's
    // own click target. afterend keeps the reading order label · (i) · …
    host.insertAdjacentElement("afterend", wrap);
    host.removeAttribute("data-hint");
  });
}

/* ── init ─────────────────────────────────────────── */
function bindClick(id, fn) {
  const el = $(id);
  if (el) el.addEventListener("click", fn);
  else logActivity("err", `init: missing element #${id}`);
}
function bindSubmit(id, fn) {
  const el = $(id);
  if (el) el.addEventListener("submit", fn);
  else logActivity("err", `init: missing form #${id}`);
}
function bindChange(id, fn) {
  const el = $(id);
  if (el) el.addEventListener("change", fn);
  else logActivity("err", `init: missing element #${id}`);
}

try {
  initHints();
  initTabs();
  initHwToggle();
  bindChange("rec_strategy", syncPolicy);
  bindChange("rec_preset", syncPolicy);
  bindSubmit("recForm", submitRec);
  bindSubmit("pgForm", submitPg);
  bindSubmit("queueForm", submitQueueJob);
  bindClick("queueRefresh", refreshQueue);
  bindClick("queueClear", clearQueue);
  bindClick("adminToggle", toggleAdmin);
  bindClick("adminRun", runFifo);
  bindChange("csvFile", importCsv);
  bindClick("activityClear", clearActivity);
  syncPolicy();
  setState("rec", "Empty");
  setState("pg", "Empty");
  refreshQueue();
  logActivity("info", "Coastline ready");
} catch (e) {
  logActivity("err", "init failed: " + (e && e.message ? e.message : e));
  console.error("Coastline init failed:", e);
}
