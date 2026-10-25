// NNPM front end: map, reachability status, dialogs, runs, runtimes, terminal.
// Plain ES module, no build step. Talks to the Flask API under /api and /events.

const boot = window.NNPM_BOOT;
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `${response.status} ${response.statusText}`);
  }
  return data;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// --- Terminal --------------------------------------------------------------------------------

const terminal = {
  element: $("#terminal"),
  filter: "",
  append(line) {
    if (this.filter && !line.includes(`[${this.filter}]`)) return;
    const span = document.createElement("span");
    span.textContent = line + "\n";
    if (line.includes("[stderr]") || line.includes("[error]")) span.className = "err";
    else if (line.includes("[task]")) span.className = "task";
    else if (line.includes("[nnpm]")) span.className = "info";
    this.element.appendChild(span);
    if (this.element.childNodes.length > 4000) this.element.removeChild(this.element.firstChild);
    this.element.scrollTop = this.element.scrollHeight;
  },
  clear() {
    this.element.textContent = "";
  },
};
$("#btn-clear-terminal").addEventListener("click", () => terminal.clear());
$("#terminal-filter").addEventListener("input", (event) => {
  terminal.filter = event.target.value.trim();
});

// --- State and map -----------------------------------------------------------------------------

const state = {
  nodes: new Map(boot.nodes.map((n) => [n.id, n])),
  markers: new Map(),
  statuses: new Map(),
  links: [],
  addMode: false,
  currentRun: null,
  logSource: null,
  currentTask: null,
  taskSource: null,
  catalog: null,
};

const map = L.map("map", { zoomControl: true });
if (boot.offline) {
  $("#map").classList.add("offline");
} else {
  L.tileLayer(boot.tilesUrl, { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors" }).addTo(map);
}
const linkLayer = L.layerGroup().addTo(map);

function statusClass(nodeId) {
  const status = state.statuses.get(nodeId);
  return status ? status.status : "unknown";
}

function endDevices() {
  return Array.from(state.nodes.values()).filter((n) => n.role !== "Base Station");
}

function tooltipHtml(node) {
  const status = statusClass(node.id);
  return `<span class="dot ${status}"></span><strong>${escapeHtml(node.id)}</strong><br>${escapeHtml(node.ip)}`;
}

function upsertMarker(node) {
  let marker = state.markers.get(node.id);
  if (!marker) {
    marker = L.marker([node.lat, node.lon], { draggable: true, title: node.id });
    marker.addTo(map);
    marker.on("click", () => showNodeInfo(node.id));
    marker.on("dragend", async () => {
      const { lat, lng } = marker.getLatLng();
      try {
        await api(`/api/nodes/${encodeURIComponent(node.id)}`, { method: "PUT", body: { lat, lon: lng } });
        state.nodes.get(node.id).lat = lat;
        state.nodes.get(node.id).lon = lng;
        await refreshLinks();
      } catch (error) {
        terminal.append(`[nnpm][error] ${error.message}`);
      }
    });
    state.markers.set(node.id, marker);
  }
  marker.setLatLng([node.lat, node.lon]);
  marker.unbindTooltip();
  marker.bindTooltip(tooltipHtml(node), { permanent: true, direction: "top", className: `status-${statusClass(node.id)}` });
  if (node.role === "Base Station") {
    marker.getElement()?.classList.add("marker-edge");
  }
}

function fitMap() {
  const points = Array.from(state.nodes.values()).map((n) => [n.lat, n.lon]);
  if (points.length) map.fitBounds(points, { padding: [40, 40], maxZoom: 15 });
  else map.setView([37.9838, 23.7275], 13);
}

function renderMarkers() {
  for (const id of Array.from(state.markers.keys())) {
    if (!state.nodes.has(id)) {
      map.removeLayer(state.markers.get(id));
      state.markers.delete(id);
    }
  }
  state.nodes.forEach(upsertMarker);
}

async function refreshLinks() {
  linkLayer.clearLayers();
  if (!$("#chk-links").checked) return;
  const { links } = await api("/api/links");
  state.links = links;
  for (const link of links) {
    const line = L.polyline([link.from_coords, link.to_coords], { color: "#c0392b", weight: 2, opacity: 0.7 });
    line.bindTooltip(`${escapeHtml(link.from)} - ${escapeHtml(link.to)}<br>${(link.distance_m / 1000).toFixed(2)} km, ${link.latency_ms.toFixed(2)} ms`, { sticky: true });
    linkLayer.addLayer(line);
  }
}
$("#chk-links").addEventListener("change", refreshLinks);

async function reloadNodes() {
  const { nodes } = await api("/api/nodes");
  state.nodes = new Map(nodes.map((n) => [n.id, n]));
  renderMarkers();
  await refreshLinks();
}

async function showNodeInfo(nodeId) {
  const { node } = await api(`/api/nodes/${encodeURIComponent(nodeId)}`);
  const stats = node.statistics;
  const status = node.status ? `${node.status.status} (${node.status.received}/3 replies)` : "not probed yet";
  $("#node-info").innerHTML = `
    <p><strong>${escapeHtml(node.id)}</strong> (${escapeHtml(node.role)})</p>
    <p>Data address ${escapeHtml(node.ip)}<br>Management address ${escapeHtml(node.management_ip || node.ip)}<br>Reachability: ${status}</p>
    ${stats ? `<p>Last latency ${stats.average_latency_ms == null ? "n/a" : stats.average_latency_ms.toFixed(2) + " ms"}<br>
      Available memory ${stats.available_memory_mb.toFixed(0)} MB<br>CPU usage ${stats.cpu_usage_percent.toFixed(1)} %<br>
      <span class="muted">measured ${escapeHtml(stats.measured_at)}, run ${escapeHtml(stats.run_id)}</span></p>` : "<p class='muted'>No measurements yet: run the benchmarking pipeline.</p>"}
    <menu><button type="button" data-edit="${escapeHtml(node.id)}">Edit</button><button type="button" data-delete="${escapeHtml(node.id)}" class="danger">Delete</button></menu>`;
}
$("#node-info").addEventListener("click", async (event) => {
  const edit = event.target.dataset.edit;
  const del = event.target.dataset.delete;
  if (edit) openNodeForm(state.nodes.get(edit));
  if (del && confirm(`Delete node ${del}?`)) {
    await api(`/api/nodes/${encodeURIComponent(del)}`, { method: "DELETE" });
    await reloadNodes();
    $("#node-info").innerHTML = "<p class='muted'>Node deleted.</p>";
  }
});

// Add a node by clicking on the map: the coordinates are recorded in nodes.json on save.
$("#btn-add-mode").addEventListener("click", (event) => {
  state.addMode = !state.addMode;
  event.currentTarget.setAttribute("aria-pressed", String(state.addMode));
  $("#map").style.cursor = state.addMode ? "crosshair" : "";
});
map.on("click", (event) => {
  if (!state.addMode) return;
  openNodeForm({ id: "", role: "End Device", ip: "", management_ip: "", lat: event.latlng.lat, lon: event.latlng.lng });
});

// --- Reachability probes (server-sent events) -------------------------------------------------

function applyStatus(snapshot) {
  for (const [id, status] of Object.entries(snapshot.nodes || {})) state.statuses.set(id, status);
  state.nodes.forEach(upsertMarker);
}
function subscribeStatus() {
  const source = new EventSource("/events/status");
  source.addEventListener("status", (event) => applyStatus(JSON.parse(event.data)));
  source.onerror = () => {
    source.close();
    setTimeout(subscribeStatus, 5000);
  };
}

// --- Network dialog ------------------------------------------------------------------------------

function openNodeForm(node) {
  const form = $("#form-node");
  form.original_id.value = node.id || "";
  form.id.value = node.id || "";
  form.role.value = node.role || "End Device";
  form.ip.value = node.ip || "";
  form.management_ip.value = node.management_ip || "";
  form.lat.value = node.lat;
  form.lon.value = node.lon;
  renderNetworkTable();
  $("#dlg-network").showModal();
}
function renderNetworkTable() {
  const body = $("#network-table tbody");
  body.innerHTML = "";
  for (const node of state.nodes.values()) {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${escapeHtml(node.id)}</td><td>${escapeHtml(node.role)}</td><td>${escapeHtml(node.ip)}</td><td>${escapeHtml(node.management_ip || "")}</td>
      <td><button type="button" data-edit="${escapeHtml(node.id)}">Edit</button> <button type="button" class="danger" data-delete="${escapeHtml(node.id)}">Delete</button></td>`;
    body.appendChild(row);
  }
}
$("#network-table").addEventListener("click", async (event) => {
  const edit = event.target.dataset.edit;
  const del = event.target.dataset.delete;
  if (edit) openNodeForm(state.nodes.get(edit));
  if (del && confirm(`Delete node ${del}?`)) {
    await api(`/api/nodes/${encodeURIComponent(del)}`, { method: "DELETE" });
    await reloadNodes();
    renderNetworkTable();
  }
});
$("#btn-edit-network").addEventListener("click", () => {
  const first = state.nodes.values().next().value || { lat: 37.98, lon: 23.73 };
  openNodeForm({ id: "", role: "End Device", ip: "", lat: first.lat, lon: first.lon });
});
$("#form-node").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const payload = {
    id: form.id.value.trim(),
    role: form.role.value,
    ip: form.ip.value.trim(),
    management_ip: form.management_ip.value.trim() || null,
    lat: Number(form.lat.value),
    lon: Number(form.lon.value),
  };
  try {
    const original = form.original_id.value;
    if (original) await api(`/api/nodes/${encodeURIComponent(original)}`, { method: "PUT", body: payload });
    else await api("/api/nodes", { method: "POST", body: payload });
    await reloadNodes();
    renderNetworkTable();
    terminal.append(`[nnpm][info] node ${payload.id} saved to nodes.json`);
    $("#dlg-network").close();
  } catch (error) {
    alert(error.message);
  }
});

// --- Schema-driven forms -----------------------------------------------------------------------------
// A flat JSON Schema (string, number, integer, boolean, enum, array of strings) becomes a
// grid of fields; "x-nnpm" hints choose node widgets. Used for the runtime block of a
// scenario, the model parameters and the parameters of a runtime action.

function fieldName(prefix, key) {
  return `${prefix}__${key}`;
}

function schemaForm(schema, prefix, options = {}) {
  const properties = (schema && schema.properties) || {};
  const required = new Set((schema && schema.required) || []);
  const values = options.values || {};
  const parts = [];
  for (const [key, prop] of Object.entries(properties)) {
    const hint = (prop["x-nnpm"] || {}).widget;
    if (hint === "hidden") continue;
    const name = fieldName(prefix, key);
    const label = escapeHtml(key.replace(/_/g, " "));
    const help = prop.description ? ` title="${escapeHtml(prop.description)}"` : "";
    const current = values[key] !== undefined ? values[key] : prop.default;
    const req = required.has(key) ? " required" : "";
    if (hint === "node") {
      const choices = endDevices().map((n) => `<option value="${escapeHtml(n.id)}"${current === n.id ? " selected" : ""}>${escapeHtml(n.id)}</option>`).join("");
      parts.push(`<label${help}>${label} <select name="${name}"${req}>${required.has(key) ? "" : '<option value="">(none)</option>'}${choices}</select></label>`);
    } else if (hint === "nodes") {
      const chosen = new Set(Array.isArray(current) ? current : []);
      const boxes = endDevices().map((n) => `<label class="inline"><input type="checkbox" name="${name}" value="${escapeHtml(n.id)}"${chosen.has(n.id) ? " checked" : ""}> ${escapeHtml(n.id)}</label>`).join("");
      parts.push(`<div${help}><span class="help">${label}</span><div class="nodes">${boxes}</div></div>`);
    } else if (prop.enum) {
      const choices = prop.enum.map((v) => `<option value="${escapeHtml(v)}"${current === v ? " selected" : ""}>${escapeHtml(v)}</option>`).join("");
      parts.push(`<label${help}>${label} <select name="${name}">${choices}</select></label>`);
    } else if (prop.type === "boolean") {
      parts.push(`<label class="inline"${help}><input type="checkbox" name="${name}"${current ? " checked" : ""}> ${label}</label>`);
    } else if (prop.type === "integer" || prop.type === "number") {
      const step = prop.type === "integer" ? "1" : "any";
      const min = prop.minimum !== undefined ? ` min="${prop.minimum}"` : "";
      const max = prop.maximum !== undefined ? ` max="${prop.maximum}"` : "";
      parts.push(`<label${help}>${label} <input type="number" step="${step}"${min}${max} name="${name}" value="${current === undefined || current === null ? "" : current}"${req}></label>`);
    } else if (prop.type === "array") {
      const text = Array.isArray(current) ? current.join(", ") : "";
      parts.push(`<label${help}>${label} <input name="${name}" value="${escapeHtml(text)}" placeholder="comma separated"${req}></label>`);
    } else if (prop.type === "object") {
      parts.push(`<label${help}>${label} <input name="${name}" value="${escapeHtml(current ? JSON.stringify(current) : "")}" placeholder="JSON object"></label>`);
    } else {
      parts.push(`<label${help}>${label} <input name="${name}" value="${escapeHtml(current === undefined || current === null ? "" : current)}"${req}></label>`);
    }
  }
  return parts.length ? `<div class="grid">${parts.join("")}</div>` : "";
}

function readSchemaForm(form, schema, prefix) {
  const properties = (schema && schema.properties) || {};
  const out = {};
  for (const [key, prop] of Object.entries(properties)) {
    const hint = (prop["x-nnpm"] || {}).widget;
    if (hint === "hidden") continue;
    const name = fieldName(prefix, key);
    if (hint === "nodes") {
      const chosen = $$(`input[name="${name}"]:checked`, form).map((i) => i.value);
      if (chosen.length) out[key] = chosen;
      continue;
    }
    const field = form.elements[name];
    if (!field) continue;
    if (prop.type === "boolean") {
      out[key] = field.checked;
      continue;
    }
    const text = (field.value || "").trim();
    if (text === "") continue;
    if (prop.type === "integer") out[key] = parseInt(text, 10);
    else if (prop.type === "number") out[key] = Number(text);
    else if (prop.type === "array") out[key] = text.split(",").map((v) => v.trim()).filter(Boolean).map((v) => (prop.items && prop.items.type !== "string" && !Number.isNaN(Number(v)) ? Number(v) : v));
    else if (prop.type === "object") {
      try {
        out[key] = JSON.parse(text);
      } catch {
        throw new Error(`${key}: not a JSON object`);
      }
    } else out[key] = text;
  }
  return out;
}

// --- Scenario dialog ----------------------------------------------------------------------------------

function nodeConfigHtml(node) {
  const id = escapeHtml(node.id);
  return `<fieldset class="node-config" data-node="${id}">
    <h4><label class="inline"><input type="checkbox" name="use_${id}" checked> ${id} (${escapeHtml(node.ip)})</label></h4>
    <div class="grid">
      <label>Link delay (ms) <input name="delay_${id}" type="number" step="any" min="0" placeholder="none"></label>
      <label>Variation <select name="variation_${id}"><option value="constant">constant</option><option value="uniform">uniform redraw</option><option value="square">square wave</option></select></label>
      <label>Range min/max (ms) <span><input name="vmin_${id}" type="number" step="any" min="0" size="5"> <input name="vmax_${id}" type="number" step="any" min="0" size="5"></span></label>
      <label>Period (s) <input name="period_${id}" type="number" step="any" min="0.1" value="10"></label>
      <label>Jitter (ms) <input name="jitter_${id}" type="number" step="any" min="0" placeholder="none"></label>
      <label>Loss (%) <input name="loss_${id}" type="number" step="any" min="0" max="100" placeholder="none"></label>
      <label>Bandwidth cap (Mbit/s) <input name="rate_${id}" type="number" step="any" min="0.1" placeholder="none"></label>
      <label>Load mode <select name="loadmode_${id}"><option value="none">none</option><option value="fixed">fixed</option><option value="regime">regime</option><option value="processes">backlogged processes</option></select></label>
      <label>CPU (% of one core) <input name="cpu_${id}" type="number" min="0" max="100" value="30"></label>
      <label>Memory (% in use) <input name="mem_${id}" type="number" min="0" max="95" value="30"></label>
      <label>Regime <select name="regime_${id}"><option>light</option><option>moderate</option><option>heavy</option></select></label>
      <label>Processes <input name="procs_${id}" type="number" min="1" value="1"></label>
      <label>Max CPU frequency (MHz) <input name="freq_${id}" type="number" min="100" placeholder="unchanged"></label>
      <label>Memory limit (MB) <input name="memlimit_${id}" type="number" min="64" placeholder="unchanged"></label>
      <label>Threads <input name="threads_${id}" type="number" min="1" placeholder="all"></label>
    </div>
  </fieldset>`;
}

function runtimeInfo(name) {
  return ((state.catalog && state.catalog.runtimes) || []).find((r) => r.name === name) || null;
}

function renderRuntimeBlock() {
  const form = $("#form-scenario");
  const runtime = runtimeInfo(form.runtime.value);
  const block = $("#runtime-block");
  if (!runtime || !runtime.schema || !Object.keys(runtime.schema.properties || {}).length) {
    block.innerHTML = "";
    return;
  }
  block.innerHTML = `<div class="schema-block"><h4>${escapeHtml(runtime.label)}</h4><p class="help">${escapeHtml(runtime.description || "")}</p>${schemaForm(runtime.schema, "rt")}</div>`;
}

function renderModelArgsBlock() {
  const form = $("#form-scenario");
  const parameters = (state.catalog && state.catalog.model_parameters) || {};
  const schema = parameters[form.model.value];
  const block = $("#model-args-block");
  if (!schema || !Object.keys(schema.properties || {}).length) {
    block.innerHTML = "";
    return;
  }
  block.innerHTML = `<div class="schema-block"><h4>Model parameters (${escapeHtml(form.model.value)})</h4>${schemaForm(schema, "ma")}</div>`;
}

function parseSweepValues(text) {
  return text.split(",").map((v) => v.trim()).filter((v) => v !== "").map((v) => {
    if (v === "true") return true;
    if (v === "false") return false;
    const number = Number(v);
    return Number.isNaN(number) ? v : number;
  });
}

function buildScenario(form) {
  const scenario = {
    name: form.name.value.trim(),
    pipeline: "experiment",
    runtime: form.runtime.value,
    model: form.model.value,
    devices: [],
    repetitions: Number(form.repetitions.value),
    warmup: Number(form.warmup.value),
    seed: Number(form.seed.value),
    settle_s: Number(form.settle_s.value),
    edge_server: { policy: form.policy.value },
    network: {},
    load: {},
    heterogeneity: {},
    optimisation: { enabled: true },
  };
  if (form.inference_enabled.checked) {
    scenario.inference = { enabled: true, partition_points: form.partition_points.value.split(",").map((v) => Number(v.trim())).filter((v) => !Number.isNaN(v)) };
  }
  const runtime = runtimeInfo(form.runtime.value);
  if (runtime && runtime.schema && Object.keys(runtime.schema.properties || {}).length) {
    scenario[runtime.name] = readSchemaForm(form, runtime.schema, "rt");
    if (runtime.external) scenario.optimisation = { enabled: false };
  }
  const parameters = (state.catalog && state.catalog.model_parameters) || {};
  if (parameters[form.model.value]) {
    const args = readSchemaForm(form, parameters[form.model.value], "ma");
    if (Object.keys(args).length) scenario.model_args = args;
  }
  const sweepPath = form.sweep_path.value.trim();
  if (sweepPath) {
    const values = parseSweepValues(form.sweep_values.value);
    if (!values.length) throw new Error("the sweep needs at least one value");
    scenario.sweep = { path: sweepPath, values };
  }
  const number = (name) => (form[name] && form[name].value !== "" ? Number(form[name].value) : null);
  for (const node of state.nodes.values()) {
    if (node.role === "Base Station" || !form[`use_${node.id}`] || !form[`use_${node.id}`].checked) continue;
    const id = node.id;
    scenario.devices.push(id);
    const net = {};
    const delay = number(`delay_${id}`);
    if (delay !== null) net.delay_ms = delay;
    const variation = form[`variation_${id}`].value;
    if (variation !== "constant") {
      net.variation = { mode: variation, min_ms: number(`vmin_${id}`) ?? delay ?? 0, max_ms: number(`vmax_${id}`) ?? delay ?? 0, period_s: number(`period_${id}`) ?? 10 };
    }
    for (const [key, field] of [["jitter_ms", "jitter"], ["loss_percent", "loss"], ["rate_mbps", "rate"]]) {
      const value = number(`${field}_${id}`);
      if (value !== null) net[key] = value;
    }
    if (Object.keys(net).length) scenario.network[id] = net;
    const loadMode = form[`loadmode_${id}`].value;
    if (loadMode === "fixed") scenario.load[id] = { mode: "fixed", cpu_percent: number(`cpu_${id}`) ?? 0, memory_percent: number(`mem_${id}`) ?? 0 };
    else if (loadMode === "regime") scenario.load[id] = { mode: "regime", regime: form[`regime_${id}`].value };
    else if (loadMode === "processes") scenario.load[id] = { mode: "processes", processes: number(`procs_${id}`) ?? 1 };
    const het = {};
    if (number(`freq_${id}`) !== null) het.max_freq_mhz = number(`freq_${id}`);
    if (number(`memlimit_${id}`) !== null) het.memory_limit_mb = number(`memlimit_${id}`);
    if (number(`threads_${id}`) !== null) het.threads = number(`threads_${id}`);
    if (Object.keys(het).length) scenario.heterogeneity[id] = het;
  }
  for (const key of ["network", "load", "heterogeneity"]) if (!Object.keys(scenario[key]).length) delete scenario[key];
  if (!scenario.devices.length) scenario.devices = null;
  return scenario;
}

$("#btn-scenario").addEventListener("click", () => {
  const container = $("#scenario-nodes");
  container.innerHTML = "";
  for (const node of endDevices()) container.insertAdjacentHTML("beforeend", nodeConfigHtml(node));
  renderRuntimeBlock();
  renderModelArgsBlock();
  $("#dlg-scenario").showModal();
});
$("#scenario-runtime").addEventListener("change", renderRuntimeBlock);
$("#scenario-model").addEventListener("change", renderModelArgsBlock);
$("#btn-derive-latency").addEventListener("click", async () => {
  const { latency_ms } = await api("/api/latency-from-map", { method: "POST" });
  const form = $("#form-scenario");
  for (const [id, entry] of Object.entries(latency_ms)) {
    if (form[`delay_${id}`] && entry.edge_server !== undefined) form[`delay_${id}`].value = entry.edge_server;
  }
  terminal.append("[nnpm][info] link latencies derived from the marker positions (5 ms + 1 ms/km, clipped to 5-50 ms)");
});
async function saveScenario(run) {
  const form = $("#form-scenario");
  try {
    const scenario = buildScenario(form);
    const saved = await api("/api/scenarios", { method: "POST", body: scenario });
    terminal.append(`[nnpm][info] scenario ${saved.name} saved as version ${saved.version} (${saved.sha256.slice(0, 12)})`);
    await refreshRuns();
    $("#dlg-scenario").close();
    if (run) await startRun({ name: saved.name, version: saved.version });
  } catch (error) {
    alert(error.message);
  }
}
$("#form-scenario").addEventListener("submit", (event) => {
  event.preventDefault();
  saveScenario(false);
});
$("#btn-save-and-run").addEventListener("click", () => saveScenario(true));

// --- Runs -----------------------------------------------------------------------------------------------

function setRunIndicator(text, running) {
  const indicator = $("#run-indicator");
  indicator.textContent = text;
  indicator.className = `indicator ${running ? "running" : "idle"}`;
}

function subscribeRunLog(runId) {
  if (state.logSource) state.logSource.close();
  const source = new EventSource(`/events/runs/${encodeURIComponent(runId)}`);
  source.addEventListener("log", (event) => {
    terminal.append(event.data);
    if (event.data.startsWith("[nnpm][done]")) {
      source.close();
      state.logSource = null;
      setRunIndicator(`${runId}: ${event.data.replace("[nnpm][done] ", "")}`, false);
      refreshRuns(); // a sweep's next run becomes current and is followed from here
      reloadNodes();
    }
  });
  source.onerror = () => source.close();
  state.logSource = source;
}

async function startRun(body) {
  try {
    const started = await api("/api/runs", { method: "POST", body });
    state.currentRun = started.run_id;
    setRunIndicator(`${started.run_id}: running`, true);
    terminal.append(`[nnpm][info] run ${started.run_id} started (scenario ${started.scenario.name} v${started.scenario.version})`);
    subscribeRunLog(started.run_id);
  } catch (error) {
    alert(error.message);
  }
}

async function refreshRuns() {
  const { runs, current, busy } = await api("/api/runs");
  const list = $("#runs-list");
  list.innerHTML = "";
  for (const run of runs.slice(0, 30)) {
    const item = document.createElement("li");
    const sweep = run.sweep_origin ? ` <span class="muted">(${escapeHtml(run.sweep_origin.path)} = ${escapeHtml(run.sweep_origin.value)})</span>` : "";
    item.innerHTML = `<span><a href="/runs/${encodeURIComponent(run.run_id)}" target="_blank">${escapeHtml(run.run_id)}</a> <span class="muted">${escapeHtml(run.scenario_name || "")}</span>${sweep}</span><span class="status-${escapeHtml(run.status)}">${escapeHtml(run.status)}</span>`;
    list.appendChild(item);
  }
  const select = $("#optimise-runs");
  select.innerHTML = runs.map((r) => `<option value="${escapeHtml(r.run_id)}">${escapeHtml(r.run_id)} (${escapeHtml(r.scenario_name || "?")}, ${escapeHtml(r.status)})</option>`).join("");
  const { scenarios } = await api("/api/scenarios");
  $("#run-scenarios").innerHTML = scenarios.flatMap((s) => s.versions.map((v) => `<option value="${escapeHtml(s.name)}:${v.version}">${escapeHtml(s.name)} v${v.version}</option>`)).join("");
  if (busy && current && current !== state.currentRun) {
    state.currentRun = current;
    setRunIndicator(`${current}: running`, true);
    subscribeRunLog(current);
  }
}

$("#btn-run").addEventListener("click", () => $("#dlg-run").showModal());
$("#form-run").addEventListener("submit", async (event) => {
  event.preventDefault();
  const [name, version] = event.currentTarget.scenario.value.split(":");
  $("#dlg-run").close();
  await startRun({ name, version: Number(version) });
});
$("#btn-benchmark").addEventListener("click", () => $("#dlg-benchmark").showModal());
$("#form-benchmark").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  // Active devices are preselected by reachability: the devices whose last probe was green.
  // Before the first probe round every registered device takes part (devices: null).
  const reachable = endDevices().filter((n) => statusClass(n.id) === "green").map((n) => n.id);
  const scenario = { name: `benchmark_${form.model.value}`, pipeline: "benchmarking", model: form.model.value, devices: reachable.length ? reachable : null, repetitions: Number(form.repetitions.value), warmup: Number(form.warmup.value), seed: Number(form.seed.value), optimisation: { enabled: false } };
  $("#dlg-benchmark").close();
  await startRun({ scenario, pipeline: "benchmarking" });
});
$("#btn-stop").addEventListener("click", async () => {
  try {
    if (state.currentTask) {
      await api(`/api/tasks/${encodeURIComponent(state.currentTask)}/stop`, { method: "POST" });
    }
    if (state.currentRun) {
      await api(`/api/runs/${encodeURIComponent(state.currentRun)}/stop`, { method: "POST" });
    }
  } catch (error) {
    alert(error.message);
  }
});

// --- Optimise dialog -------------------------------------------------------------------------------------

$("#btn-optimise").addEventListener("click", async () => {
  await refreshRuns();
  $("#optimise-result").innerHTML = "";
  $("#dlg-optimise").showModal();
});
$("#form-optimise").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const parseMaybeJson = (text) => {
    const value = text.trim();
    if (!value) return null;
    try {
      return JSON.parse(value);
    } catch {
      return value;
    }
  };
  const body = {
    run_id: form.run_id.value,
    algorithm: form.algorithm.value,
    policy: form.policy.value,
    objective: form.objective.value,
    transmission_mode: form.transmission_mode.value,
    initial_scheme: parseMaybeJson(form.initial_scheme.value),
    max_total_delays_ms: parseMaybeJson(form.max_total_delays_ms.value),
  };
  const target = $("#optimise-result");
  target.innerHTML = "<p class='muted'>Optimising...</p>";
  try {
    const { result, output_dir } = await api("/api/optimise", { method: "POST", body });
    const rows = result.per_device.map((d) => `<tr><td>${escapeHtml(d.node_id)}</td><td class="num">${d.s}</td><td class="num">${d.local_ms.toFixed(2)}</td><td class="num">${d.transmission_ms.toFixed(2)}</td><td class="num">${d.remote_ms.toFixed(2)}</td><td class="num">${d.total_ms.toFixed(2)}</td></tr>`).join("");
    target.innerHTML = `<p>Partition vector <strong>${JSON.stringify(result.optimized_partition_scheme)}</strong>, ${escapeHtml(result.objective)} = ${result.value.toFixed(6)} ${escapeHtml(result.value_unit)} (${escapeHtml(result.algorithm)}, ${escapeHtml(result.policy)}; all-local baseline ${result.all_local_baseline.value.toFixed(6)}). Logged to <code>${escapeHtml(output_dir)}</code> with the scenario.</p>
      <table><thead><tr><th>Device</th><th>s</th><th>Local (ms)</th><th>Transmission (ms)</th><th>Remote (ms)</th><th>Total (ms)</th></tr></thead><tbody>${rows}</tbody></table>`;
    terminal.append(`[nnpm][info] optimised partition vector ${JSON.stringify(result.optimized_partition_scheme)} for run ${body.run_id}`);
  } catch (error) {
    target.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
  }
});

// --- Runtimes dialog -------------------------------------------------------------------------------------
// Readiness checks of the devices for a runtime, and its maintenance actions (tasks).

function renderRuntimeActions() {
  const runtime = runtimeInfo($("#runtimes-select").value);
  $("#runtimes-description").textContent = runtime ? runtime.description : "";
  const container = $("#runtimes-actions");
  if (!runtime || !runtime.actions.length) {
    container.innerHTML = "<p class='muted'>This runtime has no maintenance action.</p>";
    return;
  }
  container.innerHTML = runtime.actions.map((action) => `
    <form class="runtime-action" data-action="${escapeHtml(action.name)}">
      <h4>${escapeHtml(action.label)}</h4>
      <p class="help">${escapeHtml(action.description)}</p>
      ${schemaForm(action.schema, `act_${action.name}`)}
      <menu><button type="submit">${escapeHtml(action.label)}</button></menu>
    </form>`).join("");
  $$("form.runtime-action", container).forEach((form) => form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const actionName = form.dataset.action;
    const spec = runtime.actions.find((a) => a.name === actionName);
    try {
      const params = readSchemaForm(form, spec.schema, `act_${actionName}`);
      const started = await api(`/api/runtimes/${encodeURIComponent(runtime.name)}/actions/${encodeURIComponent(actionName)}`, { method: "POST", body: params });
      state.currentTask = started.task_id;
      $("#runtimes-busy").textContent = `task ${started.task_id} running`;
      terminal.append(`[task][info] ${started.task_id} started`);
      subscribeTaskLog(started.task_id);
    } catch (error) {
      alert(error.message);
    }
  }));
}

function subscribeTaskLog(taskId) {
  if (state.taskSource) state.taskSource.close();
  const source = new EventSource(`/events/tasks/${encodeURIComponent(taskId)}`);
  source.addEventListener("log", (event) => {
    terminal.append(event.data);
    if (event.data.startsWith("[task][done]")) {
      source.close();
      state.taskSource = null;
      state.currentTask = null;
      $("#runtimes-busy").textContent = `task ${taskId}: ${event.data.replace("[task][done] ", "")}`;
    }
  });
  source.onerror = () => source.close();
  state.taskSource = source;
}

async function refreshRuntimeStatus() {
  const name = $("#runtimes-select").value;
  const container = $("#runtimes-status");
  container.innerHTML = "<p class='muted'>Checking the devices...</p>";
  const model = $("#runtimes-model").value.trim();
  const query = model ? `?model=${encodeURIComponent(model)}` : "";
  try {
    const { nodes } = await api(`/api/runtimes/${encodeURIComponent(name)}/status${query}`);
    const checkNames = [];
    for (const checks of Object.values(nodes)) for (const check of checks) if (!checkNames.includes(check.name)) checkNames.push(check.name);
    if (!checkNames.length) {
      container.innerHTML = "<p class='muted'>This runtime needs nothing beyond provisioning.</p>";
      return;
    }
    const head = `<tr><th>Node</th>${checkNames.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr>`;
    const rows = Object.entries(nodes).filter(([, checks]) => checks.length).map(([nodeId, checks]) => {
      const cells = checkNames.map((c) => {
        const check = checks.find((k) => k.name === c);
        return check ? `<td class="cell cell-${escapeHtml(check.status)}" title="${escapeHtml(check.detail)}">${escapeHtml(check.status)}</td>` : "<td class='cell cell-skip'></td>";
      }).join("");
      return `<tr><td>${escapeHtml(nodeId)}</td>${cells}</tr>`;
    }).join("");
    container.innerHTML = `<table class="runtime-status"><thead>${head}</thead><tbody>${rows}</tbody></table><p class="help">Hover a cell for the detail (commit, versions, paths).</p>`;
  } catch (error) {
    container.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
  }
}

$("#btn-runtimes").addEventListener("click", () => {
  renderRuntimeActions();
  $("#runtimes-status").innerHTML = "";
  $("#dlg-runtimes").showModal();
});
$("#runtimes-select").addEventListener("change", () => {
  renderRuntimeActions();
  $("#runtimes-status").innerHTML = "";
});
$("#btn-runtimes-refresh").addEventListener("click", refreshRuntimeStatus);

// --- Dialog close buttons --------------------------------------------------------------------------------

$$("[data-close]").forEach((button) => button.addEventListener("click", () => button.closest("dialog").close()));

// --- Boot -----------------------------------------------------------------------------------------------------

async function loadCatalog() {
  try {
    state.catalog = await api("/api/catalog");
  } catch (error) {
    terminal.append(`[nnpm][error] catalogue unavailable: ${error.message}`);
    state.catalog = { runtimes: [], model_parameters: {} };
  }
}

renderMarkers();
fitMap();
refreshLinks();
subscribeStatus();
loadCatalog().then(refreshRuns);
terminal.append(`[nnpm][info] ${boot.localMode ? "local mode: every node is simulated on this machine" : "connected to the testbed"}`);
