import {
  getHealth, getReadiness, getInfo, getDemoAlerts,
  analyzeConversation, analyzeBatch,
} from "/static/js/api.js";

const byId = (id) => document.getElementById(id);
const score = (value) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "—";
const count = (value) => typeof value === "number" && Number.isFinite(value) ? String(value) : "—";
const safeText = (value) => value === null || value === undefined ? "—" : String(value);

function feedback(id, message, tone = "neutral") {
  const node = byId(id);
  node.textContent = message;
  node.dataset.tone = tone;
}

function errorMessage(error) {
  if (error?.status === 422) return `Invalid message: ${error.message}`;
  if (error?.status === 503) return error.message;
  if (error?.status === 0) return error.message;
  return "Analysis failed. Please try again later.";
}

function setStatus(cardId, textId, detailId, state, title, detail) {
  byId(cardId).dataset.state = state;
  byId(textId).textContent = title;
  byId(detailId).textContent = detail;
}

async function loadStatus() {
  setStatus("api-status-card", "api-status-text", "api-status-detail", "checking", "Checking…", "Checking the health endpoint.");
  setStatus("ready-status-card", "ready-status-text", "ready-status-detail", "checking", "Checking…", "Checking local model artifacts.");
  const [health, readiness] = await Promise.allSettled([getHealth(), getReadiness()]);
  if (health.status === "fulfilled" && health.value.status === "ok") {
    setStatus("api-status-card", "api-status-text", "api-status-detail", "ok", "Online", "The API process is responding.");
  } else {
    setStatus("api-status-card", "api-status-text", "api-status-detail", "error", "Unavailable", "The health endpoint could not be confirmed.");
  }
  if (readiness.status === "fulfilled" && readiness.value.status === "ready") {
    setStatus("ready-status-card", "ready-status-text", "ready-status-detail", "ok", "Ready", "Local analysis artifacts are available.");
  } else {
    const detail = readiness.status === "rejected" && readiness.reason?.status === 503
      ? "Required analysis artifacts are unavailable."
      : "The readiness endpoint could not be confirmed.";
    setStatus("ready-status-card", "ready-status-text", "ready-status-detail", "error", "Unavailable", detail);
  }
}

function addCell(row, value, className = "") {
  const cell = document.createElement("td");
  cell.textContent = safeText(value);
  if (className) cell.className = className;
  row.append(cell);
}

function renderTerms(terms) {
  const target = byId("result-terms");
  target.replaceChildren();
  if (!Array.isArray(terms) || terms.length === 0) {
    target.textContent = "No descriptive terms available.";
    return;
  }
  for (const term of terms) {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = safeText(term);
    target.append(tag);
  }
}

function renderSingle(result) {
  byId("result-domain").textContent = safeText(result.domain);
  byId("result-domain-score").textContent = score(result.domain_confidence);
  byId("result-cluster").textContent = count(result.semantic_cluster);
  byId("result-similarity").textContent = score(result.cluster_similarity);
  renderTerms(result.cluster_descriptive_terms);
  const meta = result.analysis_metadata || {};
  byId("meta-domain-model").textContent = safeText(meta.domain_model);
  byId("meta-domain-input").textContent = safeText(meta.domain_input_mode);
  byId("meta-embedding-model").textContent = safeText(meta.embedding_model);
  byId("meta-cluster-count").textContent = count(meta.semantic_cluster_count);
  byId("single-placeholder").hidden = true;
  byId("single-result").hidden = false;
}

async function submitSingle(event) {
  event.preventDefault();
  const text = byId("customer-message").value;
  if (!text.trim()) {
    byId("single-result").hidden = true;
    byId("single-placeholder").hidden = false;
    byId("single-placeholder").querySelector("h3").textContent = "No result yet";
    byId("single-placeholder").querySelector("p").textContent = "Enter a customer message to start analysis.";
    feedback("single-feedback", "Enter a customer message before analyzing.", "error");
    byId("customer-message").focus();
    return;
  }
  const button = byId("single-submit");
  button.disabled = true;
  button.textContent = "Analyzing…";
  byId("single-result").hidden = true;
  byId("single-placeholder").hidden = false;
  byId("single-placeholder").querySelector("h3").textContent = "Analyzing conversation…";
  byId("single-placeholder").querySelector("p").textContent = "The first analysis may load the local embedding model.";
  feedback("single-feedback", "Analysis in progress…");
  try {
    const result = await analyzeConversation(text); // Keep the original transcript unchanged.
    renderSingle(result);
    feedback("single-feedback", "Analysis complete.", "success");
  } catch (error) {
    byId("single-placeholder").querySelector("h3").textContent = "No result yet";
    byId("single-placeholder").querySelector("p").textContent = "Check the message or service status, then try again.";
    feedback("single-feedback", errorMessage(error), "error");
  } finally {
    button.disabled = false;
    button.textContent = "Analyze Conversation →";
  }
}

function batchMessages() {
  return byId("batch-messages").value.split(/\r?\n/).filter((line) => line.trim());
}

function updateBatchCount() {
  const length = batchMessages().length;
  byId("batch-count").textContent = `${length} message${length === 1 ? "" : "s"} · maximum 50`;
}

function renderBatch(messages, results) {
  const body = byId("batch-table-body");
  body.replaceChildren();
  results.forEach((result, index) => {
    const row = document.createElement("tr");
    addCell(row, index + 1, "numeric");
    addCell(row, messages[index].trim().slice(0, 100), "input-preview");
    addCell(row, result.domain, "table-domain");
    addCell(row, score(result.domain_confidence), "numeric");
    addCell(row, count(result.semantic_cluster), "numeric");
    addCell(row, score(result.cluster_similarity), "numeric");
    body.append(row);
  });
  byId("batch-result").hidden = false;
}

async function submitBatch(event) {
  event.preventDefault();
  const messages = batchMessages();
  if (messages.length === 0) {
    byId("batch-result").hidden = true;
    feedback("batch-feedback", "Add at least one customer message.", "error");
    byId("batch-messages").focus();
    return;
  }
  if (messages.length > 50) {
    byId("batch-result").hidden = true;
    feedback("batch-feedback", "A batch can contain at most 50 messages.", "error");
    byId("batch-messages").focus();
    return;
  }
  const button = byId("batch-submit");
  button.disabled = true;
  button.textContent = "Analyzing batch…";
  byId("batch-result").hidden = true;
  feedback("batch-feedback", `Analyzing ${messages.length} messages…`);
  try {
    const data = await analyzeBatch(messages);
    if (!Array.isArray(data.results) || data.results.length !== messages.length) {
      throw new Error("Unexpected batch response");
    }
    renderBatch(messages, data.results);
    feedback("batch-feedback", `${messages.length} messages analyzed in input order.`, "success");
  } catch (error) {
    feedback("batch-feedback", errorMessage(error), "error");
  } finally {
    button.disabled = false;
    button.textContent = "Analyze Batch →";
  }
}

function formatTimestamp(value) {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? safeText(value) : `${parsed.toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

async function loadAlerts() {
  try {
    const data = await getDemoAlerts();
    if (data.temporal_mode !== "synthetic_demo" || data.is_real_time !== false || !Array.isArray(data.alerts)) {
      throw new Error("Unexpected temporal mode");
    }
    const body = byId("alerts-table-body");
    body.replaceChildren();
    for (const alert of data.alerts) {
      const row = document.createElement("tr");
      addCell(row, formatTimestamp(alert.timestamp), "numeric");
      addCell(row, count(alert.cluster_id), "numeric");
      addCell(row, count(alert.current_count), "numeric");
      addCell(row, typeof alert.historical_mean === "number" ? alert.historical_mean.toFixed(2) : "—", "numeric");
      addCell(row, score(alert.anomaly_score), "numeric");
      addCell(row, typeof alert.increase_ratio === "number" ? `${alert.increase_ratio.toFixed(2)}×` : "—", "numeric");
      addCell(row, Array.isArray(alert.descriptive_terms) ? alert.descriptive_terms.join(", ") : "—");
      body.append(row);
    }
    byId("alerts-table").hidden = false;
    feedback("alerts-feedback", data.alerts.length ? "" : "No synthetic alerts in this demo.");
  } catch {
    byId("alerts-table").hidden = true;
    feedback("alerts-feedback", "Synthetic demo alerts are unavailable. Check the API server and demo artifact.", "error");
  }
}

async function loadInfo() {
  try {
    const info = await getInfo();
    byId("info-version").textContent = safeText(info.api_version);
    byId("info-domain").textContent = safeText(info.domain_classifier);
    byId("info-embedding").textContent = safeText(info.semantic_embedding_model);
    byId("info-clusters").textContent = count(info.semantic_cluster_count);
    byId("info-temporal").textContent = safeText(info.temporal_mode);
    byId("info-list").hidden = false;
    feedback("info-feedback", "");
  } catch {
    byId("info-list").hidden = true;
    feedback("info-feedback", "Model information is unavailable. Check the API server.", "error");
  }
}

byId("refresh-status").addEventListener("click", loadStatus);
byId("single-form").addEventListener("submit", submitSingle);
byId("batch-form").addEventListener("submit", submitBatch);
byId("batch-messages").addEventListener("input", updateBatchCount);
updateBatchCount();
loadStatus();
loadInfo();
loadAlerts();
