import {
  getHealth, getReadiness, getInfo, getDemoAlerts,
  analyzeConversation, analyzeBatch,
} from "/static/js/api.js?v=12";

const byId = (id) => document.getElementById(id);
const score = (value) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "—";
const count = (value) => typeof value === "number" && Number.isFinite(value) ? String(value) : "—";
const safeText = (value) => value === null || value === undefined ? "—" : String(value);

// API değerlerini değiştirmeden yalnızca kullanıcıya gösterilen karşılıkları Türkçeleştirir.
const DOMAIN_LABELS = {
  agriculture: "Tarım",
  aviation: "Havacılık",
  banking: "Bankacılık",
  deliveryservice: "Teslimat",
  education: "Eğitim",
  energy: "Enerji",
  entertainment: "Eğlence",
  finance: "Finans",
  food: "Yiyecek ve İçecek",
  health: "Sağlık",
  hospitality: "Konaklama",
  insurance: "Sigorta",
  realestate: "Gayrimenkul",
  retail: "Perakende",
  technology: "Teknoloji",
  telecom: "Telekomünikasyon",
  travel: "Seyahat",
};

const domainLabel = (value) => DOMAIN_LABELS[String(value ?? "").toLowerCase()] ?? safeText(value);
const domainInputLabel = (value) => value === "customer_text" ? "Müşteri metni (customer_text)" : safeText(value);
const temporalModeLabel = (value) => value === "synthetic_demo" ? "Sentetik demo (synthetic_demo)" : safeText(value);

function feedback(id, message, tone = "neutral") {
  const node = byId(id);
  node.textContent = message;
  node.dataset.tone = tone;
}

function errorMessage(error) {
  if (error?.status === 422) return `Geçersiz mesaj: ${error.message}`;
  if (error?.status === 503) return error.message;
  if (error?.status === 0) return error.message;
  return "Analiz tamamlanamadı. Lütfen daha sonra yeniden deneyin.";
}

function setStatus(cardId, textId, detailId, state, title, detail) {
  byId(cardId).dataset.state = state;
  byId(textId).textContent = title;
  byId(detailId).textContent = detail;
}

async function loadStatus() {
  setStatus("api-status-card", "api-status-text", "api-status-detail", "checking", "Kontrol ediliyor…", "API bağlantısı kontrol ediliyor.");
  setStatus("ready-status-card", "ready-status-text", "ready-status-detail", "checking", "Kontrol ediliyor…", "Yerel model dosyaları kontrol ediliyor.");
  const [health, readiness] = await Promise.allSettled([getHealth(), getReadiness()]);
  if (health.status === "fulfilled" && health.value.status === "ok") {
    setStatus("api-status-card", "api-status-text", "api-status-detail", "ok", "Çalışıyor", "API hizmeti yanıt veriyor.");
  } else {
    setStatus("api-status-card", "api-status-text", "api-status-detail", "error", "Kullanılamıyor", "API sağlık durumu doğrulanamadı.");
  }
  if (readiness.status === "fulfilled" && readiness.value.status === "ready") {
    setStatus("ready-status-card", "ready-status-text", "ready-status-detail", "ok", "Hazır", "Analiz için gerekli yerel model dosyaları kullanılabilir.");
  } else {
    const detail = readiness.status === "rejected" && readiness.reason?.status === 503
      ? "Analiz için gerekli model dosyaları kullanılamıyor."
      : "Model hazırlık durumu doğrulanamadı.";
    setStatus("ready-status-card", "ready-status-text", "ready-status-detail", "error", "Kullanılamıyor", detail);
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
    target.textContent = "Açıklama terimi bulunamadı.";
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
  byId("result-domain").textContent = domainLabel(result.domain);
  byId("result-domain-score").textContent = score(result.domain_confidence);
  byId("result-cluster").textContent = count(result.semantic_cluster);
  byId("result-similarity").textContent = score(result.cluster_similarity);
  renderTerms(result.cluster_descriptive_terms);
  const meta = result.analysis_metadata || {};
  byId("meta-domain-model").textContent = safeText(meta.domain_model);
  byId("meta-domain-input").textContent = domainInputLabel(meta.domain_input_mode);
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
    byId("single-placeholder").querySelector("h3").textContent = "Henüz sonuç yok";
    byId("single-placeholder").querySelector("p").textContent = "Analizi başlatmak için bir müşteri mesajı girin.";
    feedback("single-feedback", "Analizden önce bir müşteri mesajı girin.", "error");
    byId("customer-message").focus();
    return;
  }
  const button = byId("single-submit");
  button.disabled = true;
  button.textContent = "Analiz ediliyor…";
  byId("single-result").hidden = true;
  byId("single-placeholder").hidden = false;
  byId("single-placeholder").querySelector("h3").textContent = "Görüşme analiz ediliyor…";
  byId("single-placeholder").querySelector("p").textContent = "İlk analizde yerel embedding modeli yüklenebilir.";
  feedback("single-feedback", "Analiz sürüyor…");
  try {
    const result = await analyzeConversation(text); // Keep the original transcript unchanged.
    renderSingle(result);
    feedback("single-feedback", "Analiz tamamlandı.", "success");
  } catch (error) {
    byId("single-placeholder").querySelector("h3").textContent = "Henüz sonuç yok";
    byId("single-placeholder").querySelector("p").textContent = "Mesajı ve sistem durumunu kontrol edip yeniden deneyin.";
    feedback("single-feedback", errorMessage(error), "error");
  } finally {
    button.disabled = false;
    button.textContent = "Görüşmeyi Analiz Et →";
  }
}

function batchMessages() {
  return byId("batch-messages").value.split(/\r?\n/).filter((line) => line.trim());
}

function updateBatchCount() {
  const length = batchMessages().length;
  byId("batch-count").textContent = `${length} mesaj · en fazla 50`;
}

function renderBatch(messages, results) {
  const body = byId("batch-table-body");
  body.replaceChildren();
  results.forEach((result, index) => {
    const row = document.createElement("tr");
    addCell(row, index + 1, "numeric");
    addCell(row, messages[index].trim().slice(0, 100), "input-preview");
    addCell(row, domainLabel(result.domain), "table-domain");
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
    feedback("batch-feedback", "En az bir müşteri mesajı ekleyin.", "error");
    byId("batch-messages").focus();
    return;
  }
  if (messages.length > 50) {
    byId("batch-result").hidden = true;
    feedback("batch-feedback", "Toplu analiz en fazla 50 mesaj içerebilir.", "error");
    byId("batch-messages").focus();
    return;
  }
  const button = byId("batch-submit");
  button.disabled = true;
  button.textContent = "Toplu analiz yapılıyor…";
  byId("batch-result").hidden = true;
  feedback("batch-feedback", `${messages.length} mesaj analiz ediliyor…`);
  try {
    const data = await analyzeBatch(messages);
    if (!Array.isArray(data.results) || data.results.length !== messages.length) {
      throw new Error("Beklenmeyen toplu analiz yanıtı");
    }
    renderBatch(messages, data.results);
    feedback("batch-feedback", `${messages.length} mesaj, girdi sırasına göre analiz edildi.`, "success");
  } catch (error) {
    feedback("batch-feedback", errorMessage(error), "error");
  } finally {
    button.disabled = false;
    button.textContent = "Toplu Analizi Başlat →";
  }
}

function formatTimestamp(value) {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? safeText(value) : `${parsed.toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

function alertSummary(alert) {
  const terms = Array.isArray(alert.descriptive_terms)
    ? alert.descriptive_terms.map(safeText).filter((term) => term !== "—").slice(0, 3)
    : [];
  const subject = terms.length ? `“${terms.join(", ")}” terimleriyle ilişkili görüşmelerde` : "Bu sorun kümesindeki görüşmelerde";
  return `${subject} olağandışı artış tespit edildi.`;
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
      addCell(row, alertSummary(alert));
      body.append(row);
    }
    byId("alerts-table").hidden = false;
    feedback("alerts-feedback", data.alerts.length ? "" : "Aktif uyarı bulunmuyor.");
  } catch {
    byId("alerts-table").hidden = true;
    feedback("alerts-feedback", "Sentetik demo uyarılarına erişilemiyor. API sunucusunu ve demo dosyasını kontrol edin.", "error");
  }
}

async function loadInfo() {
  try {
    const info = await getInfo();
    byId("info-version").textContent = safeText(info.api_version);
    byId("info-domain").textContent = safeText(info.domain_classifier);
    byId("info-embedding").textContent = safeText(info.semantic_embedding_model);
    byId("info-clusters").textContent = count(info.semantic_cluster_count);
    byId("info-temporal").textContent = temporalModeLabel(info.temporal_mode);
    byId("info-list").hidden = false;
    feedback("info-feedback", "");
  } catch {
    byId("info-list").hidden = true;
    feedback("info-feedback", "Model bilgilerine erişilemiyor. API sunucusunu kontrol edin.", "error");
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
