// HTTP concerns stay here; app.js only handles forms and safe DOM updates.
export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request(path, { method = "GET", body } = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120_000);
  try {
    const response = await fetch(path, {
      method,
      headers: { Accept: "application/json", ...(body === undefined ? {} : { "Content-Type": "application/json" }) },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
    let payload;
    try {
      payload = await response.json();
    } catch {
      throw new ApiError(response.status, "The server returned an unreadable response.");
    }
    if (!response.ok) {
      if (response.status === 422) {
        const firstError = Array.isArray(payload.detail) ? payload.detail[0] : null;
        throw new ApiError(422, firstError?.msg || "Please check the message and try again.");
      }
      if (response.status === 503) {
        throw new ApiError(503, "Analysis service unavailable. Check readiness and local artifacts.");
      }
      throw new ApiError(response.status, "The server could not complete this request.");
    }
    if (payload === null || typeof payload !== "object") {
      throw new ApiError(response.status, "The server returned an unexpected response.");
    }
    return payload;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error.name === "AbortError") throw new ApiError(0, "The request timed out. Please try again.");
    throw new ApiError(0, "Cannot reach the API server. Check that SignalDesk is running.");
  } finally {
    clearTimeout(timeout);
  }
}

export const getHealth = () => request("/health");
export const getReadiness = () => request("/ready");
export const getInfo = () => request("/api/v1/info");
export const getDemoAlerts = () => request("/api/v1/alerts/demo");
export const analyzeConversation = (customerText) =>
  request("/api/v1/analyze", { method: "POST", body: { customer_text: customerText } });
export const analyzeBatch = (messages) =>
  request("/api/v1/analyze/batch", {
    method: "POST",
    body: { items: messages.map((customerText) => ({ customer_text: customerText })) },
  });
