export class ApiError extends Error {
  status: number;
  payload: unknown;

  constructor(message: string, status = 0, payload: unknown = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

export type SessionUser = {
  username: string;
};

export type PredictionResult = {
  tags: string[];
  message?: string;
};

export type HealthState = "healthy" | "degraded" | "unavailable" | "unknown";

export type ServiceHealth = {
  label: string;
  state: HealthState;
  detail?: string;
};

export type RecentRequest = {
  id?: string;
  title?: string;
  preview: string;
  createdAt?: string;
  status: string;
  responseTimeMs?: number;
  tags: string[];
};

export type DashboardData = {
  totalRequests?: number;
  failedRequests?: number;
  averageResponseTimeMs?: number;
  successRate?: number;
  recentRequests: RecentRequest[];
  services: ServiceHealth[];
};

type LooseRecord = Record<string, unknown>;

type RequestOptions = Omit<RequestInit, "body"> & {
  data?: unknown;
};

const jsonHeaders = {
  Accept: "application/json",
};

function isRecord(value: unknown): value is LooseRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asRecord(value: unknown): LooseRecord | null {
  return isRecord(value) ? value : null;
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function firstString(...values: unknown[]): string | undefined {
  for (const value of values) {
    if (typeof value === "string") {
      const trimmed = value.trim();
      if (trimmed) {
        return trimmed;
      }
    }
  }
  return undefined;
}

function toNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return undefined;
}

function toBoolean(value: unknown): boolean | undefined {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (["true", "ok", "healthy", "ready", "available", "up", "success"].includes(normalized)) {
      return true;
    }
    if (["false", "down", "error", "failed", "unavailable", "degraded"].includes(normalized)) {
      return false;
    }
  }
  return undefined;
}

function deriveMessage(payload: unknown, fallback: string): string {
  const record = asRecord(payload);
  return (
    firstString(
      record?.message,
      record?.detail,
      record?.error,
      record?.description,
      typeof payload === "string" ? payload : undefined,
    ) ?? fallback
  );
}

async function parsePayload(response: Response): Promise<unknown> {
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }

  try {
    const text = await response.text();
    return text || null;
  } catch {
    return null;
  }
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers = new Headers(options.headers ?? {});
  headers.set("Accept", jsonHeaders.Accept);

  const init: RequestInit = {
    ...options,
    credentials: "include",
    headers,
  };

  if (options.data !== undefined) {
    headers.set("Content-Type", "application/json");
    init.body = JSON.stringify(options.data);
  }

  try {
    const response = await fetch(path, init);
    const payload = await parsePayload(response);

    if (!response.ok) {
      throw new ApiError(deriveMessage(payload, response.statusText || "Request failed."), response.status, payload);
    }

    return payload as T;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError("Unable to reach the application backend.", 0, null);
  }
}

async function requestWithFallback<T>(paths: string[], options: RequestOptions = {}): Promise<T> {
  let lastError: unknown = null;

  for (const path of paths) {
    try {
      return await request<T>(path, options);
    } catch (error) {
      lastError = error;
      if (error instanceof ApiError && error.status === 404) {
        continue;
      }
      throw error;
    }
  }

  if (lastError instanceof Error) {
    throw lastError;
  }

  throw new ApiError("No supported API endpoint was found.", 404, null);
}

function normalizeSession(payload: unknown): SessionUser | null {
  const record = asRecord(payload);
  const userRecord = asRecord(record?.user);
  const username = firstString(record?.username, record?.user_name, userRecord?.username, userRecord?.user_name);
  const authenticated = toBoolean(record?.authenticated ?? record?.ok ?? record?.success);

  if (username) {
    return { username };
  }

  if (authenticated === false) {
    return null;
  }

  return null;
}

function extractTags(payload: unknown): string[] {
  if (Array.isArray(payload)) {
    return payload.filter((value): value is string => typeof value === "string");
  }

  const record = asRecord(payload);
  const directTags = asArray(record?.tags).filter((value): value is string => typeof value === "string");
  if (directTags.length > 0) {
    return directTags;
  }

  const predictions = asArray(record?.predictions);
  if (predictions.length > 0) {
    const firstPrediction = predictions[0];
    if (Array.isArray(firstPrediction)) {
      return firstPrediction.filter((value): value is string => typeof value === "string");
    }
    if (typeof firstPrediction === "string") {
      return predictions.filter((value): value is string => typeof value === "string");
    }
  }

  const resultTags = asArray(record?.result).filter((value): value is string => typeof value === "string");
  return resultTags;
}

function normalizeHealthState(value: unknown): HealthState {
  if (typeof value === "boolean") {
    return value ? "healthy" : "unavailable";
  }

  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (["healthy", "ok", "ready", "up", "available", "success"].includes(normalized)) {
      return "healthy";
    }
    if (["degraded", "warming", "partial"].includes(normalized)) {
      return "degraded";
    }
    if (["down", "failed", "unavailable", "error"].includes(normalized)) {
      return "unavailable";
    }
  }

  const record = asRecord(value);
  if (record) {
    const statusText = firstString(record.status, record.state, record.health);
    if (statusText) {
      return normalizeHealthState(statusText);
    }

    const ok = toBoolean(record.ok ?? record.healthy ?? record.available ?? record.ready ?? record.success);
    if (ok !== undefined) {
      return ok ? "healthy" : "unavailable";
    }
  }

  return "unknown";
}

function normalizeService(label: string, value: unknown): ServiceHealth {
  const record = asRecord(value);
  return {
    label,
    state: normalizeHealthState(value),
    detail: firstString(
      record?.detail,
      record?.message,
      record?.reason,
      record?.description,
      typeof value === "string" ? value : undefined,
    ),
  };
}

function inferStatusText(item: LooseRecord): string {
  const statusText = firstString(item.status, item.state, item.result);
  if (statusText) {
    return statusText;
  }

  const success = toBoolean(item.success ?? item.ok);
  if (success === true) {
    return "success";
  }
  if (success === false) {
    return "failed";
  }
  return "recorded";
}

function normalizeRecentRequest(entry: unknown): RecentRequest | null {
  const record = asRecord(entry);
  if (!record) {
    return null;
  }

  const title = firstString(record.title, record.article_title);
  const preview =
    firstString(record.preview, record.excerpt, record.body_preview, record.input_preview, record.snippet) ??
    firstString(record.title) ??
    "No preview available";

  return {
    id: firstString(record.id, record.request_id),
    title,
    preview,
    createdAt: firstString(record.created_at, record.timestamp, record.requested_at, record.date),
    status: inferStatusText(record),
    responseTimeMs: toNumber(record.response_time_ms ?? record.duration_ms ?? record.latency_ms),
    tags: asArray(record.tags ?? record.predicted_tags).filter((value): value is string => typeof value === "string"),
  };
}

function extractRecentRequests(statusPayload: unknown): RecentRequest[] {
  const record = asRecord(statusPayload);
  const requestCollections = [
    record?.recent_requests,
    record?.recentRequests,
    record?.user_requests,
    record?.requests,
    record?.history,
  ];

  for (const collection of requestCollections) {
    const items = asArray(collection)
      .map(normalizeRecentRequest)
      .filter((value): value is RecentRequest => value !== null);
    if (items.length > 0) {
      return items;
    }
  }

  return [];
}

function extractMetrics(statusPayload: unknown): Pick<DashboardData, "totalRequests" | "failedRequests" | "averageResponseTimeMs" | "successRate"> {
  const record = asRecord(statusPayload);
  const metrics = asRecord(record?.metrics);
  const summary = asRecord(record?.summary);

  return {
    totalRequests: toNumber(record?.total_requests ?? record?.totalRequests ?? metrics?.total_requests ?? metrics?.totalRequests ?? summary?.total_requests),
    failedRequests: toNumber(record?.failed_requests ?? record?.failedRequests ?? metrics?.failed_requests ?? metrics?.failedRequests ?? summary?.failed_requests),
    averageResponseTimeMs: toNumber(
      record?.average_response_time_ms ??
        record?.averageResponseTimeMs ??
        metrics?.average_response_time_ms ??
        metrics?.averageResponseTimeMs ??
        summary?.average_response_time_ms,
    ),
    successRate: toNumber(record?.success_rate ?? record?.successRate ?? metrics?.success_rate ?? metrics?.successRate),
  };
}

function extractServices(statusPayload: unknown, healthPayload: unknown): ServiceHealth[] {
  const statusRecord = asRecord(statusPayload);
  const nestedHealth = asRecord(statusRecord?.health);
  const healthRecord = asRecord(healthPayload);

  const serviceSources: Array<[string, unknown[]]> = [
    ["Backend", [healthRecord?.backend, nestedHealth?.backend, healthPayload, statusRecord?.backend]],
    ["Database", [healthRecord?.database, healthRecord?.db, nestedHealth?.database, nestedHealth?.db, statusRecord?.database, statusRecord?.db]],
    ["ML service", [healthRecord?.ml_service, healthRecord?.mlService, nestedHealth?.ml_service, nestedHealth?.mlService, statusRecord?.ml_service, statusRecord?.mlService, statusRecord?.model]],
  ];

  return serviceSources.map(([label, candidates]) => {
    const candidate = candidates.find((value) => value !== undefined);
    return normalizeService(label, candidate);
  });
}

export async function getSession(): Promise<SessionUser | null> {
  const paths = ["/api/auth/session", "/api/session", "/api/me", "/api/auth/me"];

  for (const path of paths) {
    try {
      const payload = await request<unknown>(path);
      return normalizeSession(payload);
    } catch (error) {
      if (error instanceof ApiError && [401, 403, 404].includes(error.status)) {
        continue;
      }
      throw error;
    }
  }

  return null;
}

export async function signup(username: string, password: string): Promise<SessionUser> {
  const payload = await requestWithFallback<unknown>(["/api/auth/signup", "/api/signup"], {
    method: "POST",
    data: { username, password },
  });

  return normalizeSession(payload) ?? (await getSession()) ?? { username };
}

export async function login(username: string, password: string): Promise<SessionUser> {
  const payload = await requestWithFallback<unknown>(["/api/auth/login", "/api/login"], {
    method: "POST",
    data: { username, password },
  });

  return normalizeSession(payload) ?? (await getSession()) ?? { username };
}

export async function logout(): Promise<void> {
  await requestWithFallback<unknown>(["/api/auth/logout", "/api/logout"], {
    method: "POST",
  });
}

export async function predictTags(input: { title: string; body: string }): Promise<PredictionResult> {
  const combinedInput = [input.title.trim(), input.body.trim()].filter(Boolean).join("\n\n");
  const payload = await request<unknown>("/api/predict", {
    method: "POST",
    data: {
      title: input.title,
      body: input.body,
      text: combinedInput,
      inputs: combinedInput,
    },
  });

  return {
    tags: extractTags(payload),
    message: deriveMessage(payload, ""),
  };
}

export async function getDashboardData(): Promise<DashboardData> {
  const [statusResult, healthResult] = await Promise.allSettled([
    requestWithFallback<unknown>(["/api/status", "/api/status/overview"]),
    request<unknown>("/api/health"),
  ]);

  const statusPayload = statusResult.status === "fulfilled" ? statusResult.value : null;
  const healthPayload = healthResult.status === "fulfilled" ? healthResult.value : null;

  if (!statusPayload && !healthPayload) {
    const statusError = statusResult.status === "rejected" ? statusResult.reason : null;
    const healthError = healthResult.status === "rejected" ? healthResult.reason : null;
    throw (statusError instanceof Error ? statusError : healthError instanceof Error ? healthError : new ApiError("Unable to load system status."));
  }

  return {
    ...extractMetrics(statusPayload),
    recentRequests: extractRecentRequests(statusPayload),
    services: extractServices(statusPayload, healthPayload),
  };
}

export function isServiceUnavailableError(error: unknown): boolean {
  if (!(error instanceof ApiError)) {
    return false;
  }

  if (error.status === 503) {
    return true;
  }

  const message = `${error.message} ${JSON.stringify(error.payload)}`.toLowerCase();
  return /(ml|model|inference|predict)/.test(message) && /(unavailable|down|degraded|failed|not ready|offline)/.test(message);
}
