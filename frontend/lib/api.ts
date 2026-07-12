import type { Citation, PendingApproval, Session } from "@/lib/types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function requestJson<T>(
  path: string,
  init: RequestInit = {},
  token?: string,
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...init.headers,
    },
  });
  if (!res.ok) {
    const body = await res.text();
    let detail = body;
    try {
      detail = (JSON.parse(body) as { detail?: string }).detail ?? body;
    } catch {
      // keep raw body
    }
    throw new ApiError(res.status, detail || res.statusText);
  }
  return (await res.json()) as T;
}

export function login(email: string, password: string): Promise<Session> {
  return requestJson<Session>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function searchCorpus(
  token: string,
  q: string,
  regulation?: string,
): Promise<Citation[]> {
  const params = new URLSearchParams({ q });
  if (regulation) params.set("regulation", regulation);
  const data = await requestJson<{ results: Citation[] }>(
    `/search?${params}`,
    {},
    token,
  );
  return data.results;
}

export async function fetchApprovals(token: string): Promise<PendingApproval[]> {
  const data = await requestJson<{ pending: PendingApproval[] }>(
    "/approvals",
    {},
    token,
  );
  return data.pending;
}

function streamRequest(
  path: string,
  body: unknown,
  token: string,
  signal?: AbortSignal,
): Promise<Response> {
  return fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal,
  });
}

export function streamChat(
  token: string,
  message: string,
  threadId: string | null,
  signal?: AbortSignal,
): Promise<Response> {
  return streamRequest(
    "/chat",
    { message, ...(threadId ? { thread_id: threadId } : {}) },
    token,
    signal,
  );
}

export function streamResume(
  token: string,
  threadId: string,
  status: "approved" | "rejected",
  comment: string,
  signal?: AbortSignal,
): Promise<Response> {
  return streamRequest(
    `/chat/${threadId}/resume`,
    { status, comment },
    token,
    signal,
  );
}
