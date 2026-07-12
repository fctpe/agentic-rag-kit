import type { Session } from "@/lib/types";

const STORAGE_KEY = "agentic-rag-session";
const SESSION_EVENT = "session-change";

function parseSession(raw: string | null): Session | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Session;
    return parsed.token ? parsed : null;
  } catch {
    return null;
  }
}

export function saveSession(session: Session): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  window.dispatchEvent(new Event(SESSION_EVENT));
}

export function clearSession(): void {
  window.localStorage.removeItem(STORAGE_KEY);
  window.dispatchEvent(new Event(SESSION_EVENT));
}

export function canReview(role: string): boolean {
  return role === "analyst" || role === "admin";
}

// useSyncExternalStore plumbing: getSnapshot must be referentially stable,
// so cache the parsed session keyed by the raw storage string.
let cachedRaw: string | null | undefined;
let cachedSession: Session | null = null;

export function sessionSnapshot(): Session | null {
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (raw !== cachedRaw) {
    cachedRaw = raw;
    cachedSession = parseSession(raw);
  }
  return cachedSession;
}

export function subscribeToSession(callback: () => void): () => void {
  window.addEventListener("storage", callback);
  window.addEventListener(SESSION_EVENT, callback);
  return () => {
    window.removeEventListener("storage", callback);
    window.removeEventListener(SESSION_EVENT, callback);
  };
}
