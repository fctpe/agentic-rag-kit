"use client";

import { useSyncExternalStore } from "react";

import { sessionSnapshot, subscribeToSession } from "@/lib/auth";
import type { Session } from "@/lib/types";

const serverSnapshot = () => undefined;

/**
 * Current auth session. `undefined` while rendering on the server (unknown),
 * `null` when signed out, a `Session` when signed in.
 */
export function useSession(): Session | null | undefined {
  return useSyncExternalStore(subscribeToSession, sessionSnapshot, serverSnapshot);
}
