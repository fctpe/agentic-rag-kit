"use client";

import { useCallback, useRef, useState } from "react";

import { streamChat, streamResume } from "@/lib/api";
import { parseSSEStream } from "@/lib/sse";
import type {
  ApprovalRequest,
  ChatMessageData,
  Citation,
  Grounding,
} from "@/lib/types";

interface ChatStreamOptions {
  token: string;
  onUnauthorized: () => void;
}

export function useChatStream({ token, onUnauthorized }: ChatStreamOptions) {
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [approval, setApproval] = useState<ApprovalRequest | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const patchMessage = useCallback(
    (id: string, patch: Partial<ChatMessageData>) => {
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, ...patch } : m)),
      );
    },
    [],
  );

  const consume = useCallback(
    async (response: Response) => {
      const headerThread = response.headers.get("X-Thread-Id");
      if (headerThread) setThreadId(headerThread);

      const assistantId = crypto.randomUUID();
      let text = "";
      setMessages((prev) => [
        ...prev,
        { id: assistantId, role: "assistant", content: "", streaming: true },
      ]);

      if (!response.body) throw new Error("Response has no body");
      for await (const event of parseSSEStream(response.body)) {
        const data = JSON.parse(event.data) as Record<string, unknown>;
        switch (event.event) {
          case "token":
            text += data.text as string;
            patchMessage(assistantId, { content: text });
            break;
          case "drafting":
            // Report run: the backend suppresses tokens, so nothing will fill
            // this bubble. Say what is happening rather than sitting on
            // "Thinking…" until the banner appears.
            patchMessage(assistantId, {
              content: "",
              drafting: true,
            });
            break;
          case "approval_required":
            // The stream ends here; the draft lives in the banner until a
            // decision is made. The bubble is empty by construction now — the
            // backend never streamed a report — but it is still dropped, since
            // the banner owns the draft.
            setMessages((prev) => prev.filter((m) => m.id !== assistantId));
            setApproval({
              draft: (data.draft as string) ?? "",
              citations: (data.citations as Citation[]) ?? [],
            });
            return;
          case "citations":
            patchMessage(assistantId, {
              citations: data.citations as Citation[],
            });
            break;
          case "grounding":
            patchMessage(assistantId, {
              grounding: {
                grounded: data.grounded as boolean,
                issues: (data.issues as string[]) ?? [],
              } satisfies Grounding,
            });
            break;
          case "done":
            setThreadId(data.thread_id as string);
            patchMessage(assistantId, {
              content: data.content as string,
              streaming: false,
            });
            return;
          case "error":
            setMessages((prev) => prev.filter((m) => m.id !== assistantId));
            setError(data.message as string);
            return;
        }
      }
      // Stream closed without a terminal event; keep whatever streamed.
      patchMessage(assistantId, { streaming: false });
    },
    [patchMessage],
  );

  const run = useCallback(
    async (start: (signal: AbortSignal) => Promise<Response>) => {
      const controller = new AbortController();
      abortRef.current?.abort();
      abortRef.current = controller;
      setStreaming(true);
      setError(null);
      try {
        const response = await start(controller.signal);
        if (response.status === 401) {
          onUnauthorized();
          return;
        }
        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`);
        }
        await consume(response);
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : String(err));
          setMessages((prev) => prev.filter((m) => !m.streaming));
        }
      } finally {
        if (abortRef.current === controller) {
          setStreaming(false);
          abortRef.current = null;
        }
      }
    },
    [consume, onUnauthorized],
  );

  const send = useCallback(
    (message: string) => {
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "user", content: message },
      ]);
      return run((signal) => streamChat(token, message, threadId, signal));
    },
    [run, threadId, token],
  );

  const resume = useCallback(
    (status: "approved" | "rejected", comment: string) => {
      if (!threadId) return Promise.resolve();
      setApproval(null);
      return run((signal) =>
        streamResume(token, threadId, status, comment, signal),
      );
    },
    [run, threadId, token],
  );

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setMessages([]);
    setThreadId(null);
    setApproval(null);
    setStreaming(false);
    setError(null);
  }, []);

  return { messages, threadId, approval, streaming, error, send, resume, reset };
}
