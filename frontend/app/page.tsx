"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useCallback, useEffect, useRef, useState } from "react";

import ApprovalBanner from "@/components/ApprovalBanner";
import ChatMessage from "@/components/ChatMessage";
import CitationPanel from "@/components/CitationPanel";
import { useChatStream } from "@/hooks/useChatStream";
import { useSession } from "@/hooks/useSession";
import { canReview, clearSession } from "@/lib/auth";

const SAMPLE_PROMPTS = [
  "What obligations apply to providers of high-risk AI systems?",
  "When is a data protection impact assessment required under GDPR?",
  "Draft a compliance gap analysis memo for an AI chatbot that processes personal data.",
];

export default function ChatPage() {
  const router = useRouter();
  const session = useSession();
  const [input, setInput] = useState("");
  const [active, setActive] = useState<{ id: string; index: number } | null>(
    null,
  );
  const [panelCollapsed, setPanelCollapsed] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (session === null) router.replace("/login");
  }, [session, router]);

  const onUnauthorized = useCallback(() => {
    clearSession();
    router.replace("/login");
  }, [router]);

  const { messages, approval, streaming, error, send, resume, reset } =
    useChatStream({ token: session?.token ?? "", onUnauthorized });

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, approval]);

  if (!session) return null;

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || streaming || approval) return;
    setInput("");
    void send(text);
  };

  const selectCitation = (id: string, index: number) => {
    setActive({ id, index });
    setPanelCollapsed(false);
  };

  const activeMessage = active
    ? messages.find((m) => m.id === active.id)
    : undefined;
  const panelCitations =
    activeMessage?.citations ??
    [...messages].reverse().find((m) => m.citations?.length)?.citations ??
    [];

  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 gap-4 px-4 py-4">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="mb-3 flex items-center justify-between">
          <h1 className="text-sm font-semibold text-slate-700">
            Regulatory research chat
          </h1>
          <button
            type="button"
            onClick={() => {
              reset();
              setActive(null);
            }}
            className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
          >
            New conversation
          </button>
        </div>

        <div className="flex flex-1 flex-col gap-4">
          {messages.length === 0 && !approval && (
            <div className="mt-16 text-center">
              <h2 className="text-lg font-semibold text-slate-800">
                Ask about the EU AI Act or GDPR
              </h2>
              <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">
                Answers are grounded in the regulation text with inline
                citations. Draft reports require human approval before release.
              </p>
              <div className="mx-auto mt-5 flex max-w-lg flex-col gap-2">
                {SAMPLE_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => setInput(prompt)}
                    className="rounded-lg border border-slate-200 bg-white px-4 py-2.5 text-left text-sm text-slate-600 hover:border-blue-300 hover:text-slate-900"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message}
              onCitationClick={(index) => selectCitation(message.id, index)}
            />
          ))}

          {approval && (
            <div>
              <ApprovalBanner
                approval={approval}
                disabled={streaming || !canReview(session.role)}
                onDecide={(status, comment) => void resume(status, comment)}
              />
              {!canReview(session.role) && (
                <p className="mt-1.5 text-xs text-slate-500">
                  Your role ({session.role}) cannot decide approvals — an
                  analyst or admin must review this draft.
                </p>
              )}
            </div>
          )}

          {error && (
            <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <form
          onSubmit={submit}
          className="sticky bottom-0 mt-4 bg-slate-50 pb-4 pt-2"
        >
          <div className="flex items-end gap-2 rounded-xl border border-slate-300 bg-white p-2 shadow-sm focus-within:border-blue-400">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  e.currentTarget.form?.requestSubmit();
                }
              }}
              rows={2}
              placeholder={
                approval
                  ? "Resolve the pending approval to continue…"
                  : "Ask a compliance question…"
              }
              disabled={!!approval}
              className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1 text-sm focus:outline-none disabled:text-slate-400"
            />
            <button
              type="submit"
              disabled={streaming || !!approval || !input.trim()}
              className="rounded-lg bg-blue-700 px-4 py-2 text-sm font-medium text-white hover:bg-blue-800 disabled:opacity-40"
            >
              {streaming ? "Streaming…" : "Send"}
            </button>
          </div>
        </form>
      </div>

      <CitationPanel
        citations={panelCitations}
        activeIndex={active?.index ?? null}
        collapsed={panelCollapsed}
        onToggle={setPanelCollapsed}
      />
    </main>
  );
}
