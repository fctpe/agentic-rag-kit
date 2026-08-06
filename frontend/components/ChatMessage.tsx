"use client";

import Markdown from "@/components/Markdown";
import type { ChatMessageData } from "@/lib/types";

interface ChatMessageProps {
  message: ChatMessageData;
  onCitationClick?: (index: number) => void;
}

export default function ChatMessage({
  message,
  onCitationClick,
}: ChatMessageProps) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] whitespace-pre-wrap rounded-xl rounded-br-sm bg-slate-800 px-4 py-2.5 text-sm leading-relaxed text-white">
          {message.content}
        </div>
      </div>
    );
  }

  const grounding = message.grounding;
  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-xl rounded-bl-sm border border-slate-200 bg-white px-4 py-3 text-sm shadow-sm">
        {grounding && !grounding.grounded && (
          <div className="mb-2 inline-flex flex-wrap items-center gap-1.5 rounded-md border border-amber-300 bg-amber-50 px-2 py-1 text-xs text-amber-800">
            <span className="font-semibold">Not fully grounded</span>
            {grounding.issues.length > 0 && (
              <span>— {grounding.issues.join("; ")}</span>
            )}
          </div>
        )}
        {/*
          A qa answer streams before `verify` runs, so there is a window where
          text is on screen and no verdict exists. Rendering nothing during that
          window presented an unchecked answer exactly like a checked one — the
          fail-open the grounding node exists to prevent. Absent verdict is
          shown as absent, and stays shown if the run ends without one.
        */}
        {!grounding && message.content && (
          <div className="mb-2 inline-flex items-center gap-1.5 rounded-md border border-slate-300 bg-slate-50 px-2 py-1 text-xs text-slate-600">
            <span className="font-semibold">
              {message.streaming ? "Checking sources…" : "Unverified"}
            </span>
            {!message.streaming && (
              <span>— no grounding verdict was produced for this answer</span>
            )}
          </div>
        )}
        {message.content ? (
          <Markdown
            content={message.content}
            onCitationClick={onCitationClick}
          />
        ) : (
          <span className="text-slate-400">
            {message.drafting
              ? "Drafting a report — it is held for approval before you see it…"
              : "Thinking…"}
          </span>
        )}
        {message.citationIssues && message.citationIssues.length > 0 && (
          <ul className="mt-2 list-disc space-y-0.5 rounded-md border border-slate-300 bg-slate-50 px-5 py-2 text-xs text-slate-600">
            {/* Position, not text: one answer can carry the same unresolvable
                marker twice, and duplicate keys make React drop or duplicate
                a row. */}
            {message.citationIssues.map((issue, i) => (
              <li key={i}>{issue}</li>
            ))}
          </ul>
        )}
        {message.streaming && message.content && (
          <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-slate-400 align-baseline" />
        )}
      </div>
    </div>
  );
}
