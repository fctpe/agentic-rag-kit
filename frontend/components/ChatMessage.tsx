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
        {message.content ? (
          <Markdown
            content={message.content}
            onCitationClick={onCitationClick}
          />
        ) : (
          <span className="text-slate-400">Thinking…</span>
        )}
        {message.streaming && message.content && (
          <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-slate-400 align-baseline" />
        )}
      </div>
    </div>
  );
}
