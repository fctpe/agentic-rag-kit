"use client";

import { useEffect, useRef } from "react";

import CitationCard from "@/components/CitationCard";
import type { Citation } from "@/lib/types";

interface CitationPanelProps {
  citations: Citation[];
  activeIndex: number | null;
  collapsed: boolean;
  onToggle: (collapsed: boolean) => void;
}

export default function CitationPanel({
  citations,
  activeIndex,
  collapsed,
  onToggle,
}: CitationPanelProps) {
  const itemRefs = useRef(new Map<number, HTMLLIElement>());

  useEffect(() => {
    if (activeIndex === null) return;
    itemRefs.current
      .get(activeIndex)
      ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [activeIndex]);

  if (citations.length === 0) return null;

  if (collapsed) {
    return (
      <button
        type="button"
        onClick={() => onToggle(false)}
        className="sticky top-20 h-fit rounded-lg border border-slate-200 bg-white px-2 py-3 text-xs font-medium text-slate-600 hover:bg-slate-50"
        title="Show citations"
      >
        <span className="[writing-mode:vertical-rl]">
          Citations ({citations.length})
        </span>
      </button>
    );
  }

  return (
    <aside className="sticky top-20 flex h-fit max-h-[calc(100vh-6rem)] w-80 shrink-0 flex-col rounded-lg border border-slate-200 bg-slate-50">
      <div className="flex items-center justify-between border-b border-slate-200 px-3 py-2">
        <h2 className="text-sm font-semibold text-slate-700">
          Citations ({citations.length})
        </h2>
        <button
          type="button"
          onClick={() => onToggle(true)}
          className="rounded px-1.5 py-0.5 text-xs text-slate-500 hover:bg-slate-200"
          title="Collapse panel"
        >
          Hide
        </button>
      </div>
      <ul className="flex flex-col gap-2 overflow-y-auto p-3">
        {citations.map((citation) => (
          <li
            key={citation.index}
            ref={(el) => {
              if (el) itemRefs.current.set(citation.index, el);
              else itemRefs.current.delete(citation.index);
            }}
          >
            <CitationCard
              citation={citation}
              active={citation.index === activeIndex}
            />
          </li>
        ))}
      </ul>
    </aside>
  );
}
