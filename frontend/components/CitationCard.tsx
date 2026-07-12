import RegulationBadge from "@/components/RegulationBadge";
import type { Citation } from "@/lib/types";

interface CitationCardProps {
  citation: Citation;
  active?: boolean;
}

export default function CitationCard({ citation, active }: CitationCardProps) {
  return (
    <div
      className={`rounded-lg border bg-white p-3 text-sm transition-colors ${
        active ? "border-blue-400 ring-2 ring-blue-100" : "border-slate-200"
      }`}
    >
      <div className="flex items-center gap-2">
        <span className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-slate-100 text-[11px] font-semibold text-slate-600">
          {citation.index}
        </span>
        <RegulationBadge regulation={citation.regulation} />
        <span className="font-medium text-slate-700">{citation.article}</span>
      </div>
      <p className="mt-1.5 text-[13px] font-medium text-slate-800">
        {citation.heading}
      </p>
      <p className="mt-1 line-clamp-3 text-[13px] leading-snug text-slate-500">
        {citation.snippet}
      </p>
      <div className="mt-2 flex items-center justify-between">
        <a
          href={citation.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-[13px] font-medium text-blue-700 hover:underline"
        >
          View on EUR-Lex
        </a>
        <span className="text-[11px] tabular-nums text-slate-400">
          score {citation.score.toFixed(3)}
        </span>
      </div>
    </div>
  );
}
