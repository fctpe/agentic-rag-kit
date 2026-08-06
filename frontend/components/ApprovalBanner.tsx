"use client";

import { useState } from "react";

import Markdown from "@/components/Markdown";
import type { ApprovalRequest } from "@/lib/types";

interface ApprovalBannerProps {
  approval: ApprovalRequest;
  disabled?: boolean;
  onDecide: (status: "approved" | "rejected", comment: string) => void;
}

export default function ApprovalBanner({
  approval,
  disabled,
  onDecide,
}: ApprovalBannerProps) {
  const [comment, setComment] = useState("");

  return (
    <div className="rounded-lg border border-amber-300 bg-amber-50 p-4">
      <div className="flex items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-amber-500" />
        <h2 className="text-sm font-semibold text-amber-900">
          Approval required
        </h2>
      </div>
      <p className="mt-1 text-xs text-amber-800">
        The assistant drafted a compliance report. Review it before it is
        released.
      </p>
      {approval.citationIssues.length > 0 && (
        <ul className="mt-2 list-disc space-y-0.5 rounded-md border border-amber-200 bg-white px-5 py-2 text-xs text-amber-900">
          {/* Keyed by position, not by text: the same issue legitimately
              repeats when one answer carries the same unresolvable marker
              twice, and two identical keys make React drop or duplicate a
              row. The list is rendered once per approval and never reordered,
              so the index is stable for its lifetime. */}
          {approval.citationIssues.map((issue, i) => (
            <li key={i}>{issue}</li>
          ))}
        </ul>
      )}
      <div className="mt-3 max-h-64 overflow-y-auto rounded-md border border-amber-200 bg-white p-3 text-sm">
        <Markdown content={approval.draft} />
      </div>
      <textarea
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        placeholder="Optional comment for the audit trail…"
        rows={2}
        className="mt-3 w-full resize-none rounded-md border border-amber-200 bg-white px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 focus:border-amber-400 focus:outline-none"
      />
      <div className="mt-2 flex gap-2">
        <button
          type="button"
          disabled={disabled}
          onClick={() => onDecide("approved", comment)}
          className="rounded-md bg-blue-700 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-800 disabled:opacity-50"
        >
          Approve
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => onDecide("rejected", comment)}
          className="rounded-md border border-slate-300 bg-white px-4 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
        >
          Reject
        </button>
      </div>
    </div>
  );
}
