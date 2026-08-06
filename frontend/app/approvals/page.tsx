"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import Markdown from "@/components/Markdown";
import RegulationBadge from "@/components/RegulationBadge";
import { useSession } from "@/hooks/useSession";
import { ApiError, fetchApprovals, streamResume } from "@/lib/api";
import { canReview, clearSession } from "@/lib/auth";
import { parseSSEStream } from "@/lib/sse";
import type { PendingApproval } from "@/lib/types";

export default function ApprovalsPage() {
  const router = useRouter();
  const session = useSession();
  const [pending, setPending] = useState<PendingApproval[] | null>(null);
  const [comments, setComments] = useState<Record<string, string>>({});
  const [decidingId, setDecidingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const signOut = useCallback(() => {
    clearSession();
    router.replace("/login");
  }, [router]);

  useEffect(() => {
    if (session === null) {
      router.replace("/login");
      return;
    }
    if (!session) return;
    let cancelled = false;
    fetchApprovals(session.token)
      .then((items) => {
        if (!cancelled) setPending(items);
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 401) signOut();
        else if (err instanceof ApiError && err.status === 403) {
          setError("Your role cannot view approvals (analyst or admin required).");
        } else {
          setError(err instanceof Error ? err.message : String(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [session, router, signOut]);

  const decide = async (
    approval: PendingApproval,
    status: "approved" | "rejected",
  ) => {
    if (!session) return;
    setDecidingId(approval.id);
    setError(null);
    try {
      const response = await streamResume(
        session.token,
        approval.thread_id,
        status,
        comments[approval.id] ?? "",
      );
      if (response.status === 401) {
        signOut();
        return;
      }
      if (!response.ok || !response.body) {
        throw new Error(`Resume failed with status ${response.status}`);
      }
      // Drain the continuation stream so the run completes server-side.
      for await (const event of parseSSEStream(response.body)) {
        if (event.event === "error") {
          // The server deliberately no longer sends exception text — it can
          // carry bound query text and personal data. The request id is what
          // makes the failure findable in the logs, so it has to survive to
          // the surface or the generic message is a dead end.
          const data = JSON.parse(event.data) as { message: string; request_id?: string };
          throw new Error(
            data.request_id ? `${data.message} (${data.request_id})` : data.message,
          );
        }
      }
      setPending((prev) =>
        prev ? prev.filter((item) => item.id !== approval.id) : prev,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setDecidingId(null);
    }
  };

  if (!session) return null;
  const reviewer = canReview(session.role);

  return (
    <main className="mx-auto w-full max-w-4xl flex-1 px-4 py-6">
      <h1 className="text-lg font-semibold text-slate-900">
        Pending approvals
      </h1>
      <p className="mt-1 text-sm text-slate-500">
        Draft reports interrupted at the approval gate. Approving releases the
        answer to the requester&apos;s conversation.
      </p>

      {error && (
        <div className="mt-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </div>
      )}

      {pending && pending.length === 0 && (
        <div className="mt-8 rounded-lg border border-dashed border-slate-300 bg-white p-8 text-center text-sm text-slate-500">
          No approvals waiting for review.
        </div>
      )}

      <ul className="mt-4 space-y-4">
        {pending?.map((approval) => (
          <li
            key={approval.id}
            className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm"
          >
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500">
              <span className="font-mono">thread {approval.thread_id}</span>
              <span>
                requested {new Date(approval.requested_at).toLocaleString()}
              </span>
            </div>
            {(approval.payload.citations?.length ?? 0) > 0 && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {approval.payload.citations?.map((citation) => (
                  <span
                    key={citation.index}
                    className="inline-flex items-center gap-1 rounded border border-slate-200 px-1.5 py-0.5 text-[11px] text-slate-600"
                  >
                    <RegulationBadge regulation={citation.regulation} />
                    {citation.article}
                  </span>
                ))}
              </div>
            )}
            {(approval.payload.citation_issues?.length ?? 0) > 0 && (
              <ul className="mt-2 list-disc space-y-0.5 rounded-md border border-amber-200 bg-amber-50 px-5 py-2 text-xs text-amber-900">
                {/* Position, not text: one answer can carry the same
                    unresolvable marker twice, and duplicate keys make React
                    drop or duplicate a row. */}
                {approval.payload.citation_issues?.map((issue, i) => (
                  <li key={i}>{issue}</li>
                ))}
              </ul>
            )}
            <div className="mt-3 max-h-72 overflow-y-auto rounded-md border border-slate-200 bg-slate-50 p-3 text-sm">
              <Markdown content={approval.payload.draft ?? ""} />
            </div>
            <textarea
              value={comments[approval.id] ?? ""}
              onChange={(e) =>
                setComments((prev) => ({
                  ...prev,
                  [approval.id]: e.target.value,
                }))
              }
              placeholder="Optional comment…"
              rows={2}
              className="mt-3 w-full resize-none rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-400 focus:outline-none"
            />
            <div className="mt-2 flex gap-2">
              <button
                type="button"
                disabled={!reviewer || decidingId !== null}
                onClick={() => void decide(approval, "approved")}
                className="rounded-md bg-blue-700 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-800 disabled:opacity-50"
              >
                {decidingId === approval.id ? "Submitting…" : "Approve"}
              </button>
              <button
                type="button"
                disabled={!reviewer || decidingId !== null}
                onClick={() => void decide(approval, "rejected")}
                className="rounded-md border border-slate-300 bg-white px-4 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
              >
                Reject
              </button>
            </div>
          </li>
        ))}
      </ul>
    </main>
  );
}
