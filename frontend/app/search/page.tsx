"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";

import CitationCard from "@/components/CitationCard";
import { useSession } from "@/hooks/useSession";
import { ApiError, searchCorpus } from "@/lib/api";
import { clearSession } from "@/lib/auth";
import type { Citation } from "@/lib/types";

const REGULATION_FILTERS = [
  { value: "", label: "All" },
  { value: "ai_act", label: "AI Act" },
  { value: "gdpr", label: "GDPR" },
];

export default function SearchPage() {
  const router = useRouter();
  const session = useSession();
  const [query, setQuery] = useState("");
  const [regulation, setRegulation] = useState("");
  const [results, setResults] = useState<Citation[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (session === null) router.replace("/login");
  }, [session, router]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!session || query.trim().length < 2) return;
    setBusy(true);
    setError(null);
    try {
      setResults(
        await searchCorpus(session.token, query.trim(), regulation || undefined),
      );
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        clearSession();
        router.replace("/login");
        return;
      }
      setError(
        err instanceof ApiError && err.status === 403
          ? "Your role cannot search the corpus (analyst or admin required)."
          : err instanceof Error
            ? err.message
            : String(err),
      );
    } finally {
      setBusy(false);
    }
  };

  if (!session) return null;

  return (
    <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-6">
      <h1 className="text-lg font-semibold text-slate-900">Corpus search</h1>
      <p className="mt-1 text-sm text-slate-500">
        Hybrid search over the EU AI Act and GDPR text, ranked by relevance.
      </p>

      <form onSubmit={submit} className="mt-4 flex gap-2">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. biometric identification"
          className="flex-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm focus:border-blue-400 focus:outline-none"
        />
        <select
          value={regulation}
          onChange={(e) => setRegulation(e.target.value)}
          className="rounded-md border border-slate-300 bg-white px-2 py-2 text-sm text-slate-700 focus:outline-none"
        >
          {REGULATION_FILTERS.map((filter) => (
            <option key={filter.value} value={filter.value}>
              {filter.label}
            </option>
          ))}
        </select>
        <button
          type="submit"
          disabled={busy || query.trim().length < 2}
          className="rounded-md bg-blue-700 px-4 py-2 text-sm font-medium text-white hover:bg-blue-800 disabled:opacity-40"
        >
          {busy ? "Searching…" : "Search"}
        </button>
      </form>

      {error && (
        <div className="mt-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </div>
      )}

      {results && (
        <div className="mt-5 space-y-3">
          {results.length === 0 ? (
            <p className="text-sm text-slate-500">No passages matched.</p>
          ) : (
            results.map((citation) => (
              <CitationCard key={citation.index} citation={citation} />
            ))
          )}
        </div>
      )}
    </main>
  );
}
