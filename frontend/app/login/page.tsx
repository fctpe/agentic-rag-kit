"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";

import { ApiError, login } from "@/lib/api";
import { saveSession } from "@/lib/auth";

const DEMO_ACCOUNTS = [
  { email: "viewer@example.com", role: "viewer" },
  { email: "analyst@example.com", role: "analyst" },
  { email: "admin@example.com", role: "admin" },
];

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const session = await login(email, password);
      saveSession(session);
      router.replace("/");
    } catch (err) {
      setError(
        err instanceof ApiError && err.status === 401
          ? "Invalid credentials"
          : "Login failed. Is the backend running?",
      );
      setBusy(false);
    }
  };

  return (
    <main className="flex flex-1 items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <div className="mb-6 text-center">
          <span className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-blue-700 text-sm font-bold text-white">
            CA
          </span>
          <h1 className="text-xl font-semibold text-slate-900">
            Compliance Assistant
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            EU AI Act & GDPR research with cited answers
          </p>
        </div>
        <form
          onSubmit={submit}
          className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm"
        >
          <label className="block text-sm font-medium text-slate-700">
            Email
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
              placeholder="analyst@example.com"
            />
          </label>
          <label className="mt-4 block text-sm font-medium text-slate-700">
            Password
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
              placeholder="••••••••"
            />
          </label>
          {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
          <button
            type="submit"
            disabled={busy}
            className="mt-5 w-full rounded-md bg-blue-700 py-2 text-sm font-medium text-white hover:bg-blue-800 disabled:opacity-50"
          >
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>
        <div className="mt-4 rounded-lg border border-slate-200 bg-white p-4 text-xs text-slate-500">
          <p className="font-medium text-slate-600">Demo accounts</p>
          <ul className="mt-1.5 space-y-1">
            {DEMO_ACCOUNTS.map((account) => (
              <li key={account.email} className="flex justify-between">
                <button
                  type="button"
                  onClick={() => {
                    setEmail(account.email);
                    setPassword("demo1234");
                  }}
                  className="font-mono text-blue-700 hover:underline"
                >
                  {account.email}
                </button>
                <span className="uppercase">{account.role}</span>
              </li>
            ))}
          </ul>
          <p className="mt-1.5">
            Password for all: <span className="font-mono">demo1234</span>
          </p>
        </div>
      </div>
    </main>
  );
}
