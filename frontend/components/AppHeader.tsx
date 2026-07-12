"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";

import { useSession } from "@/hooks/useSession";
import { canReview, clearSession } from "@/lib/auth";

const NAV = [
  { href: "/", label: "Chat" },
  { href: "/approvals", label: "Approvals", reviewOnly: true },
  { href: "/search", label: "Search", reviewOnly: true },
] as const;

export default function AppHeader() {
  const pathname = usePathname();
  const router = useRouter();
  const session = useSession();

  if (pathname === "/login") return null;

  const signOut = () => {
    clearSession();
    router.replace("/login");
  };

  return (
    <header className="sticky top-0 z-10 border-b border-slate-200 bg-white">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-md bg-blue-700 text-xs font-bold text-white">
              CA
            </span>
            <span className="text-sm font-semibold text-slate-900">
              Compliance Assistant
            </span>
          </Link>
          <nav className="flex items-center gap-1">
            {NAV.map((item) => {
              if (
                "reviewOnly" in item &&
                item.reviewOnly &&
                (!session || !canReview(session.role))
              ) {
                return null;
              }
              const active = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`rounded-md px-2.5 py-1.5 text-sm ${
                    active
                      ? "bg-slate-100 font-medium text-slate-900"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
        {session && (
          <div className="flex items-center gap-3">
            <span className="hidden text-xs text-slate-500 sm:inline">
              {session.email}
              <span className="ml-1.5 rounded bg-slate-100 px-1.5 py-0.5 font-medium uppercase text-slate-600">
                {session.role}
              </span>
            </span>
            <button
              type="button"
              onClick={signOut}
              className="rounded-md border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50"
            >
              Sign out
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
