import type { Regulation } from "@/lib/types";

const STYLES: Record<Regulation, { label: string; className: string }> = {
  ai_act: { label: "AI Act", className: "bg-blue-100 text-blue-800" },
  gdpr: { label: "GDPR", className: "bg-emerald-100 text-emerald-800" },
};

export default function RegulationBadge({
  regulation,
}: {
  regulation: Regulation;
}) {
  const style = STYLES[regulation] ?? {
    label: regulation,
    className: "bg-slate-100 text-slate-700",
  };
  return (
    <span
      className={`inline-block rounded px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${style.className}`}
    >
      {style.label}
    </span>
  );
}
