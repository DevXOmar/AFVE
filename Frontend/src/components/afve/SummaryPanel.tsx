import type { Claim } from "@/lib/mock";

interface SummaryPanelProps {
  claims: Claim[];
}

export function SummaryPanel({ claims }: SummaryPanelProps) {
  const verified = claims.filter((c) => c.score >= 0.75).length;
  const uncertain = claims.filter((c) => c.score >= 0.4 && c.score < 0.75).length;
  const unsupported = claims.filter((c) => c.score < 0.4).length;

  const items = [
    { icon: "✔", label: "Verified", count: verified, color: "text-score-green" },
    { icon: "⚠", label: "Uncertain", count: uncertain, color: "text-score-yellow" },
    { icon: "❌", label: "Unsupported", count: unsupported, color: "text-score-red" },
  ];

  return (
    <div className="grid grid-cols-3 gap-3 animate-slide-up">
      {items.map((item) => (
        <div key={item.label} className="rounded-lg border border-border bg-card p-4 text-center">
          <span className={`text-2xl ${item.color}`}>{item.icon}</span>
          <p className="text-2xl font-bold font-mono mt-1 text-foreground">{item.count}</p>
          <p className="text-xs text-muted-foreground mt-0.5">{item.label}</p>
        </div>
      ))}
    </div>
  );
}
