import type { Claim } from "@/lib/mock";

interface ClaimCardProps {
  claim: Claim;
  index: number;
}

function getLabelIcon(label: string) {
  if (label === "Verified") return "✢";
  if (label === "Hallucinated") return "✎";
  return "⚠";
}

function getLabelColor(score: number) {
  if (score >= 0.75) return "text-score-green";
  if (score >= 0.4) return "text-score-yellow";
  return "text-score-red";
}

export function ClaimCard({ claim, index }: ClaimCardProps) {
  return (
    <div
      className="rounded-lg border border-border bg-card p-4 animate-slide-up"
      style={{ animationDelay: `${index * 80}ms` }}
    >
      <div className="flex items-start gap-3">
        <span className={`text-lg mt-0.5 ${getLabelColor(claim.score)}`}>
          {getLabelIcon(claim.label)}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center space-x-4 mb-3">
            <div className="text-sm font-semibold">
              Factuality: {Math.round(claim.score * 100)}%
            </div>
          </div>
          
          <p className="text-sm text-foreground font-medium mb-1.5">{claim.claim}</p>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {claim.explanation}
          </p>
        </div>
      </div>
    </div>
  );
}
