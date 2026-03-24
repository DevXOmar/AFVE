import { useState } from "react";
import type { Claim } from "@/lib/mock";

interface ClaimCardProps {
  claim: Claim;
  index: number;
}

function getLabelIcon(label: string) {
  if (label === "Verified") return "✔";
  if (label === "Hallucinated") return "❌";
  return "⚠";
}

function getLabelColor(score: number) {
  if (score >= 0.75) return "text-score-green";
  if (score >= 0.4) return "text-score-yellow";
  return "text-score-red";
}

function getBadgeBg(score: number) {
  if (score >= 0.75) return "bg-score-green/15 text-score-green";
  if (score >= 0.4) return "bg-score-yellow/15 text-score-yellow";
  return "bg-score-red/15 text-score-red";
}

export function ClaimCard({ claim, index }: ClaimCardProps) {
  const [expanded, setExpanded] = useState(false);

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
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${getBadgeBg(claim.score)}`}>
              {claim.label} · {claim.score.toFixed(2)}
            </span>
          </div>
          <p className="text-sm text-foreground font-medium">{claim.claim}</p>
          <p className="text-xs text-muted-foreground mt-1.5 leading-relaxed">
            {claim.explanation}
          </p>

          {claim.evidence.length > 0 && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="mt-2 text-xs text-primary hover:underline"
            >
              {expanded ? "Hide evidence" : `Show evidence (${claim.evidence.length})`}
            </button>
          )}

          {expanded && (
            <div className="mt-2 space-y-2">
              {claim.evidence.map((ev, i) => (
                <div key={i} className="rounded-md bg-muted p-3 text-xs">
                  <a
                    href={ev.source_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline break-all"
                  >
                    {ev.source_url}
                  </a>
                  <p className="mt-1 text-muted-foreground italic">"{ev.snippet}"</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
