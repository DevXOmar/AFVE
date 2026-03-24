interface ScoreCardProps {
  score: number;
  label: string;
}

function getScoreColor(score: number): string {
  if (score >= 0.75) return "text-score-green";
  if (score >= 0.4) return "text-score-yellow";
  return "text-score-red";
}

function getScoreBg(score: number): string {
  if (score >= 0.75) return "bg-score-green/10 border-score-green/20";
  if (score >= 0.4) return "bg-score-yellow/10 border-score-yellow/20";
  return "bg-score-red/10 border-score-red/20";
}

export function ScoreCard({ score, label }: ScoreCardProps) {
  return (
    <div className={`rounded-lg border p-6 text-center animate-slide-up ${getScoreBg(score)}`}>
      <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
        Factuality Score
      </p>
      <p className={`text-5xl font-bold font-mono ${getScoreColor(score)}`}>
        {score.toFixed(2)}
      </p>
      <p className={`mt-2 text-sm font-medium ${getScoreColor(score)}`}>{label}</p>
    </div>
  );
}
