export function LoadingSkeleton() {
  return (
    <div className="space-y-4 animate-slide-up">
      <div className="rounded-lg border border-border bg-card p-5">
        <div className="h-3 w-24 rounded bg-muted animate-shimmer bg-gradient-to-r from-muted via-surface-elevated to-muted bg-[length:200%_100%] mb-3" />
        <div className="space-y-2">
          <div className="h-3 w-full rounded bg-muted animate-shimmer bg-gradient-to-r from-muted via-surface-elevated to-muted bg-[length:200%_100%]" />
          <div className="h-3 w-3/4 rounded bg-muted animate-shimmer bg-gradient-to-r from-muted via-surface-elevated to-muted bg-[length:200%_100%]" />
        </div>
      </div>
      <div className="rounded-lg border border-border bg-card p-6 flex justify-center">
        <div className="h-12 w-20 rounded bg-muted animate-shimmer bg-gradient-to-r from-muted via-surface-elevated to-muted bg-[length:200%_100%]" />
      </div>
      {[1, 2].map((i) => (
        <div key={i} className="rounded-lg border border-border bg-card p-4">
          <div className="h-3 w-48 rounded bg-muted animate-shimmer bg-gradient-to-r from-muted via-surface-elevated to-muted bg-[length:200%_100%] mb-2" />
          <div className="h-3 w-full rounded bg-muted animate-shimmer bg-gradient-to-r from-muted via-surface-elevated to-muted bg-[length:200%_100%]" />
        </div>
      ))}
    </div>
  );
}
