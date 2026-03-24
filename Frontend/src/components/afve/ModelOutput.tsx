import { useState } from "react";

interface ModelOutputProps {
  output: string;
}

export function ModelOutput({ output }: ModelOutputProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(output);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="rounded-lg border border-border bg-card p-5 animate-slide-up">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Model Output
        </h2>
        <button
          onClick={handleCopy}
          className="text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {copied ? "✓ Copied" : "Copy"}
        </button>
      </div>
      <p className="text-foreground/90 leading-relaxed">{output}</p>
    </div>
  );
}
