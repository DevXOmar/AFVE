import React, { useState } from "react";
import { ChevronDown, ChevronUp, Link as LinkIcon } from "lucide-react";

interface Evidence {
  text: string;
  source?: string;
  url?: string;
  score?: number;
}

interface EvidenceAccordionProps {
  evidence: Evidence[];
}

export function EvidenceAccordion({ evidence }: EvidenceAccordionProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!evidence || evidence.length === 0) return null;

  return (
    <div className="mt-3 border-t border-slate-100 pt-3">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center text-xs font-medium text-slate-500 hover:text-slate-800 transition-colors focus:outline-none"
      >
        {isOpen ? <ChevronUp className="w-3 h-3 mr-1" /> : <ChevronDown className="w-3 h-3 mr-1" />}
        {isOpen ? "Hide Evidence" : `View Evidence (${evidence.length})`}
      </button>

      {isOpen && (
        <div className="mt-2 space-y-2">
          {evidence.map((ev, i) => (
            <div key={i} className="bg-slate-50 p-3 rounded-md border border-slate-100 text-sm">
              <p className="text-slate-700 leading-relaxed mb-2">"{ev.text}"</p>
              
              {(ev.url || ev.source) && (
                <div className="flex items-center gap-1.5 text-xs text-blue-600 font-medium overflow-hidden">
                  <LinkIcon className="w-3 h-3 flex-shrink-0" />
                  <a href={ev.url || "#"} target="_blank" rel="noreferrer" className="truncate hover:underline">
                    {ev.url || ev.source}
                  </a>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
