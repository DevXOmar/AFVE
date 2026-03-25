import React from "react";
import { CheckCircle, AlertTriangle, XCircle } from "lucide-react";
import { EvidenceAccordion } from "./EvidenceAccordion";

interface Claim {
  claim?: string;
  claim_text?: string;
  label: "VERIFIED" | "UNCERTAIN" | "UNSUPPORTED" | "CONTRADICTED";
  confidence_score?: number;
  evidence: Array<{ text?: string; snippet?: string; source?: string; url?: string; source_url?: string; score?: number }>;
}

export function ClaimCard({ claim }: { claim: Claim }) {
  const isSupported = claim.label === "VERIFIED";
  const isInconclusive = claim.label === "UNCERTAIN" || claim.label === "UNSUPPORTED";
  const isContradicted = claim.label === "CONTRADICTED";

  return (
    <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-4 overflow-hidden transform transition-all duration-300 translate-y-0 opacity-100">
      <div className="flex items-start justify-between gap-3 mb-3">
        <p className="text-slate-800 text-sm font-medium leading-normal flex-1">
          {claim.claim_text || claim.claim}
        </p>

        <div className="flex flex-col items-end shrink-0 gap-1 mt-0.5">
          {isSupported && (
            <span className="flex items-center gap-1 text-xs font-semibold text-emerald-700 bg-emerald-50 px-2 py-1 rounded-full border border-emerald-100">
              <CheckCircle className="w-3 h-3" /> Supported
            </span>
          )}
          {isInconclusive && (
            <span className="flex items-center gap-1 text-xs font-semibold text-amber-700 bg-amber-50 px-2 py-1 rounded-full border border-amber-100">
              <AlertTriangle className="w-3 h-3" /> Uncertain
            </span>
          )}
          {isContradicted && (
            <span className="flex items-center gap-1 text-xs font-semibold text-rose-700 bg-rose-50 px-2 py-1 rounded-full border border-rose-100">
              <XCircle className="w-3 h-3" /> Contradicted
            </span>
          )}
          
          {claim.confidence_score !== undefined && (
            <span className="text-[10px] text-slate-400 font-medium">
              Conf: {(claim.confidence_score).toFixed(2)}
            </span>
          )}
        </div>
      </div>
      
      <EvidenceAccordion evidence={claim.evidence} />
    </div>
  );
}
