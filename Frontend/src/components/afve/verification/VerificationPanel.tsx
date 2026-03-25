import React, { useEffect, useState } from "react";
import { X, Search, ExternalLink, CheckCircle, AlertTriangle, XCircle } from "lucide-react";
import { ClaimCard } from "./ClaimCard";

interface AFVEReport {
  claims: any[];
  global_evidence?: any[];
  overall_score?: number;
  overall_label?: string;
  factuality_percentage?: number;
  factuality_rating?: string;
}

interface VerificationPanelProps {
  isOpen: boolean;
  onClose: () => void;
  isLoading: boolean;
  report?: AFVEReport | null;
}

const LOADING_STEPS = [
  "Extracting claims...",
  "Classifying factual content...",
  "Retrieving sources...",
  "Cross-checking evidence...",
];

export function VerificationPanel({ isOpen, onClose, isLoading, report }: VerificationPanelProps) {
  const [loadingStep, setLoadingStep] = useState(0);

  useEffect(() => {
    if (!isLoading) {
      setLoadingStep(0);
      return;
    }

    const interval = setInterval(() => {
      setLoadingStep((prev) => (prev < LOADING_STEPS.length - 1 ? prev + 1 : prev));
    }, 1500);

    return () => clearInterval(interval);
  }, [isLoading]);

  if (!isOpen) return null;

  const hasClaims = report?.claims && report.claims.length > 0;
  
  const stats = hasClaims ? {
    supported: report.claims.filter(c => c.label === "VERIFIED").length,
    inconclusive: report.claims.filter(c => c.label === "UNCERTAIN" || c.label === "UNSUPPORTED").length,
    contradicted: report.claims.filter(c => c.label === "CONTRADICTED").length,
  } : { supported: 0, inconclusive: 0, contradicted: 0 };

  return (
    <>
      <div 
        className="fixed inset-0 bg-slate-900/20 backdrop-blur-sm z-40 transition-opacity duration-300"
        onClick={onClose}
      />

      <div className="fixed top-0 right-0 h-full w-[450px] max-w-full bg-white shadow-2xl z-50 flex flex-col transform transition-transform duration-300 translate-x-0 overflow-hidden border-l border-slate-200">
        <div className="flex items-center justify-between p-4 bg-slate-50 border-b border-slate-200 shrink-0">
          <div className="flex items-center gap-2 text-slate-800 font-semibold">
            <Search className="w-5 h-5 text-indigo-600" />
            Agentic Verification Report
          </div>
          <button 
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-slate-700 hover:bg-slate-200 rounded-md transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 bg-slate-50/50">
          {isLoading ? (
            <div className="h-full flex flex-col items-center justify-center text-center space-y-6">
              <div className="w-12 h-12 rounded-full border-4 border-indigo-100 border-t-indigo-600 animate-spin" />
              <div className="space-y-2">
                <p className="text-sm font-medium text-indigo-600 animate-pulse">
                  AFVE Agent is working
                </p>
                <p className="text-slate-500 text-sm">
                  {LOADING_STEPS[loadingStep]}
                </p>
              </div>
            </div>
          ) : !report ? (
            <div className="text-center text-slate-500 mt-10">
              No report available.
            </div>
          ) : !hasClaims ? (
            <div className="flex flex-col items-center justify-center p-8 bg-white rounded-xl border border-slate-200 shadow-sm text-center">
              <div className="w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center mb-4">
                <Search className="w-6 h-6 text-slate-400" />
              </div>
              <h3 className="text-slate-800 font-medium mb-2">No verifiable factual claims detected</h3>
              <p className="text-slate-500 text-sm leading-relaxed">
                This response is conversational or subjective and does not require verification.
              </p>
            </div>
          ) : (
            <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-500">
              <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Overall Factuality</div>
                <div className="flex items-end justify-between gap-4">
                  <div>
                    <div className="text-3xl font-bold text-slate-900">
                      {report.factuality_percentage !== undefined ? `${report.factuality_percentage}%` : `${((report.overall_score || 0) * 100).toFixed(1)}%`}
                    </div>
                    <div className="text-sm text-slate-500 mt-1">
                      {report.factuality_rating || report.overall_label || "Uncertain"}
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
                <div className="text-center flex-1 border-r border-slate-100">
                  <div className="text-2xl font-bold text-emerald-600">{stats.supported}</div>
                  <div className="text-xs font-medium text-slate-500 uppercase tracking-wider mt-1">Verified</div>
                </div>
                <div className="text-center flex-1 border-r border-slate-100">
                  <div className="text-2xl font-bold text-amber-500">{stats.inconclusive}</div>
                  <div className="text-xs font-medium text-slate-500 uppercase tracking-wider mt-1">Inconclusive</div>
                </div>
                <div className="text-center flex-1">
                  <div className="text-2xl font-bold text-rose-500">{stats.contradicted}</div>
                  <div className="text-xs font-medium text-slate-500 uppercase tracking-wider mt-1">Contradicted</div>
                </div>
              </div>

              <div className="space-y-3">
                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Extracted Claims</h4>
                {report.claims.map((claim, idx) => (
                  <div 
                    key={idx} 
                    className="opacity-0 animate-slide-up"
                    style={{ animationDelay: `${idx * 150}ms` }}
                  >
                    <ClaimCard claim={claim} />
                  </div>
                ))}
              </div>
              {report.global_evidence && report.global_evidence.length > 0 && (
                <div className="space-y-3 mt-8">
                  <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Top Evidence</h4>
                  <div className="space-y-3">
                    {report.global_evidence.map((ev, idx) => (
                      <div key={idx} className="bg-white border border-slate-200 rounded-lg p-3 text-sm flex gap-3 shadow-sm">
                        <div className="mt-0.5">
                          {ev.entailment_status === 'supports' ? (
                            <CheckCircle className="w-4 h-4 text-emerald-500" />
                          ) : ev.entailment_status === 'contradicts' ? (
                            <XCircle className="w-4 h-4 text-rose-500" />
                          ) : (
                            <AlertTriangle className="w-4 h-4 text-amber-500" />
                          )}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-slate-700 leading-snug mb-1.5">{ev.snippet}</p>
                          <a href={ev.simple_url || ev.source_url} target="_blank" rel="noreferrer" className="text-xs text-indigo-600 hover:text-indigo-800 transition-colors inline-flex items-center gap-1 max-w-full">
                            <ExternalLink className="w-3 h-3 flex-shrink-0" />
                            <span className="truncate">{ev.source_url}</span>
                          </a>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

            </div>
          )}
        </div>
      </div>
    </>
  );
}
