import re

with open("/Users/óhaikmohammedomar/LLMLens/AFVE/Frontend/src/components/afve/verification/VerificationPanel.tsx", "r") as f:
    text = f.read()

# Interface add
text = text.replace("claims: any[];", "claims: any[];\n  global_evidence?: any[];")


if "import { ExternalLink, CheckCircle, AlertTriangle, XCircle } from \"lucide-react\";" not in text:
    text = text.replace("import { X, Search } from \"lucide-react\";", "import { X, Search, ExternalLink, CheckCircle, AlertTriangle, XCircle } from \"lucide-react\";")

# Render addition
evidence_render = """
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
                          <a href={ev.source_url} target="_blank" rel="noreferrer" className="teyt-xs text-indigo-600 hover:text-indigo-800 transition-colors inline-flex items-center gap-1 max-w-full">
                            <ExternalLink className="w-3 h-3 flex-shrink-0" />
                            <span className="truncate">{new URL(ev.source_url).hostname || ev.source_url}</span>
                          </a>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
"""

if "Top Evidence" not in text:
    text = text.replace("<ClaimCard claim={claim} />\n                  </div>\n                ))}\n              </div>", "<ClaimCard claim={claim} />\n                  </div>\n                ))}\n              </div>" + evidence_render)
    
with open("/Users/shaikmohammedomar/LLMLens/AFVE/Frontend/src/components/afve/verification/VerificationPanel.tsx", "w") as f:
    f.write(text)
