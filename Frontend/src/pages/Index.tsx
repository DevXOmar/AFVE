import { useState } from "react";
import { PromptInput } from "@/components/afve/PromptInput";
import { ModelOutput } from "@/components/afve/ModelOutput";
import { VerificationPanel } from "@/components/afve/verification/VerificationPanel";
import { Search, Loader2 } from "lucide-react";

export default function Index() {
  const [isGenerating, setIsGenerating] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);
  
  const [modelOutput, setModelOutput] = useState<string | null>(null);
  const [currentPrompt, setCurrentPrompt] = useState<string>("");
  const [currentModel, setCurrentModel] = useState<string>("qwen2.5-coder:7b");
  const [report, setReport] = useState<any | null>(null);
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async (prompt: string, model: string) => {
    setIsGenerating(true);
    setError(null);
    setModelOutput(null);
    setReport(null);
    setIsPanelOpen(false);
    setCurrentPrompt(prompt);
    setCurrentModel(model);

    try {
      // 1. Generate model response first
      const genRes = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, model }),
      });

      if (!genRes.ok) throw new Error("Failed to generate response");
      const genData = await genRes.json();
      setModelOutput(genData.model_output);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRunVerification = async () => {
    setIsVerifying(true);
    setIsPanelOpen(true);
    setReport(null);

    try {
      const runRes = await fetch("/api/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: crypto.randomUUID(), prompt: currentPrompt,
          model_output: modelOutput,
          config: { check_claims: true }
        }),
      });

      if (!runRes.ok) throw new Error("Verification failed");
      const reportData = await runRes.json();
      setReport(reportData);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsVerifying(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 pb-20 font-sans">
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-indigo-600 p-1.5 rounded-md">
              <Search className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-indigo-800">
              AFVE
            </h1>
          </div>
          <p className="text-sm text-slate-500 font-medium hidden sm:block">Agentic Factual Verification Engine</p>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-4 pt-10">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold mb-3 tracking-tight">Ask & Verify</h2>
          <p className="text-slate-500 max-w-lg mx-auto leading-relaxed">
            Get answers from open-weights models and use our intelligent agent to dynamically verify factual claims in the response.
          </p>
        </div>

        <PromptInput 
          onSubmit={handleGenerate} 
          isLoading={isGenerating || isVerifying} 
          model={currentModel}
          onModelChange={setCurrentModel}
          models={[
            { id: "qwen2.5-coder:7b", name: "Qwen 2.5 Coder (7B)" },
            { id: "gemma2:2b", name: "Gemma 2 (2B)" },
          ]}
        />

        {error && (
          <div className="mt-6 p-4 bg-rose-50 border border-rose-100 rounded-lg text-rose-700 text-sm">
            <span className="font-semibold">Error:</span> {error}
          </div>
        )}

        {isGenerating && (
          <div className="mt-8 flex flex-col items-center justify-center p-10 bg-white rounded-xl shadow-sm border border-slate-200">
            <div className="w-10 h-10 border-4 border-indigo-100 border-t-indigo-600 rounded-full animate-spin mb-4" />
            <p className="text-slate-700 font-medium">Generating response...</p>
            <p className="text-sm text-slate-500 mt-2">{currentModel} is thinking.</p>
          </div>
        )}

        {modelOutput && !isGenerating && (
          <div className="mt-8 space-y-6 animate-slide-up">
            <ModelOutput output={modelOutput} />
            
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-5 bg-indigo-50/50 border border-indigo-100 rounded-xl shadow-sm">
              <div>
                <h4 className="text-sm font-semibold text-indigo-900 flex items-center gap-2">
                  <Search className="w-4 h-4 text-indigo-600" />
                  Need to check the facts?
                </h4>
                <p className="text-xs text-indigo-700/70 mt-1">The AFVE agent can extract and verify the claims in this text.</p>
              </div>
              <button
                onClick={handleRunVerification}
                disabled={isVerifying}
                className="w-full sm:w-auto flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2.5 rounded-lg text-sm font-medium transition-all shadow-sm disabled:opacity-50 hover:shadow-md"
              >
                {isVerifying ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Agent is working
                  </>
                ) : (
                  <>
                    Run Verification
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </main>

      <VerificationPanel 
        isOpen={isPanelOpen} 
        onClose={() => setIsPanelOpen(false)} 
        isLoading={isVerifying}
        report={report}
      />
    </div>
  );
}
