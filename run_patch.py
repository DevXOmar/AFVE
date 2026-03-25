import re
import os

with open('agents.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update NLI Loading to be strictly global
old_nli_loading = """_nli_pipeline = None
_reranker = None


def _get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is None:
        try:
            from transformers import pipeline  # type: ignore
            _nli_pipeline = pipeline(
                "text-classification",
                model=settings.nli_model_name,
                device=-1,  # CPU
            )
            logger.info("NLI loaded: %s", settings.nli_model_name)
        except Exception as exc:
            logger.warning("NLI unavailable (%s); using heuristic.", exc)
    return _nli_pipeline"""

new_nli_loading = """# Eagerly load NLI globally (Never per request/claim/loop)
NLI_PIPELINE = None
_reranker = None

try:
    from transformers import pipeline  # type: ignore
    NLI_PIPELINE = pipeline(
        "text-classification",
        model=settings.nli_model_name,
        device=-1,  # CPU
    )
    logger.info("NLI loaded globally: %s", settings.nli_model_name)
except Exception as exc:
    logger.warning("NLI unavailable globally (%s).", exc)

def _get_nli_pipeline():
    return NLI_PIPELINE"""

text = text.replace(old_nli_loading, new_nli_loading)

# We also need to fix VerificationEngine if it was not patched previously
def extract_class(text, class_name, next_class):
    start = text.find(f"class {class_name}:")
    end = text.find(f"class {next_class}:", start) if next_class else len(text)
    return text[start:end]

engine_old = extract_class(text, "VerificationEngine", "ScoringCalibrator")
engine_new = """class VerificationEngine:
    \"\"\"
    Scores each (claim, evidence) pair using ONLY NLI (cross-encoder).
    \"\"\"

    def __init__(self, llm_client=None) -> None:
        self._llm = None  # Explicitly remove LLM usage

    async def verify(self, retrieval: RetrievalResult, config) -> tuple[float, float, float, float, list[EvidenceSnippet]]:
        claim_text = retrieval.claim.contextualized_text
        snippets = retrieval.snippets
        analysis = retrieval.claim.analysis

        if not snippets:
            return 0.0, 0.0, 0.0, 0.0, snippets

        triple = analysis.triple if analysis else RelationalTriple()
        nli_hyp = _build_hypothesis(triple, claim_text)
        
        # NLI is already globally loaded
        import asyncio
        tasks = [
            asyncio.to_thread(self._score_snippet_sync, nli_hyp, s)
            for s in snippets
        ]
        annotated: list[EvidenceSnippet] = await asyncio.gather(*tasks)

        nli_agg = _mean([s.nli_score for s in annotated])
        llm_agg = 0.0 # Disabled
        source_weight = _mean([s.nli_score * CREDIBILITY_MULTIPLIERS[s.credibility_tier.value] for s in annotated])

        sup = sum(1 for s in annotated if s.entailment_status == EntailmentStatus.SUPPORTS)
        con = sum(1 for s in annotated if s.entailment_status == EntailmentStatus.CONTRADICTS)
        n = len(annotated)
        cross_agreement = 0.0 if n == 0 else ((sup - con) / n + 1.0) / 2.0

        return nli_agg, llm_agg, source_weight, cross_agreement, annotated

    def _score_snippet_sync(self, nli_hyp: str, snippet: EvidenceSnippet) -> EvidenceSnippet:
        # Uses purely global NLI model, accessed safely
        pipeline = _get_nli_pipeline()
        
        if pipeline:
            try:
                res = pipeline([{"text": nli_hyp, "text_pair": snippet.snippet}])[0]
                label = res["label"].upper()
                score = res["score"]
                
                # Strict Mapping
                if label == "AGREEMENT" or label == "ENTAILMENT":
                    status = EntailmentStatus.SUPPORTS
                elif label == "CONTRADICTION":
                    status = EntailmentStatus.CONTRADICTS
                else:
                    status = EntailmentStatus.NEUTRAL
            except Exception as e:
                logger.warning(f"NLI failure: {e}")
                status = EntailmentStatus.NEUTRAL
                score = 0.5
        else:
            status = EntailmentStatus.NEUTRAL
            score = 0.5

        snippet.entailment_status = status
        snippet.nli_score = score
        snippet.llm_judge_score = 0.0
        return snippet

# ===========================================================================
# D. Scoring Calibrator
# ===========================================================================

"""

text = text.replace(engine_old, engine_new)

with open('agents.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Patching complete.")