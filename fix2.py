import sys

with open('agents.py', 'r', encoding='utf-8') as f:
    text = f.read()

start_idx = text.find("class VerificationEngine:")
if start_idx == -1:
    print("Could not find VerificationEngine")
    sys.exit(1)

end_idx = text.find("# ===========================================================================\n# NumericComparator", start_idx)
if end_idx == -1:
    print("Could not find NumericComparator")
    sys.exit(1)

engine_code = """class VerificationEngine:
    \"\"\"
    Scores each (claim, evidence) pair using ONLY the global NLI (cross-encoder).
    NO LLM usage happens here.
    \"\"\"

    def __init__(self, llm_client=None) -> None:
        self._llm = None  # Explicitly remove LLM usage

    async def verify(
        self, retrieval: RetrievalResult, config
    ) -> tuple[float, float, float, float, list[EvidenceSnippet]]:
        claim_text = retrieval.claim.contextualized_text
        snippets   = retrieval.snippets
        analysis   = retrieval.claim.analysis

        if not snippets:
            return 0.0, 0.0, 0.0, 0.0, snippets

        triple = analysis.triple if analysis else RelationalTriple()
        nli_hyp = _build_hypothesis(triple, claim_text)

        import asyncio
        tasks = [
            asyncio.to_thread(self._score_snippet_sync, nli_hyp, s)
            for s in snippets
        ]
        annotated: list[EvidenceSnippet] = await asyncio.gather(*tasks)

        nli_agg = _mean([s.nli_score for s in annotated])
        llm_agg = 0.0 # Disabled
        
        # Source credibility based only on NLI score
        source_weight = _mean([
            s.nli_score * CREDIBILITY_MULTIPLIERS.get(s.credibility_tier.value, 1.0)
            for s in annotated
        ])

        sup = sum(1 for s in annotated if s.entailment_status == EntailmentStatus.SUPPORTS)
        con = sum(1 for s in annotated if s.entailment_status == EntailmentStatus.CONTRADICTS)
        n = len(annotated)
        cross_agreement = 0.0 if n == 0 else ((sup - con) / n + 1.0) / 2.0

        return nli_agg, llm_agg, source_weight, cross_agreement, annotated

    def _score_snippet_sync(self, nli_hyp: str, snippet: EvidenceSnippet) -> EvidenceSnippet:
        # Load NLI correctly from the global single instance
        pipeline = _get_nli_pipeline()
        
        if pipeline:
            try:
                res = pipeline([{"text": nli_hyp, "text_pair": snippet.snippet}])[0]
                label = res["label"].upper()
                score = res["score"]
                
                # Strict NLI to Status Mapping
                if label in ["AGREEMENT", "ENTAILMENT", "SUPPORTS"]:
                    status = EntailmentStatus.SUPPORTS
                elif label in ["CONTRADICTION", "CONTRADICTS"]:
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
        
        # Keep compatibility for these fields
        snippet.subject_match_score = 1.0
        snippet.relation_match_score = 1.0
        
        return snippet

"""

text = text[:start_idx] + engine_code + "\n\n" + text[end_idx:]


# Replace the NLI Loader to be EAGER and properly instantiated at top level
old_nli = """_nli_pipeline = None
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

new_nli = """# Eagerly load NLI globally at module startup.
# Never instantiate per claim or request.
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
    # Returns the singleton NLI loaded above
    return NLI_PIPELINE"""

text = text.replace(old_nli, new_nli)

with open('agents.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Engine and NLI successfully patched!")
