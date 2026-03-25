import re

with open('agents.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update NLI Loading to Eager Global Singleton
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

new_nli = """# Eagerly load NLI globally (Never per request/claim/loop)
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

text = text.replace(old_nli, new_nli)


def rewrite_section(text, start_pattern, end_pattern, replacement):
    start_match = re.search(start_pattern, text)
    if not start_match: print(f"Failed to find {start_pattern}"); return text
    
    end_match = re.search(end_pattern, text[start_match.start():])
    if not end_match: print(f"Failed to find {end_pattern}"); return text
    
    start_idx = start_match.start()
    end_idx = start_idx + end_match.start()
    
    return text[:start_idx] + replacement + text[end_idx:]


# 2. ClaimExtractionAgent rewrite
claim_ext_replacement = """class ClaimExtractionAgent:
    \"\"\"
    Rule-based extraction of verifiable claims, followed by hybrid triple decomposition.
    \"\"\"

    def __init__(self, llm_client=None, decomposer=None) -> None:
        self._llm = llm_client
        self._decomposer = decomposer or RelationalDecomposer(llm_client=llm_client)

    def is_verifiable(self, text: str) -> bool:
        \"\"\"Rule-based classifier to reject questions, opinions, and conversational text.\"\"\"
        text = text.strip()
        if not text or len(text.split()) < 3: return False
        if text.endswith('?'): return False
        
        lower_text = text.lower()
        unverifiable_prefixes = [
            "i think", "i believe", "in my opinion", "hello", "hi", "how can i",
            "here are", "below is", "let me", "sure", "of course", "personally",
            "arguably", "some might say"
        ]
        if any(lower_text.startswith(p) for p in unverifiable_prefixes):
            return False
            
        return True

    def _split_sentences(self, text: str) -> list[RawClaim]:
        import re
        pat = re.compile(r"(?<!\\w\.\\w.)(?<![A-Z][a-z]\\.)(?<=\\.|\\?|!)\\s")
        parts = pat.split(text.strip())
        claims: list[RawClaim] = []
        for idx, part in enumerate(parts):
            part = part.strip()
            if not part: continue
            is_v = self.is_verifiable(part)
            claims.append(RawClaim(text=part, sentence_index=idx, is_verifiable=is_v))
        return claims

    async def extract(self, request: VerificationRequest) -> list[ResolvedClaim]:
        raw = self._split_sentences(request.model_output)
        verifiable = [c for c in raw if c.is_verifiable]
        logger.info("Extracted %d verifiable / %d total sentences.", len(verifiable), len(raw))

        texts = [c.text for c in verifiable]
        
        import asyncio
        atom_tasks = [
            self._decomposer.decompose(texts[i], verifiable[i].sentence_index, request.config)
            for i in range(len(verifiable))
        ]
        atom_lists = await asyncio.gather(*atom_tasks, return_exceptions=True)

        claims: list[ResolvedClaim] = []
        for i, res in enumerate(atom_lists):
            if isinstance(res, Exception):
                logger.warning("Decomposition error sentence %d: %s", i, res)
                continue
            for atom in res:
                claims.append(ResolvedClaim(
                    original_text=atom.original_sentence,
                    contextualized_text=atom.atomic_text,
                    sentence_index=atom.sentence_index,
                    atom_index=atom.atom_index,
                    analysis=atom.analysis,
                ))

        logger.info("Generated %d atomic claims from %d sentences.", len(claims), len(verifiable))
        return claims

"""
text = rewrite_section(text, r"class ClaimExtractionAgent:", r"# ===========================================================================\n# B\. Retrieval Agent", claim_ext_replacement)

# 3. RelationalDecomposer rewrite
decomp_replacement = """class RelationalDecomposer:
    \"\"\"
    Hybrid approach: Try S-V-O regex parsing, fallback to LLM.
    \"\"\"

    def __init__(self, llm_client=None) -> None:
        self._llm = llm_client

    def _rule_based_extract(self, sentence: str) -> RelationalTriple | None:
        \"\"\"Simple Subject-Verb-Object rule.\"\"\"
        import re
        match = re.match(r'^([\\w\\s]+?)\\s+(is|was|are|were|has|have|had|founded|built)\\s+(.+)$', sentence, re.IGNORECASE)
        if match:
            s, p, o = match.groups()
            return RelationalTriple(subject=s.strip(), predicate=p.strip(), object=o.rstrip('.').strip(), qualifier="")
        return None

    async def decompose(self, sentence: str, sentence_idx: int, config) -> list[AtomicClaim]:
        triple = self._rule_based_extract(sentence)
        if not triple and self._llm:
            try:
                system = "Extract Subject, Predicate, Object from sentence. Return JSON: {'s': '', 'p': '', 'o': '', 'q': ''}"
                raw = await self._llm.complete(system, sentence, max_tokens=150)
                import json
                data = json.loads(raw.strip().lstrip("```json").rstrip("```").strip())
                triple = RelationalTriple(
                    subject=data.get('s','').strip(),
                    predicate=data.get('p','').strip(),
                    object=data.get('o','').strip(),
                    qualifier=data.get('q','').strip()
                )
            except Exception as e:
                logger.warning(f"Decomposition fallback failed: {e}")
                
        if not triple:
            words = sentence.split()
            subject = words[0] if words else ""
            triple = RelationalTriple(subject=subject, predicate="is", object=sentence, qualifier="")

        analysis = RelationalAnalysis(
            triple=triple,
            atomic_text=sentence,
            search_queries=[f"{triple.subject} {triple.predicate} {triple.object}".strip()],
            numeric_assertions=[],
            claim_type=ClaimType.GENERAL,
            truth_prior=0.5
        )
        return [AtomicClaim(original_sentence=sentence, atomic_text=sentence, sentence_index=sentence_idx, atom_index=0, analysis=analysis)]

"""
text = rewrite_section(text, r"class RelationalDecomposer:", r"# ===========================================================================\n# C\. Fact Verification", decomp_replacement)

# 4. VerificationEngine rewrite
engine_replace = """class VerificationEngine:
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
        # Uses purely global Eager NLI model (never loads dynamically in loops)
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

"""
text = rewrite_section(text, r"class VerificationEngine:", r"# ===========================================================================\n# D\. Scoring Calibrator", engine_replace)

# 5. ScoringCalibrator rewrite
scorer_replace = """class ScoringCalibrator:
    \"\"\"
    Maps NLI results securely to final verification labels.
    \"\"\"

    @staticmethod
    def compute_claim_score(
        nli: float, llm_judge: float, source_weight: float,
        cross_agreement: float, relation_score: float = 0.0,
        numeric_result=None, has_evidence: bool = True,
        analysis=None, snippets=None
    ) -> float:
        if not has_evidence: return 0.0
        return nli # Use raw NLI directly for deterministic final logic

    @staticmethod
    def map_score_to_label(
        score: float, has_evidence: bool, snippets: list[EvidenceSnippet] | None = None
    ) -> VerificationLabel:
        if not has_evidence or not snippets:
            return VerificationLabel.UNSUPPORTED

        # Strictly inspect entailment statuses for mapping
        statuses = [s.entailment_status for s in snippets]
        
        if any(s == EntailmentStatus.CONTRADICTS for s in statuses):
            return VerificationLabel.CONTRADICTED
        elif any(s == EntailmentStatus.SUPPORTS for s in statuses):
            return VerificationLabel.VERIFIED
        else:
            return VerificationLabel.UNCERTAIN

"""
text = rewrite_section(text, r"class ScoringCalibrator:", r"# ===========================================================================\n# E\. Explanation Generator", scorer_replace)


with open('agents.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Patching correct 3.")