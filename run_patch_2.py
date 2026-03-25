import re

with open('agents.py', 'r', encoding='utf-8') as f:
    text = f.read()

def extract_class(text, class_name, next_class_or_header):
    start = text.find(f"class {class_name}:")
    end = text.find(next_class_or_header, start)
    return text[start:end]

# Rewrite ClaimExtractionAgent
claim_ext_old = extract_class(text, "ClaimExtractionAgent", "# ===========================================================================\n# B. Retrieval Agent")
claim_ext_new = """class ClaimExtractionAgent:
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
        pat = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s")
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

text = text.replace(claim_ext_old, claim_ext_new)

# Rewrite RelationalDecomposer
decomp_old = extract_class(text, "RelationalDecomposer", "# ===========================================================================\n# C. Fact Verification")
decomp_new = """class RelationalDecomposer:
    \"\"\"
    Hybrid approach: Try S-V-O regex parsing, fallback to LLM.
    \"\"\"

    def __init__(self, llm_client=None) -> None:
        self._llm = llm_client

    def _rule_based_extract(self, sentence: str) -> RelationalTriple | None:
        \"\"\"Simple Subject-Verb-Object rule.\"\"\"
        import re
        match = re.match(r'^([\w\s]+?)\s+(is|was|are|were|has|have|had|founded|built)\s+(.+)$', sentence, re.IGNORECASE)
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

text = text.replace(decomp_old, decomp_new)

# Rewrite ScoringCalibrator
scorer_old = extract_class(text, "ScoringCalibrator", "# ===========================================================================\n# E. Explanation Generator")
scorer_new = """class ScoringCalibrator:
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

text = text.replace(scorer_old, scorer_new)

with open('agents.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Patching 2 complete.")