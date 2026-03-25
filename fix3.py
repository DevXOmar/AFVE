import re

with open('agents.py', 'r', encoding='utf-8') as f:
    text = f.read()

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

    @staticmethod
    def compute_overall_score(claims: list[VerifiedClaim]) -> float:
        if not claims:
            return 0.0
        
        # Base score natively driven by NLI
        base_score = sum(c.score for c in claims) / len(claims) if claims else 0.0
        
        # Map penalties to updated ENUM 
        contradicted_count  = sum(1 for c in claims if c.label == VerificationLabel.CONTRADICTED)
        unsupported_count  = sum(1 for c in claims if c.label == VerificationLabel.UNSUPPORTED)
        
        n = max(len(claims), 1)
        
        # Penalties:
        # CONTRADICTED penalty (-0.3)
        # UNSUPPORTED penalty (-0.1)
        penalty = (contradicted_count * 0.3) / n + (unsupported_count * 0.1) / n
        
        return round(max(base_score - penalty, 0.0), 4)

"""

text = re.sub(r'class ScoringCalibrator:.*?(?=# ===========================================================================\n# E\. Explanation Generator)', scorer_new + '\n\n', text, flags=re.DOTALL)

with open('agents.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Fixed ScoringCalibrator gracefully.")