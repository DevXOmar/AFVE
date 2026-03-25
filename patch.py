import re

with open('agents.py', 'r') as f:
    text = f.read()

replacement = """class ScoringCalibrator:
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
        return nli

    @staticmethod
    def map_score_to_label(
        score: float, has_evidence: bool, snippets: list | None = None
    ):
        if not has_evidence or not snippets:
            return VerificationLabel.UNSUPPORTED

        statuses = [s.entailment_status for s in snippets]
        
        if any(s == EntailmentStatus.CONTRADICTS for s in statuses):
            return VerificationLabel.CONTRADICTED
        elif any(s == EntailmentStatus.SUPPORTS for s in statuses):
            return VerificationLabel.VERIFIED
        else:
            return VerificationLabel.UNCERTAIN

    @staticmethod
    def score_to_label(score: float):
        if score >= 0.70:
            return VerificationLabel.VERIFIED
        elif score >= 0.40:
            return VerificationLabel.UNCERTAIN
        elif score > 0.0:
            return VerificationLabel.CONTRADICTED
        return VerificationLabel.UNSUPPORTED

    @staticmethod
    def compute_overall_score(claims: list) -> float:
        if not claims:
            return 0.0
        
        base_score = sum(c.score for c in claims) / len(claims) if claims else 0.0
        
        contradicted_count  = sum(1 for c in claims if c.label == VerificationLabel.CONTRADICTED)
        unsupported_count  = sum(1 for c in claims if c.label == VerificationLabel.UNSUPPORTED)
        
        n = max(len(claims), 1)
        penalty = (contradicted_count * 0.3) / n + (unsupported_count * 0.1) / n
        
        return round(max(base_score - penalty, 0.0), 4)

"""

# Replace ScoringCalibrator block correctly.
# Find where it starts and ends
pattern = r'class ScoringCalibrator:.*?(?=\n\n\n# ===========================================================================\n# E\. Explanation Generator)'
text = re.sub(pattern, replacement, text, flags=re.DOTALL)

# Fix the scoring logic parameter in _verify_claim
text = text.replace(
    'label       = self._calibrator.score_to_label(score)',
    'label       = self._calibrator.map_score_to_label(score, has_evidence, annotated)'
)

with open('agents.py', 'w') as f:
    f.write(text)

print("Patch applied.")
