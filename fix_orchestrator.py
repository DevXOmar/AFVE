import re

file_path = "agents.py"
with open(file_path, "r") as f:
    orig = f.read()

target_code = """
        return VerificationReport(
            metadata=JobMetadata(
                job_id=request.job_id,
                processing_time_ms=elapsed_ms,
                sources_queried=sum(len(c.evidence) for c in valid),
                claims_extracted=len(claims),
                atoms_generated=len(claims),
                claims_verified=len(valid),
                retrieval_failures=no_ev,
                no_evidence_claims=no_ev,
                relation_unverified_count=rel_unverif,
            ),
            overall_score=overall_score,
            overall_label=overall_label,
            claims=valid,
        )
"""

replacement = """
        # Global Evidence Aggregation
        all_evidence = []
        for c in valid:
            all_evidence.extend(c.evidence)
            
        # Deduplicate
        seen_keys = set()
        deduped = []
        for e in all_evidence:
            key = f"{e.source_url}::{e.snippet.strip()}"
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(e)
                
        # Rank by NLI output and credibility
        def _global_sort(e):
            score = e.nli_score
            if e.entailment_status == EntailmentStatus.SUPPORTS:
                score += 2.0
            elif e.entailment_status == EntailmentStatus.CONTRADICTS:
                # Contradicting evidence is also valuable to show
                score += 1.0
            # Credibility tier bonus
            if e.credibility_tier == CredibilityTier.TIER_1:
                score += 0.5
            elif e.credibility_tier == CredibilityTier.TIER_2:
                score += 0.2
            return float(score)
            
        deduped.sort(key=_global_sort, reverse=True)
        top_global_evidence = deduped[:5]
        
        # Clear evidence per claim directly in response
        for c in valid:
            c.evidence = []

        return VerificationReport(
            metadata=JobMetadata(
                job_id=request.job_id,
                processing_time_ms=elapsed_ms,
                sources_queried=len(all_evidence),
                claims_extracted=len(claims),
                atoms_generated=len(claims),
                claims_verified=len(valid),
                retrieval_failures=no_ev,
                no_evidence_claims=no_ev,
                relation_unverified_count=rel_unverif,
            ),
            overall_score=overall_score,
            overall_label=overall_label,
            claims=valid,
            global_evidence=top_global_evidence,
        )
"""

orig = orig.replace(target_code.strip(), replacement.strip())

with open(file_path, "w") as f:
    f.write(orig)

print("Orchestrator Replaced!")
