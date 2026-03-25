import re

file_path = "agents.py"
with open(file_path, "r") as f:
    orig = f.read()

# 1. replace `if len(p) > 30:` with `if len(re.findall(r"\w+", p)) >= 5:`
orig = orig.replace("""                p = re.sub(r"\\s+", " ", p).strip()
                if len(p) > 30:
                    out.append(p)""", 
"""                p = re.sub(r"\\s+", " ", p).strip()
                if len(re.findall(r"\\w+", p)) >= 5:
                    out.append(p)""")

# 2. replace `top_pairs = scored[:3]` with `top_pairs = scored[:5]`
orig = orig.replace("top_pairs = scored[:3]", "top_pairs = scored[:5]")

# 3. From NLI results (annotated), select top 2-3 based on highest entailment score, strongest keyword overlap.
target_code = """
        annotated: list[EvidenceSnippet]
        if getattr(config, "enable_nli", True):
            triple = analysis.triple if analysis else RelationalTriple()
            nli_hyp = _build_hypothesis(triple, claim_text)
            import asyncio
            tasks = [asyncio.to_thread(self._score_snippet_sync, nli_hyp, s) for s in selected]
            annotated = await asyncio.gather(*tasks)
        else:
            annotated = selected
"""

replacement = """
        annotated: list[EvidenceSnippet]
        if getattr(config, "enable_nli", True):
            triple = analysis.triple if analysis else RelationalTriple()
            nli_hyp = _build_hypothesis(triple, claim_text)
            import asyncio
            tasks = [asyncio.to_thread(self._score_snippet_sync, nli_hyp, s) for s in selected]
            annotated = list(await asyncio.gather(*tasks))
            
            # Sort by NLI score (if SUPPORTS, score + 1.0) and semantic score.
            def _sort_key(snip: EvidenceSnippet) -> float:
                status_mult = 1.0 if snip.entailment_status == EntailmentStatus.SUPPORTS else (0.5 if snip.entailment_status == EntailmentStatus.NEUTRAL else 0.0)
                sem = getattr(snip, "semantic_score", 0.0)
                return float(snip.nli_score + status_mult + sem)
            
            annotated.sort(key=_sort_key, reverse=True)
            # Keep top 3 
            annotated = annotated[:3]
        else:
            annotated = selected[:3]
"""

orig = orig.replace(target_code.strip(), replacement.strip())

with open(file_path, "w") as f:
    f.write(orig)

print("Replaced!")
