import asyncio
from agents import VerificationOrchestrator

async def main():
    print("Testing locally")
    orchestrator = VerificationOrchestrator()
    report = await orchestrator.run("Who developed the theory of relativity?", "Albert Einstein developed the theory of relativity in 1905.")
    print("global_ev=", [[e.snippet, e.entailment_status] for e in report.global_evidence])
    print("claims=", [(c.claim, len(c.evidence)) for c in report.claims])

asyncio.run(main())
