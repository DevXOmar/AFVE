import re

file_path = "models.py"
with open(file_path, "r") as f:
    orig = f.read()

target_code = """
    claims: list[VerifiedClaim]
    reasoning: str | None = None
    uncertainty_signals: dict[str, Any] = Field(default_factory=dict)
"""

replacement = """
    claims: list[VerifiedClaim]
    reasoning: str | None = None
    uncertainty_signals: dict[str, Any] = Field(default_factory=dict)
    global_evidence: list[EvidenceSnippet] = Field(default_factory=list)
"""

orig = orig.replace(target_code.strip(), replacement.strip())

with open(file_path, "w") as f:
    f.write(orig)

print("Models Replaced!")
