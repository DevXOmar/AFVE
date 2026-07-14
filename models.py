"""
models.py — Pydantic v2 schemas for AFVE (Relational Fact Verification Engine).

Core architectural change
──────────────────────────
The previous model used a flat `primary_entity` string as the retrieval anchor,
which caused entity validation to check "is Apple mentioned?" rather than
"does Apple's Wikipedia article say it was founded in 1976?".

This revision replaces that with a proper relational triple model:

  RelationalTriple
    subject  — the entity performing or possessing the predicate  ("Elon Musk")
    predicate — the relation being asserted                        ("founded")
    object   — what the predicate resolves to                     ("Tesla")
    qualifier — optional precision modifier                        ("in 2003")

  AtomicClaim
    The unit of verification after multi-claim splitting.
    One sentence can split into multiple AtomicClaims:
      "Apple was founded in 1976 and employs 150,000 people"
      → AtomicClaim(triple=(Apple, founded, -, 1976))
      → AtomicClaim(triple=(Apple, employs, 150000 people, -))

  NumericAssertion
    Structured representation of any numeric claim (count, date, percent,
    money, rank) with unit and comparison_direction. Used by the numeric
    comparator to evaluate retrieved values against the claimed value
    without relying on NLI to catch order-of-magnitude errors.

  RelationalAnalysis (replaces ClaimAnalysis)
    The output of the RelationalDecomposer per AtomicClaim.
    Contains the triple, atomic claim text, subject-first search queries,
    numeric_assertions, claim_type, truth_prior.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class VerificationLabel(str, Enum):
    VERIFIED = "VERIFIED"
    MOSTLY_TRUE = "MOSTLY_TRUE"
    LIKELY_FALSE = "LIKELY_FALSE"
    UNCERTAIN = "UNCERTAIN"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"
    NO_VERIFIABLE_CONTENT = "NO_VERIFIABLE_CONTENT"


class EntailmentStatus(str, Enum):
    SUPPORTS = "supports"
    WEAK_SUPPORT = "weak_support"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class CredibilityTier(int, Enum):
    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3


class ClaimType(str, Enum):
    """
    Semantic type of the atomic claim's predicate.
    DATE       — temporal attribute (founded, born, released…)
    MEMBERSHIP — subject is member / founder / employee of object
    ATTRIBUTE  — subject has numeric/qualitative property
    AUTHORSHIP — subject created / wrote / invented object
    ROLE       — subject holds a position at / within object
    LOCATION   — subject is situated at / in object
    CAUSAL     — subject caused / led to object
    COMPARATIVE— subject ranks relative to object
    EXISTENCE  — subject exists / is a real entity
    GENERAL    — any other verifiable relation
    """
    DATE = "date"
    MEMBERSHIP = "membership"
    ATTRIBUTE = "attribute"
    AUTHORSHIP = "authorship"
    ROLE = "role"
    LOCATION = "location"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    EXISTENCE = "existence"
    GENERAL = "general"


class ClaimCategory(str, Enum):
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    ABSTRACT = "abstract"


class VerificationMode(str, Enum):
    FACTUAL = "factual"
    EXPLANATION = "explanation"


class EntityLabel(str, Enum):
    PERSON = "PERSON"
    ORG = "ORG"
    GPE = "GPE"
    LOC = "LOC"
    DATE = "DATE"
    NUMBER = "NUMBER"
    PERCENT = "PERCENT"
    MONEY = "MONEY"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    OTHER = "OTHER"


class NumericUnit(str, Enum):
    YEAR = "year"
    COUNT = "count"
    PERCENT = "percent"
    MONEY = "money"
    RANK = "rank"
    DISTANCE = "distance"
    WEIGHT = "weight"
    NONE = "none"


class ComparisonResult(str, Enum):
    MATCHES = "matches"          # retrieved value == claimed ± tolerance
    CLOSE = "close"              # within 10% for floats, ±1 year for dates
    DIVERGES = "diverges"        # outside close tolerance
    CONTRADICTS = "contradicts"  # direction / sign / order-of-magnitude wrong
    NOT_FOUND = "not_found"      # no numeric value found in evidence


# ---------------------------------------------------------------------------
# Relational triple model
# ---------------------------------------------------------------------------


class EntitySpan(BaseModel):
    """A named entity extracted from a claim with its semantic role."""
    text: str
    label: EntityLabel
    role: str = ""     # "subject" | "object" | "qualifier" | "predicate"
    start: int = 0
    end: int = 0


class RelationalTriple(BaseModel):
    """
    The core semantic unit. Decomposing claims into triples makes the
    verification question explicit:

      Claim: "Elon Musk co-founded Tesla Motors in 2003"
      Triple:
        subject   = "Elon Musk"          (PERSON)
        predicate = "co-founded"
        object    = "Tesla Motors"        (ORG)
        qualifier = "in 2003"            (DATE)

    The retrieval query is always built from the SUBJECT, not the object,
    because Wikipedia/search articles are titled after subjects, not predicates.
    Correct query: "Elon Musk Tesla co-founded" NOT "Tesla Motors founders".

    For existence claims (no object): subject is the entity, predicate is
    the existence assertion, object is empty.
    """
    subject: str = ""
    subject_label: EntityLabel = EntityLabel.OTHER
    predicate: str = ""
    object: str = ""
    object_label: EntityLabel = EntityLabel.OTHER
    qualifier: str = ""          # e.g. "in 2003", "at $10 billion", "with 50,000 employees"
    qualifier_label: EntityLabel = EntityLabel.OTHER
    polarity: Literal["positive", "negative"] = "positive"


class NumericAssertion(BaseModel):
    """
    Structured form of a numeric claim for exact/approximate comparison
    against retrieved evidence. This is what prevents "20% revenue growth"
    from passing as correct when evidence says "2% revenue growth".
    """
    raw_text: str                          # original string in claim, e.g. "1976"
    normalized_value: float | None = None  # parsed float, e.g. 1976.0
    unit: NumericUnit = NumericUnit.NONE
    # For date ranges (decade approximations)
    tolerance_low: float | None = None
    tolerance_high: float | None = None


class RelationalAnalysis(BaseModel):
    """
    Structured output produced by the RelationalDecomposer for one AtomicClaim.
    Replaces the old flat ClaimAnalysis.
    """
    claim_type: ClaimType
    claim_category: ClaimCategory = ClaimCategory.FACTUAL
    triple: RelationalTriple
    atomic_claim_text: str              # the specific sub-claim being verified
    # Subject-first queries (always query by subject, not object)
    subject_queries: list[str] = Field(default_factory=list)
    # Structured numeric assertions extracted from the claim
    numeric_assertions: list[NumericAssertion] = Field(default_factory=list)
    # Bayesian prior for this claim type
    truth_prior: float = Field(default=0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class VerificationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    strictness: Annotated[float, Field(ge=0.0, le=1.0)] = 0.8
    max_retrieval_attempts: Annotated[int, Field(ge=1, le=5)] = 3
    top_k_snippets: Annotated[int, Field(ge=1, le=20)] = 5
    enable_llm_judge: bool = True
    enable_nli: bool = True
    enable_web_search: bool = True
    enable_kb_search: bool = True
    enable_relational_decomposition: bool = True
    # Split compound sentences into multiple atomic claims
    enable_multi_claim_splitting: bool = True


class VerificationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: UUID
    prompt: Annotated[str, Field(min_length=1, max_length=8_000)]
    model_output: Annotated[str, Field(min_length=1, max_length=32_000)]
    config: VerificationConfig = Field(default_factory=VerificationConfig)


class OllamaRunRequest(BaseModel):
    """Run an Ollama model and then verify the generated output with AFVE."""

    model_config = ConfigDict(frozen=True)

    model: Annotated[str, Field(min_length=1, max_length=200)]
    prompt: Annotated[str, Field(min_length=1, max_length=8_000)]
    # Optional override for verification behavior
    config: VerificationConfig = Field(default_factory=VerificationConfig)
    # Optional caller-provided job id (otherwise server generates one)
    job_id: UUID | None = None


# ---------------------------------------------------------------------------
# Internal pipeline schemas
# ---------------------------------------------------------------------------


class RawClaim(BaseModel):
    text: str
    sentence_index: int
    is_verifiable: bool = True


class AtomicClaim(BaseModel):
    """
    A single verifiable fact unit. Multiple AtomicClaims can derive from
    one sentence after multi-claim splitting.
    """
    original_sentence: str
    atomic_text: str                # the specific sub-claim text
    sentence_index: int
    atom_index: int = 0             # 0 = first atom from that sentence
    analysis: RelationalAnalysis | None = None


class ResolvedClaim(BaseModel):
    """
    Backward-compatible wrapper that carries one AtomicClaim through
    the retrieval and verification pipeline. The analysis field is the
    RelationalAnalysis from the decomposer.
    """
    original_text: str
    contextualized_text: str        # coreference-resolved form
    sentence_index: int
    atom_index: int = 0
    analysis: RelationalAnalysis | None = None
    relevance_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    claim_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.6
    verification_mode: VerificationMode = VerificationMode.FACTUAL


class EvidenceSnippet(BaseModel):
    """
    Retrieved evidence for one atomic claim.

    subject_match_score  — does the snippet discuss the SUBJECT of the triple?
    object_match_score   — does the snippet mention the OBJECT of the triple?
    relation_match_score — does the snippet address the predicate?
    numeric_result       — structured numeric comparison result
    """
    source_url: str
    snippet: str
    credibility_tier: CredibilityTier
    entailment_status: EntailmentStatus = EntailmentStatus.NEUTRAL
    llm_entailment_status: EntailmentStatus | None = None

    nli_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    llm_judge_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    # Relational match scores (replaces flat entity_match_score)
    subject_match_score: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    object_match_score: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    relation_match_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    # Semantic match strength between claim and evidence (used for scoring).
    semantic_score: Annotated[float, Field(ge=0.0)] = 0.0
    numeric_result: ComparisonResult = ComparisonResult.NOT_FOUND

    claim_type: ClaimType = ClaimType.GENERAL

    # ------------------------------------------------------------------
    # Frontend compatibility fields
    # ------------------------------------------------------------------

    @computed_field(return_type=str)
    @property
    def text(self) -> str:
        # UI components expect `text`; backend uses `snippet`.
        return self.snippet

    @computed_field(return_type=str)
    @property
    def url(self) -> str:
        # UI components expect `url`; backend uses `source_url`.
        return self.source_url

    @computed_field(return_type=str)
    @property
    def pair_classification(self) -> str:
        """3-way claim↔evidence classification.

        - SUPPORTS: same meaning
        - CONTRADICTS: opposite meaning (high-confidence)
        - IRRELEVANT: low similarity / generic / unrelated

        We keep EntailmentStatus as-is for backward compatibility.
        """
        sem = 0.0
        try:
            sem = float(self.semantic_score)
        except Exception:
            sem = 0.0

        if sem < 0.35:
            return "IRRELEVANT"
        if self.entailment_status == EntailmentStatus.SUPPORTS:
            return "SUPPORTS"
        if self.entailment_status == EntailmentStatus.WEAK_SUPPORT:
            return "WEAK_SUPPORT"
        if self.entailment_status == EntailmentStatus.CONTRADICTS:
            return "CONTRADICTS"
        return "IRRELEVANT"

    @field_validator("source_url")
    @classmethod
    def url_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source_url must not be empty")
        return v


class RetrievalResult(BaseModel):
    claim: ResolvedClaim
    snippets: list[EvidenceSnippet]
    queries_attempted: int
    retrieval_success: bool


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


class VerifiedClaim(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    claim: str
    contextualized_claim: str
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    label: VerificationLabel
    explanation: str
    evidence: list[EvidenceSnippet]

    claim_type: ClaimType = ClaimType.GENERAL
    claim_category: ClaimCategory = ClaimCategory.FACTUAL
    triple: RelationalTriple | None = None
    entities: list[EntitySpan] = Field(default_factory=list)
    numeric_assertions: list[NumericAssertion] = Field(default_factory=list)
    numeric_comparison: ComparisonResult = ComparisonResult.NOT_FOUND
    claim_year: int | None = None
    reference_year: int | None = None
    validation_errors: list[str] = Field(default_factory=list)
    top_supporting_evidence: list[EvidenceSnippet] = Field(default_factory=list)
    top_contradicting_evidence: list[EvidenceSnippet] = Field(default_factory=list)
    relevance_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    claim_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.6

    subject_validated: bool = False   # subject found in ≥1 snippet
    relation_verified: bool = False   # predicate confirmed in evidence
    has_evidence: bool = True
    truth_prior: float = 0.5

    nli_component: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    llm_judge_component: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    source_weight_component: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    cross_source_agreement: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    numeric_score_component: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5

    @computed_field(return_type=str)
    @property
    def claim_text(self) -> str:
        return self.claim

    @computed_field(return_type=float)
    @property
    def confidence_score(self) -> float:
        return self.score


class JobMetadata(BaseModel):
    job_id: UUID
    processing_time_ms: int
    sources_queried: int
    claims_extracted: int
    atoms_generated: int = 0          # after multi-claim splitting
    claims_verified: int
    retrieval_failures: int = 0
    no_evidence_claims: int = 0
    relation_unverified_count: int = 0
    verification_mode: VerificationMode = VerificationMode.FACTUAL


class VerificationReport(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    metadata: JobMetadata
    overall_score: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    overall_label: VerificationLabel
    claims: list[VerifiedClaim]
    reasoning: str | None = None
    uncertainty_signals: dict[str, Any] = Field(default_factory=dict)
    global_evidence: list[EvidenceSnippet] = Field(default_factory=list)
    factuality_percentage: Annotated[float | None, Field(ge=0.0, le=100.0)] = None
    factuality_rating: str | None = None
    sources_used: list[str] = Field(default_factory=list)
    mode: VerificationMode = VerificationMode.FACTUAL
    scoring_note: str | None = None

    @model_validator(mode="after")
    def populate_uncertainty_signals(self) -> VerificationReport:
        contradicted = [c for c in self.claims if c.label == VerificationLabel.CONTRADICTED]
        uncertain = [c for c in self.claims if c.label == VerificationLabel.UNCERTAIN]
        no_evidence = [c for c in self.claims if not c.has_evidence]
        relation_unverified = [c for c in self.claims if not c.relation_verified and c.has_evidence]
        numeric_contradictions = [
            c for c in self.claims
            if c.numeric_comparison == ComparisonResult.CONTRADICTS
        ]

        type_breakdown: dict[str, int] = {}
        for c in self.claims:
            type_breakdown[c.claim_type.value] = type_breakdown.get(c.claim_type.value, 0) + 1

        self.uncertainty_signals = {
            "contradicted_by_label_count": len(contradicted),
            "uncertain_claim_count": len(uncertain),
            "no_evidence_claim_count": len(no_evidence),
            "relation_unverified_count": len(relation_unverified),
            "numeric_contradiction_count": len(numeric_contradictions),
            "min_claim_score": min((c.score for c in self.claims), default=1.0),
            "score_variance": _variance([c.score for c in self.claims]),
            "contradicted_by_evidence_count": sum(
                1 for c in self.claims
                if any(e.entailment_status == EntailmentStatus.CONTRADICTS for e in c.evidence)
            ),
            "claim_type_breakdown": type_breakdown,
        }
        if self.overall_score is not None:
            self.factuality_percentage = round(self.overall_score * 100.0, 1)
            self.factuality_rating = _score_to_rating(self.factuality_percentage)
        else:
            self.factuality_percentage = None
            self.factuality_rating = None

        sources: list[str] = []
        seen = set()
        for ev in self.global_evidence:
            if ev.source_url not in seen:
                seen.add(ev.source_url)
                sources.append(ev.source_url)
        self.sources_used = sources
        return self


class OllamaRunResponse(BaseModel):
    job_id: UUID
    model: str
    prompt: str
    model_output: str
    report: VerificationReport


# ---------------------------------------------------------------------------
# Async job status schemas
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatusResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    result: VerificationReport | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Prometheus / observability
# ---------------------------------------------------------------------------


class MetricsSummary(BaseModel):
    total_jobs_processed: int
    average_latency_ms: float
    hallucination_rate: float
    cache_hit_rate: float
    no_evidence_rate: float = 0.0
    contradiction_rate: float = 0.0
    relation_verification_rate: float = 0.0
    numeric_contradiction_rate: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _score_to_rating(score_percent: float | None) -> str | None:
    if score_percent is None:
        return None
    if score_percent <= 20.0:
        return "Insufficient Evidence"
    if score_percent >= 85.0:
        return "Highly Reliable"
    if score_percent >= 70.0:
        return "Mostly True"
    if score_percent >= 50.0:
        return "Acceptable / Mixed"
    return "Likely Incorrect"

class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str

class OllamaGenerateResponse(BaseModel):
    model: str
    prompt: str
    model_output: str
