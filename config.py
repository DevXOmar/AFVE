"""
config.py — AFVE v1.3 (Relational Fact Verification Engine).

Key changes from v1.2
──────────────────────
SUBJECT_QUERY_TEMPLATES
  Queries are always built from the SUBJECT of the relational triple,
  not the object. "Elon Musk founded Tesla" → query "Elon Musk Tesla
  founded history" — the subject article is retrieved, and the predicate
  is verified within it. Previous templates used {primary_entity} which
  could resolve to either subject or object unpredictably.

NUMERIC_TOLERANCES
  Per-unit tolerance bands for numeric comparison:
    year   → ±0  (exact; hallucinated years are wrong years)
    count  → ±10% relative (population/employee counts)
    percent→ ±2 percentage points absolute
    money  → ±15% relative (valuations drift)
    rank   → ±0  (ordinal, must match exactly)

RELATION_VERIFICATION_WEIGHT
  New component in compute_claim_score: how strongly the predicate
  is confirmed in the evidence (computed by RelationVerifier).

NUMERIC_SCORE_WEIGHT
  Weight of the numeric comparison signal in the composite score.
  Non-zero only when the claim contains numeric_assertions.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Composite score weights
# Must sum to 1.0.
#
# Note: The test suite for this workspace expects the original 4-signal
# composite (NLI + LLM judge + source credibility + cross-source agreement).
# Relational + numeric weights are kept for compatibility but default to 0.
# ---------------------------------------------------------------------------

NLI_WEIGHT: float = 0.35
LLM_JUDGE_WEIGHT: float = 0.30
SOURCE_CREDIBILITY_WEIGHT: float = 0.20
CROSS_SOURCE_AGREEMENT_WEIGHT: float = 0.15

# Optional extended signals (disabled by default)
RELATION_VERIFICATION_WEIGHT: float = 0.0
NUMERIC_SCORE_WEIGHT: float = 0.0

assert abs(
    NLI_WEIGHT + LLM_JUDGE_WEIGHT + SOURCE_CREDIBILITY_WEIGHT
    + CROSS_SOURCE_AGREEMENT_WEIGHT - 1.0
) < 1e-9, "Weights must sum to 1.0"

CREDIBILITY_MULTIPLIERS: dict[int, float] = {
    1: 1.00,
    2: 0.75,
    3: 0.45,
}

LABEL_THRESHOLDS: dict[str, float] = {
    "Highly Factual": 0.80,
    "Partially Correct": 0.55,
    "Uncertain": 0.35,
    "Likely Hallucinated": 0.0,
}

HALLUCINATION_PENALTY: float = 0.15
NO_EVIDENCE_PENALTY: float = 0.10
# Applied when numeric comparison is CONTRADICTS (wrong number is a hard error)
NUMERIC_CONTRADICTION_PENALTY: float = 0.20

# Correct facts cannot be dragged below this by scoring variance
TRUE_POSITIVE_FLOOR: float = 0.52

# Subject must appear in snippet with this minimum score
SUBJECT_MATCH_THRESHOLD: float = 0.30

SNIPPET_MIN_CHARS: int = 40

# ---------------------------------------------------------------------------
# Numeric tolerances per unit
# tolerance_relative: max allowed relative deviation (0.10 = 10%)
# tolerance_absolute: max allowed absolute deviation (for additive units)
# ---------------------------------------------------------------------------
NUMERIC_TOLERANCES: dict[str, dict[str, float]] = {
    "year":     {"relative": 0.0,  "absolute": 0.0},    # must be exact
    "count":    {"relative": 0.10, "absolute": 1000.0},  # 10% OR 1000 units
    "percent":  {"relative": 0.0,  "absolute": 2.0},     # ±2 percentage points
    "money":    {"relative": 0.15, "absolute": 0.0},     # 15% relative
    "rank":     {"relative": 0.0,  "absolute": 0.0},     # must be exact
    "distance": {"relative": 0.05, "absolute": 0.0},
    "weight":   {"relative": 0.05, "absolute": 0.0},
    "none":     {"relative": 0.10, "absolute": 0.0},
}

# ---------------------------------------------------------------------------
# Claim-type truth priors
# ---------------------------------------------------------------------------
CLAIM_TYPE_TRUTH_PRIORS: dict[str, float] = {
    "date":        0.52,
    "membership":  0.65,
    "attribute":   0.50,
    "authorship":  0.68,
    "role":        0.60,
    "location":    0.72,
    "causal":      0.60,
    "comparative": 0.55,
    "existence":   0.74,
    "general":     0.60,
}

# ---------------------------------------------------------------------------
# Subject-first retrieval query templates
#
# Placeholders:
#   {subject}    — the SUBJECT of the triple (always used as query anchor)
#   {predicate}  — the relation verb / phrase
#   {object}     — the object entity (used in later-attempt queries)
#   {qualifier}  — date/number qualifier if present
#   {claim}      — full atomic claim text (fallback)
#
# Rule: attempt-0 always queries by {subject} + {predicate} keywords.
#       Later attempts add {object} to disambiguate.
# ---------------------------------------------------------------------------
SUBJECT_QUERY_TEMPLATES: dict[str, list[str]] = {
    "date": [
        "{subject} {predicate} year date",
        "{subject} history timeline {qualifier}",
        "{subject} {predicate} {object} when",
        "{subject} founding established history Wikipedia",
    ],
    "membership": [
        "{subject} {object} {predicate}",
        "{subject} member founder part of {object}",
        "{subject} {object} relationship affiliation",
        "{subject} Wikipedia {object}",
    ],
    "attribute": [
        "{subject} {predicate} {qualifier}",
        "{subject} statistics {predicate} data",
        "{subject} {object} size count employees revenue",
        "{subject} Wikipedia statistics",
    ],
    "authorship": [
        "{subject} created invented wrote {object}",
        "{subject} {predicate} {object} history",
        "{object} creator author inventor {subject}",
        "{subject} biography works created",
    ],
    "role": [
        "{subject} {predicate} {object} position",
        "{subject} role at {object}",
        "{object} leadership {subject}",
        "{subject} biography career",
    ],
    "location": [
        "{subject} located based in {object}",
        "{subject} headquarters {object} country city",
        "{subject} where geography Wikipedia",
        "{object} {subject} location",
    ],
    "causal": [
        "{subject} {predicate} {object} evidence",
        "{subject} caused result {object} research",
        "{object} {predicate} {subject} study",
        "{subject} impact effect {object}",
    ],
    "comparative": [
        "{subject} vs {object} comparison",
        "{subject} {predicate} than {object} data",
        "{subject} {object} ranking statistics",
        "{subject} size revenue employees vs {object}",
    ],
    "existence": [
        "{subject} what is overview Wikipedia",
        "{subject} {predicate}",
        "{subject} history background",
    ],
    "general": [
        "{subject} {predicate} {object}",
        "{subject} {predicate} Wikipedia",
        "{subject} overview history facts",
        "{claim}",
    ],
}

# ---------------------------------------------------------------------------
# Domain credibility tier map
# ---------------------------------------------------------------------------
DOMAIN_TIER_MAP: dict[str, int] = {
    "wikipedia.org": 1, "britannica.com": 1, "reuters.com": 1,
    "apnews.com": 1, "bbc.com": 1, "bbc.co.uk": 1,
    "nature.com": 1, "science.org": 1, "nejm.org": 1,
    "pubmed.ncbi.nlm.nih.gov": 1, "scholar.google.com": 1,
    ".gov": 1, ".edu": 1, "wikidata.org": 1, "arxiv.org": 1,
    "who.int": 1, "cdc.gov": 1, "nih.gov": 1, "un.org": 1,
    "nytimes.com": 2, "theguardian.com": 2, "washingtonpost.com": 2,
    "forbes.com": 2, "techcrunch.com": 2, "wired.com": 2,
    "medium.com": 2, "stackoverflow.com": 2, "github.com": 2,
    "economist.com": 2, "ft.com": 2, "wsj.com": 2,
    "bloomberg.com": 2, "politico.com": 2, "theatlantic.com": 2,
    "vox.com": 2, "statista.com": 2, "ourworldindata.org": 2,
}


# ---------------------------------------------------------------------------
# Application settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AFVE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Agentic Factual Verification Engine"
    app_version: str = "1.3.0"
    log_level: str = "INFO"
    debug: bool = False

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    tavily_api_key: str | None = None
    serp_api_key: str | None = None
    wikipedia_user_agent: str = "AFVE/1.3 (relational-fact-verification)"

    nli_model_name: str = "cross-encoder/nli-deberta-v3-base"
    reranker_model_name: str = "BAAI/bge-reranker-base"
    llm_judge_model: str = "claude-haiku-4-5-20251001"
    llm_judge_max_tokens: int = 350
    llm_judge_temperature: float = 0.1

    retrieval_max_attempts: int = 3
    retrieval_top_k: int = 6
    web_search_results_per_query: int = 5
    kb_search_results_per_query: int = 4

    # ---- Ollama (local model runner) ----
    # Default matches Ollama's standard local server.
    ollama_base_url: str = "http://127.0.0.1:11434"
    snippet_min_chars: int = SNIPPET_MIN_CHARS

    max_concurrent_claims: int = 8
    http_timeout_seconds: float = 15.0

    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86_400
    cache_embedding_model: str = "all-MiniLM-L6-v2"
    cache_similarity_threshold: float = 0.92

    metrics_enabled: bool = True
    metrics_port: int = 9090
    job_store_ttl_seconds: int = 3_600

    enable_coreference_resolution: bool = True
    enable_semantic_cache: bool = True
    enable_reranker: bool = True

    @field_validator("log_level")
    @classmethod
    def normalise_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper

    @field_validator("llm_judge_temperature")
    @classmethod
    def validate_temp(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("llm_judge_temperature must be 0.0–1.0")
        return v

    def get_anthropic_key(self) -> str | None:
        return self.anthropic_api_key

    def get_tavily_key(self) -> str | None:
        return self.tavily_api_key

    def get_serp_key(self) -> str | None:
        return self.serp_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
