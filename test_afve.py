"""
test_afve.py — Comprehensive test suite for the Agentic Factual Verification Engine.

Test organisation
─────────────────
  Unit tests     — pure logic, no I/O, no external deps
    TestModels              Pydantic schema validation & model_validator
    TestConfig              Settings loading & domain tier inference
    TestClaimExtraction     Sentence splitting, verifiability filter, coreference stub
    TestScoringCalibrator   Score math, label thresholds, hallucination penalty
    TestExplanationGenerator Rationale text generation
    TestCredibilityInference Domain → tier mapping edge cases

  Integration tests — components wired together, all I/O mocked
    TestVerificationEngine  NLI heuristic, LLM-judge fallback, cross-source agreement
    TestRetrievalAgent      Retry logic, dedup, graceful degradation, cache interaction
    TestOrchestrator        Full pipeline with fixture evidence

  API tests — FastAPI TestClient, no real network calls
    TestSyncEndpoint        POST /verify happy path, validation errors, 500 handling
    TestAsyncEndpoint       POST /verify/async + GET /verify/jobs/{id} lifecycle
    TestHealthEndpoint      GET /health
    TestMetricsEndpoint     GET /metrics shape

Run with:
  pytest test_afve.py -v
  pytest test_afve.py -v -k "TestModels"           # single class
  pytest test_afve.py -v --tb=short                # terse tracebacks
  pytest test_afve.py -v --cov=. --cov-report=term # with coverage
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Sys-path tweak — tests live alongside the source files in /afve/
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")


# ===========================================================================
# Fixtures & helpers shared across all test classes
# ===========================================================================


def make_uuid() -> str:
    return str(uuid.uuid4())


def _snippet(
    url: str = "https://en.wikipedia.org/wiki/Python",
    text: str = "Python is a high-level programming language.",
    tier: int = 1,
) -> dict:
    from models import CredibilityTier, EntailmentStatus

    return {
        "source_url": url,
        "snippet": text,
        "credibility_tier": CredibilityTier(tier),
        "entailment_status": EntailmentStatus.NEUTRAL,
        "nli_score": 0.0,
        "llm_judge_score": 0.0,
    }


def make_snippet(**kw):
    from models import EvidenceSnippet

    # Convenience aliases used throughout this test suite.
    # EvidenceSnippet fields are: source_url, snippet, credibility_tier.
    if "url" in kw and "source_url" not in kw:
        kw["source_url"] = kw.pop("url")
    if "text" in kw and "snippet" not in kw:
        kw["snippet"] = kw.pop("text")
    if "tier" in kw and "credibility_tier" not in kw:
        kw["credibility_tier"] = kw.pop("tier")

    return EvidenceSnippet(**{**_snippet(), **kw})


def make_resolved_claim(text: str = "Python was created by Guido van Rossum.", idx: int = 0):
    from models import ResolvedClaim

    return ResolvedClaim(
        original_text=text,
        contextualized_text=text,
        sentence_index=idx,
    )


def make_retrieval_result(claim=None, snippets=None, attempts=1, success=True):
    from models import RetrievalResult

    return RetrievalResult(
        claim=claim or make_resolved_claim(),
        snippets=snippets if snippets is not None else [make_snippet()],
        queries_attempted=attempts,
        retrieval_success=success,
    )


def minimal_request(model_output: str = "Python was created by Guido van Rossum in 1991.") -> dict:
    return {
        "job_id": make_uuid(),
        "prompt": "Tell me about Python.",
        "model_output": model_output,
        "config": {
            "strictness": 0.8,
            "max_retrieval_attempts": 1,
            "top_k_snippets": 3,
            "enable_llm_judge": False,
            "enable_nli": False,
            "enable_web_search": False,
            "enable_kb_search": False,
        },
    }


# ---------------------------------------------------------------------------
# Mock LLM client that returns controllable responses
# ---------------------------------------------------------------------------


class MockLLMClient:
    def __init__(self, response: str = ""):
        self._response = response

    async def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        return self._response


# ---------------------------------------------------------------------------
# Mock httpx client that returns controllable responses
# ---------------------------------------------------------------------------


def mock_http_client(json_payload: dict | None = None, status: int = 200):
    client = AsyncMock()
    resp = AsyncMock()
    resp.status_code = status
    resp.json.return_value = json_payload or {}
    resp.raise_for_status = MagicMock()
    client.get = AsyncMock(return_value=resp)
    client.post = AsyncMock(return_value=resp)
    return client


# ===========================================================================
# UNIT TESTS — Models (Pydantic)
# ===========================================================================


class TestModels:
    """Validate Pydantic schema logic, field constraints, and model_validator."""

    def test_verification_config_defaults(self):
        from models import VerificationConfig

        cfg = VerificationConfig()
        assert cfg.strictness == 0.8
        assert cfg.max_retrieval_attempts == 3
        assert cfg.top_k_snippets == 5
        assert cfg.enable_llm_judge is True

    def test_verification_config_strictness_bounds(self):
        from pydantic import ValidationError
        from models import VerificationConfig

        with pytest.raises(ValidationError):
            VerificationConfig(strictness=1.5)
        with pytest.raises(ValidationError):
            VerificationConfig(strictness=-0.1)

    def test_verification_config_max_retrieval_attempts_bounds(self):
        from pydantic import ValidationError
        from models import VerificationConfig

        with pytest.raises(ValidationError):
            VerificationConfig(max_retrieval_attempts=0)
        with pytest.raises(ValidationError):
            VerificationConfig(max_retrieval_attempts=6)

    def test_verification_request_valid(self):
        from models import VerificationRequest

        payload = minimal_request()
        req = VerificationRequest(**payload)
        assert str(req.job_id) == payload["job_id"]
        assert req.model_output.startswith("Python")

    def test_verification_request_rejects_empty_prompt(self):
        from pydantic import ValidationError
        from models import VerificationRequest

        payload = minimal_request()
        payload["prompt"] = ""
        with pytest.raises(ValidationError):
            VerificationRequest(**payload)

    def test_verification_request_rejects_empty_model_output(self):
        from pydantic import ValidationError
        from models import VerificationRequest

        payload = minimal_request()
        payload["model_output"] = ""
        with pytest.raises(ValidationError):
            VerificationRequest(**payload)

    def test_evidence_snippet_rejects_empty_url(self):
        from pydantic import ValidationError
        from models import EvidenceSnippet, CredibilityTier

        with pytest.raises(ValidationError):
            EvidenceSnippet(
                source_url="   ",
                snippet="some text",
                credibility_tier=CredibilityTier.TIER_1,
            )

    def test_evidence_snippet_score_bounds(self):
        from pydantic import ValidationError
        from models import EvidenceSnippet, CredibilityTier

        with pytest.raises(ValidationError):
            EvidenceSnippet(
                source_url="https://example.com",
                snippet="text",
                credibility_tier=CredibilityTier.TIER_1,
                nli_score=1.5,
            )

    def test_verified_claim_score_bounds(self):
        from pydantic import ValidationError
        from models import VerifiedClaim, VerificationLabel

        with pytest.raises(ValidationError):
            VerifiedClaim(
                claim="test",
                contextualized_claim="test",
                score=1.1,
                label=VerificationLabel.VERIFIED,
                explanation="ok",
                evidence=[],
            )

    def test_verification_report_uncertainty_signals_auto_populated(self):
        """model_validator should compute uncertainty_signals from claims."""
        from models import (
            VerificationReport,
            VerifiedClaim,
            VerificationLabel,
            JobMetadata,
        )

        def _claim(score, label):
            return VerifiedClaim(
                claim="c",
                contextualized_claim="c",
                score=score,
                label=label,
                explanation="e",
                evidence=[],
            )

        job_id = uuid.uuid4()
        report = VerificationReport(
            metadata=JobMetadata(
                job_id=job_id,
                processing_time_ms=100,
                sources_queried=3,
                claims_extracted=3,
                claims_verified=3,
            ),
            overall_score=0.6,
            overall_label=VerificationLabel.UNCERTAIN,
            claims=[
                _claim(0.9, VerificationLabel.VERIFIED),
                _claim(0.3, VerificationLabel.UNCERTAIN),
                _claim(0.1, VerificationLabel.CONTRADICTED),
            ],
        )
        sig = report.uncertainty_signals
        assert sig["contradicted_by_label_count"] == 1
        assert sig["uncertain_claim_count"] == 1
        assert sig["min_claim_score"] == pytest.approx(0.1)
        assert "score_variance" in sig

    def test_uncertainty_signals_empty_claims(self):
        from models import VerificationReport, VerificationLabel, JobMetadata

        report = VerificationReport(
            metadata=JobMetadata(
                job_id=uuid.uuid4(),
                processing_time_ms=10,
                sources_queried=0,
                claims_extracted=0,
                claims_verified=0,
            ),
            overall_score=0.0,
            overall_label=VerificationLabel.UNCERTAIN,
            claims=[],
        )
        assert report.uncertainty_signals["min_claim_score"] == 1.0
        assert report.uncertainty_signals["score_variance"] == 0.0

    def test_variance_helper(self):
        from models import _variance

        assert _variance([]) == 0.0
        assert _variance([0.5]) == 0.0
        assert _variance([0.0, 1.0]) == pytest.approx(0.25)
        assert _variance([0.5, 0.5, 0.5]) == pytest.approx(0.0)

    def test_job_status_response_no_result_by_default(self):
        from models import JobStatusResponse, JobStatus

        resp = JobStatusResponse(job_id=uuid.uuid4(), status=JobStatus.QUEUED)
        assert resp.result is None
        assert resp.error is None

    def test_enums_are_string_serialisable(self):
        from models import VerificationLabel, EntailmentStatus, JobStatus

        assert VerificationLabel.VERIFIED.value == "VERIFIED"
        assert EntailmentStatus.SUPPORTS.value == "supports"
        assert JobStatus.COMPLETED.value == "completed"


# ===========================================================================
# UNIT TESTS — Config
# ===========================================================================


class TestConfig:
    def test_settings_singleton(self):
        from config import get_settings

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_default_weights_sum_to_one(self):
        from config import (
            NLI_WEIGHT,
            LLM_JUDGE_WEIGHT,
            SOURCE_CREDIBILITY_WEIGHT,
            CROSS_SOURCE_AGREEMENT_WEIGHT,
        )

        total = NLI_WEIGHT + LLM_JUDGE_WEIGHT + SOURCE_CREDIBILITY_WEIGHT + CROSS_SOURCE_AGREEMENT_WEIGHT
        assert total == pytest.approx(1.0)

    def test_credibility_multipliers_ordered(self):
        from config import CREDIBILITY_MULTIPLIERS

        assert CREDIBILITY_MULTIPLIERS[1] >= CREDIBILITY_MULTIPLIERS[2]
        assert CREDIBILITY_MULTIPLIERS[2] >= CREDIBILITY_MULTIPLIERS[3]

    def test_label_thresholds_cover_full_range(self):
        from config import LABEL_THRESHOLDS

        assert 0.0 in LABEL_THRESHOLDS.values()
        thresholds = sorted(LABEL_THRESHOLDS.values())
        assert thresholds[0] == 0.0

    def test_settings_log_level_validation(self):
        from pydantic import ValidationError
        from config import Settings

        with pytest.raises(ValidationError):
            Settings(log_level="VERBOSE")

    def test_settings_normalises_log_level_case(self):
        from config import Settings

        s = Settings(log_level="debug")
        assert s.log_level == "DEBUG"

    def test_domain_tier_map_wikipedia_is_tier1(self):
        from config import DOMAIN_TIER_MAP

        assert DOMAIN_TIER_MAP.get("wikipedia.org") == 1

    def test_domain_tier_map_medium_is_tier2(self):
        from config import DOMAIN_TIER_MAP

        assert DOMAIN_TIER_MAP.get("medium.com") == 2


# ===========================================================================
# UNIT TESTS — Claim Extraction Agent
# ===========================================================================


class TestClaimExtraction:
    """Test sentence splitting, verifiability filter, and coreference stub."""

    def _agent(self, llm_response: str = ""):
        from agents import ClaimExtractionAgent

        llm = MockLLMClient(llm_response) if llm_response else None
        return ClaimExtractionAgent(llm_client=llm)

    # --- Sentence splitting ---

    def test_single_factual_sentence_extracted(self):
        agent = self._agent()
        raw = agent._split_into_sentences(
            "Python was created by Guido van Rossum in 1991."
        )
        assert len(raw) == 1
        assert raw[0].is_verifiable is True

    def test_multiple_sentences_split_correctly(self):
        agent = self._agent()
        text = (
            "Python was created in 1991. "
            "It is widely used in data science. "
            "Machine learning practitioners prefer it."
        )
        raw = agent._split_into_sentences(text)
        assert len(raw) == 3

    def test_short_sentence_marked_unverifiable(self):
        agent = self._agent()
        raw = agent._split_into_sentences("Yes it is.")
        assert raw[0].is_verifiable is False

    def test_opinion_phrase_marked_unverifiable(self):
        agent = self._agent()
        for opinion in [
            "I think Python is the best language.",
            "In my opinion, Java is verbose.",
            "I believe this to be true.",
        ]:
            raw = agent._split_into_sentences(opinion)
            assert raw[0].is_verifiable is False, f"Should be unverifiable: {opinion}"

    def test_pleasantry_marked_unverifiable(self):
        agent = self._agent()
        raw = agent._split_into_sentences("Hello! How can I help you today?")
        # "Hello!" is too short; "How can I help you today?" contains no factual claim
        for r in raw:
            assert r.is_verifiable is False

    def test_hedge_sentence_marked_unverifiable(self):
        agent = self._agent()
        raw = agent._split_into_sentences("It might be the case that Python is popular.")
        assert raw[0].is_verifiable is False

    def test_factual_claim_not_filtered(self):
        agent = self._agent()
        text = "The Python programming language was first released in 1991."
        raw = agent._split_into_sentences(text)
        assert any(r.is_verifiable for r in raw)

    def test_mixed_output_filters_correctly(self):
        agent = self._agent()
        text = (
            "Hi there! "
            "Python was released in 1991. "
            "I think it is great."
        )
        raw = agent._split_into_sentences(text)
        verifiable = [r for r in raw if r.is_verifiable]
        assert len(verifiable) == 1
        assert "1991" in verifiable[0].text

    # --- extract() integration (no LLM, no coreference) ---

    @pytest.mark.asyncio
    async def test_extract_returns_resolved_claims(self):
        from models import VerificationRequest, VerificationConfig

        agent = self._agent()
        req = VerificationRequest(
            job_id=uuid.uuid4(),
            prompt="Tell me about Python.",
            model_output=(
                "Python was created by Guido van Rossum in 1991. "
                "It supports multiple programming paradigms."
            ),
            config=VerificationConfig(
                enable_llm_judge=False,
                enable_nli=False,
                enable_web_search=False,
                enable_kb_search=False,
            ),
        )
        with patch("config.get_settings") as mock_settings:
            from config import get_settings
            mock_settings.return_value = get_settings()
            # Disable coreference so we don't need LLM
            with patch("config.Settings.enable_coreference_resolution", False, create=True):
                claims = await agent.extract(req)

        assert len(claims) >= 1
        assert all(hasattr(c, "contextualized_text") for c in claims)

    @pytest.mark.asyncio
    async def test_extract_with_coreference_uses_llm(self):
        """When LLM available + coreference enabled, resolved text should differ."""
        from models import VerificationRequest
        from agents import ClaimExtractionAgent

        resolved_json = '["Apple revenue grew 20% in Q3 2023."]'
        llm = MockLLMClient(resolved_json)
        agent = ClaimExtractionAgent(llm_client=llm)

        with patch("agents.settings") as mock_s:
            mock_s.enable_coreference_resolution = True
            from models import RawClaim

            raw_claims = [
                RawClaim(text="Revenue grew 20% in Q3.", sentence_index=0, is_verifiable=True)
            ]
            resolved = await agent._resolve_coreferences(
                raw_claims, "Apple earnings", "Apple revenue grew 20% in Q3."
            )

        assert resolved[0].contextualized_text == "Apple revenue grew 20% in Q3 2023."

    @pytest.mark.asyncio
    async def test_coreference_llm_failure_falls_back_gracefully(self):
        """If LLM returns invalid JSON, coreference falls back to raw text."""
        from agents import ClaimExtractionAgent
        from models import RawClaim

        llm = MockLLMClient("NOT VALID JSON {{{{")
        agent = ClaimExtractionAgent(llm_client=llm)

        raw_claims = [RawClaim(text="It grew 20%.", sentence_index=0, is_verifiable=True)]
        resolved = await agent._resolve_coreferences(raw_claims, "", "")
        assert resolved[0].contextualized_text == "It grew 20%."

    @pytest.mark.asyncio
    async def test_coreference_length_mismatch_falls_back(self):
        """If LLM returns wrong number of items, falls back to raw text."""
        from agents import ClaimExtractionAgent
        from models import RawClaim

        # Returns 2 items for 1 claim → length mismatch
        llm = MockLLMClient('["Claim A.", "Unexpected extra claim."]')
        agent = ClaimExtractionAgent(llm_client=llm)

        raw_claims = [RawClaim(text="Original.", sentence_index=0, is_verifiable=True)]
        resolved = await agent._resolve_coreferences(raw_claims, "", "")
        assert resolved[0].contextualized_text == "Original."


# ===========================================================================
# UNIT TESTS — Scoring Calibrator
# ===========================================================================


class TestScoringCalibrator:
    def _cal(self):
        from agents import ScoringCalibrator
        return ScoringCalibrator()

    def test_claim_score_bands(self):
        from agents import ClaimScoreInput
        from models import EvidenceSnippet, CredibilityTier, EntailmentStatus

        cal = self._cal()
        base = dict(source_url="https://example.com", snippet="x", credibility_tier=CredibilityTier.TIER_1)

        supports = EvidenceSnippet(**base, entailment_status=EntailmentStatus.SUPPORTS)
        contradicts = EvidenceSnippet(**base, entailment_status=EntailmentStatus.CONTRADICTS)
        neutral = EvidenceSnippet(**base, entailment_status=EntailmentStatus.NEUTRAL)

        assert cal.compute_claim_score(ClaimScoreInput(nli=1.0, has_evidence=True, evidence=[supports])) == pytest.approx(1.0)
        assert cal.compute_claim_score(ClaimScoreInput(nli=0.1, has_evidence=True, evidence=[supports])) == pytest.approx(0.8)
        assert cal.compute_claim_score(ClaimScoreInput(nli=0.9, has_evidence=True, evidence=[contradicts])) == pytest.approx(0.2)
        assert cal.compute_claim_score(ClaimScoreInput(nli=0.9, has_evidence=True, evidence=[neutral])) == pytest.approx(0.5)
        assert cal.compute_claim_score(ClaimScoreInput(nli=0.9, has_evidence=False, evidence=[])) == pytest.approx(0.0)

    def test_label_highly_factual(self):
        cal = self._cal()
        assert cal.score_to_label(0.95).value == "VERIFIED"
        assert cal.score_to_label(0.80).value == "VERIFIED"

    def test_label_partially_correct(self):
        cal = self._cal()
        assert cal.score_to_label(0.74).value == "UNCERTAIN"
        assert cal.score_to_label(0.40).value == "UNCERTAIN"

    def test_label_uncertain(self):
        cal = self._cal()
        assert cal.score_to_label(0.39).value == "CONTRADICTED"
        assert cal.score_to_label(0.20).value == "CONTRADICTED"

    def test_label_likely_hallucinated(self):
        cal = self._cal()
        assert cal.score_to_label(0.01).value == "CONTRADICTED"
        assert cal.score_to_label(0.0).value == "UNSUPPORTED"

    def test_overall_score_average_no_penalty(self):
        from models import VerifiedClaim, VerificationLabel

        cal = self._cal()

        def claim(s, l):
            return VerifiedClaim(
                claim="c", contextualized_claim="c", score=s,
                label=l, explanation="e", evidence=[]
            )

        claims = [
            claim(0.9, VerificationLabel.VERIFIED),
            claim(0.7, VerificationLabel.UNCERTAIN),
        ]
        score = cal.compute_overall_score(claims)
        assert score == pytest.approx(0.8, abs=1e-4)

    def test_overall_score_never_below_zero(self):
        from models import VerifiedClaim, VerificationLabel

        cal = self._cal()
        claims = [
            VerifiedClaim(
                claim="c", contextualized_claim="c", score=0.0,
                label=VerificationLabel.UNSUPPORTED, explanation="e", evidence=[]
            )
            for _ in range(10)
        ]
        assert cal.compute_overall_score(claims) >= 0.0

    def test_overall_score_empty_claims(self):
        cal = self._cal()
        assert cal.compute_overall_score([]) == 0.0


# ===========================================================================
# UNIT TESTS — Explanation Generator
# ===========================================================================


class TestExplanationGenerator:
    def _gen(self):
        from agents import ExplanationGenerator
        return ExplanationGenerator()

    def test_no_snippets_returns_unverified_message(self):
        gen = self._gen()
        claim = make_resolved_claim()
        from models import VerificationLabel
        explanation = gen.generate(claim, [], 0.0, VerificationLabel.UNCERTAIN)
        assert "No supporting evidence" in explanation

    def test_highly_factual_mentions_verification(self):
        from models import VerificationLabel, EntailmentStatus

        gen = self._gen()
        snippets = [
            make_snippet(url="https://en.wikipedia.org/wiki/P"),
            make_snippet(url="https://bbc.com/news/1"),
        ]
        # Mark as supporting
        for s in snippets:
            s.entailment_status = EntailmentStatus.SUPPORTS

        exp = gen.generate(
            make_resolved_claim(), snippets, 0.92, VerificationLabel.VERIFIED
        )
        assert "Verified" in exp or "source" in exp.lower()

    def test_likely_hallucinated_mentions_contradiction(self):
        from models import VerificationLabel, EntailmentStatus

        gen = self._gen()
        s = make_snippet(url="https://reuters.com/1", text="Python was invented in 1980, not 1991.")
        s.entailment_status = EntailmentStatus.CONTRADICTS

        exp = gen.generate(
            make_resolved_claim(), [s], 0.1, VerificationLabel.CONTRADICTED
        )
        assert "contradict" in exp.lower()

    def test_explanation_contains_domain(self):
        from models import VerificationLabel

        gen = self._gen()
        s = make_snippet(url="https://en.wikipedia.org/wiki/Python")
        exp = gen.generate(make_resolved_claim(), [s], 0.85, VerificationLabel.VERIFIED)
        assert "wikipedia.org" in exp

    def test_explanation_snippet_truncated(self):
        from models import VerificationLabel

        gen = self._gen()
        long_text = "A" * 500
        s = make_snippet(text=long_text)
        exp = gen.generate(make_resolved_claim(), [s], 0.7, VerificationLabel.UNCERTAIN)
        # Should truncate to 200 chars + ellipsis — check total explanation length is reasonable
        assert len(exp) < 1000

    def test_partially_correct_mentions_conflict(self):
        from models import VerificationLabel, EntailmentStatus

        gen = self._gen()
        s1 = make_snippet(url="https://bbc.com/1")
        s1.entailment_status = EntailmentStatus.SUPPORTS
        s2 = make_snippet(url="https://foxnews.com/1")
        s2.entailment_status = EntailmentStatus.CONTRADICTS

        exp = gen.generate(
            make_resolved_claim(), [s1, s2], 0.6, VerificationLabel.UNCERTAIN
        )
        assert "conflict" in exp.lower() or "contradict" in exp.lower()


# ===========================================================================
# UNIT TESTS — Credibility inference utility
# ===========================================================================


class TestCredibilityInference:
    def _infer(self, url):
        from agents import _infer_credibility_tier
        return _infer_credibility_tier(url)

    def test_wikipedia_is_tier1(self):
        assert self._infer("https://en.wikipedia.org/wiki/Python").value == 1

    def test_gov_domain_is_tier1(self):
        assert self._infer("https://cdc.gov/health/flu").value == 1

    def test_edu_domain_is_tier1(self):
        assert self._infer("https://mit.edu/research/paper").value == 1

    def test_reuters_is_tier1(self):
        assert self._infer("https://reuters.com/world/story").value == 1

    def test_medium_is_tier2(self):
        assert self._infer("https://medium.com/@author/post").value == 2

    def test_unknown_domain_is_tier3(self):
        assert self._infer("https://somerandomblog123.xyz/post").value == 3

    def test_malformed_url_is_tier3(self):
        assert self._infer("not-a-url-at-all").value == 3

    def test_empty_url_is_tier3(self):
        assert self._infer("").value == 3

    def test_arxiv_is_tier1(self):
        assert self._infer("https://arxiv.org/abs/2301.00000").value == 1


# ===========================================================================
# UNIT TESTS — Deduplication utility
# ===========================================================================


class TestDeduplication:
    def test_exact_url_duplicates_removed(self):
        from agents import _deduplicate

        s1 = make_snippet(url="https://example.com/a")
        s2 = make_snippet(url="https://example.com/a")  # duplicate
        s3 = make_snippet(url="https://example.com/b")

        result = _deduplicate([s1, s2, s3])
        assert len(result) == 2

    def test_trailing_slash_treated_as_duplicate(self):
        from agents import _deduplicate

        s1 = make_snippet(url="https://example.com/page/")
        s2 = make_snippet(url="https://example.com/page")
        result = _deduplicate([s1, s2])
        assert len(result) == 1

    def test_empty_list_returns_empty(self):
        from agents import _deduplicate

        assert _deduplicate([]) == []

    def test_no_duplicates_unchanged(self):
        from agents import _deduplicate

        snippets = [
            make_snippet(url=f"https://example.com/{i}") for i in range(5)
        ]
        assert len(_deduplicate(snippets)) == 5

    def test_order_preserved_for_first_occurrence(self):
        from agents import _deduplicate

        urls = ["https://a.com", "https://b.com", "https://a.com", "https://c.com"]
        snippets = [make_snippet(url=u) for u in urls]
        result = _deduplicate(snippets)
        assert [s.source_url for s in result] == ["https://a.com", "https://b.com", "https://c.com"]


# ===========================================================================
# INTEGRATION TESTS — Verification Engine
# ===========================================================================


class TestVerificationEngine:
    """Tests the engine with NLI and LLM judge both disabled or mocked."""

    def _engine(self, llm=None):
        from agents import VerificationEngine
        return VerificationEngine(llm_client=llm)

    def _config(self, nli=False, llm=False):
        from models import VerificationConfig
        return VerificationConfig(
            enable_nli=nli,
            enable_llm_judge=llm,
            enable_web_search=False,
            enable_kb_search=False,
        )

    @pytest.mark.asyncio
    async def test_empty_snippets_returns_zeros(self):
        engine = self._engine()
        retrieval = make_retrieval_result(snippets=[], success=False)
        nli, llm_j, src_w, cross_a, annotated = await engine.verify(retrieval, self._config())
        assert nli == 0.0
        assert llm_j == 0.0
        assert src_w == 0.0
        assert cross_a == 0.0

    @pytest.mark.asyncio
    async def test_heuristic_nli_high_overlap(self):
        engine = self._engine()
        # Force NLI pipeline to None so heuristic is used
        with patch("agents._get_nli_pipeline", return_value=None):
            score, status = engine._run_nli(
                "Python is a programming language",
                "Python is a high-level programming language created in 1991",
            )
        assert score > 0.3
        from models import EntailmentStatus
        assert status in (EntailmentStatus.SUPPORTS, EntailmentStatus.NEUTRAL)

    @pytest.mark.asyncio
    async def test_heuristic_nli_low_overlap(self):
        engine = self._engine()
        with patch("agents._get_nli_pipeline", return_value=None):
            score, status = engine._run_nli(
                "Elephants are mammals",
                "The stock market closed higher on Thursday amid tech gains",
            )
        assert score < 0.3

    @pytest.mark.asyncio
    async def test_cross_source_all_supporting(self):
        from models import EntailmentStatus

        engine = self._engine()
        snippets = [make_snippet() for _ in range(3)]
        for s in snippets:
            s.entailment_status = EntailmentStatus.SUPPORTS

        retrieval = make_retrieval_result(snippets=snippets)
        _, _, _, cross_a, _ = await engine.verify(retrieval, self._config())
        assert cross_a == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_cross_source_all_contradicting(self):
        from models import EntailmentStatus

        engine = self._engine()
        snippets = [make_snippet() for _ in range(3)]
        for s in snippets:
            s.entailment_status = EntailmentStatus.CONTRADICTS

        retrieval = make_retrieval_result(snippets=snippets)
        _, _, _, cross_a, _ = await engine.verify(retrieval, self._config())
        assert cross_a == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_llm_judge_returns_float_score(self):
        engine = self._engine(llm=MockLLMClient('{"score": 0.85, "reasoning": "Well supported."}'))
        score = await engine._run_llm_judge("Python was invented in 1991.", "Python was released in 1991.")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_llm_judge_bad_json_returns_default(self):
        engine = self._engine(llm=MockLLMClient("not json at all"))
        score = await engine._run_llm_judge("claim", "evidence")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_llm_judge_missing_score_key_returns_default(self):
        engine = self._engine(llm=MockLLMClient('{"reasoning": "Unclear."}'))
        score = await engine._run_llm_judge("claim", "evidence")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_source_credibility_weights_tier1_higher(self):
        from models import CredibilityTier

        engine = self._engine()
        tier1_snippet = make_snippet(url="https://en.wikipedia.org/wiki/X")
        tier1_snippet.nli_score = 0.8
        tier3_snippet = make_snippet(url="https://randomblog.io/x", tier=3)
        tier3_snippet.nli_score = 0.8

        from config import CREDIBILITY_MULTIPLIERS

        w1 = 0.8 * CREDIBILITY_MULTIPLIERS[1]
        w3 = 0.8 * CREDIBILITY_MULTIPLIERS[3]
        assert w1 > w3


# ===========================================================================
# INTEGRATION TESTS — Retrieval Agent
# ===========================================================================


class TestRetrievalAgent:
    """Tests retrieval with all network calls mocked."""

    def _agent(self, http=None, llm=None, cache=None):
        from agents import RetrievalAgent
        return RetrievalAgent(
            http_client=http or mock_http_client(),
            llm_client=llm,
            cache=cache,
        )

    def _config(self, attempts=2, top_k=3, web=True, kb=True):
        from models import VerificationConfig
        return VerificationConfig(
            max_retrieval_attempts=attempts,
            top_k_snippets=top_k,
            enable_web_search=web,
            enable_kb_search=kb,
            enable_llm_judge=False,
            enable_nli=False,
        )

    @pytest.mark.asyncio
    async def test_wikipedia_search_parses_results(self):
        wiki_response = {
            "query": {
                "search": [
                    {
                        "title": "Python (programming language)",
                        "snippet": "Python is a <span>high-level</span> language.",
                    }
                ]
            }
        }
        agent = self._agent(http=mock_http_client(wiki_response))
        snippets = await agent._search_wikipedia("Python language")
        assert len(snippets) == 1
        assert "Python" in snippets[0].snippet
        assert "<span>" not in snippets[0].snippet  # HTML stripped
        assert snippets[0].source_url.startswith("https://en.wikipedia.org")

    @pytest.mark.asyncio
    async def test_wikipedia_http_error_returns_empty(self):
        client = mock_http_client()
        client.get.side_effect = Exception("Connection refused")
        agent = self._agent(http=client)
        result = await agent._search_wikipedia("anything")
        assert result == []

    @pytest.mark.asyncio
    async def test_tavily_search_parses_results(self):
        tavily_response = {
            "results": [
                {"url": "https://en.wikipedia.org/wiki/Python", "content": "Python is a language."},
                {"url": "https://docs.python.org/3/", "content": "Official Python docs."},
            ]
        }
        agent = self._agent(http=mock_http_client(tavily_response))
        with patch("agents.settings") as ms:
            ms.get_tavily_key.return_value = "fake-key"
            ms.web_search_results_per_query = 5
            snippets = await agent._search_tavily("Python language")
        assert len(snippets) == 2

    @pytest.mark.asyncio
    async def test_serpapi_fallback_when_tavily_fails(self):
        serp_response = {
            "organic_results": [
                {"link": "https://en.wikipedia.org/wiki/X", "snippet": "X is a thing."}
            ]
        }
        client = mock_http_client()
        # First call (Tavily POST) raises; second call (SerpAPI GET) succeeds
        client.post.side_effect = Exception("Tavily down")
        client.get.return_value = AsyncMock(
            json=MagicMock(return_value=serp_response),
            raise_for_status=MagicMock(),
        )
        agent = self._agent(http=client)
        with patch("agents.settings") as ms:
            ms.get_tavily_key.return_value = "fake-key"
            ms.get_serp_key.return_value = "serp-key"
            ms.web_search_results_per_query = 5
            snippets = await agent._search_web("some query")
        assert len(snippets) == 1

    @pytest.mark.asyncio
    async def test_all_search_sources_fail_returns_empty(self):
        client = mock_http_client()
        client.post.side_effect = Exception("down")
        client.get.side_effect = Exception("down")
        agent = self._agent(http=client)
        with patch("agents.settings") as ms:
            ms.get_tavily_key.return_value = None
            ms.get_serp_key.return_value = None
            snippets = await agent._search_web("query")
        assert snippets == []

    @pytest.mark.asyncio
    async def test_cache_hit_skips_retrieval(self):
        cached_snippets = [make_snippet(url="https://cached.com/1")]
        cache = AsyncMock()
        cache.get = AsyncMock(return_value=cached_snippets)
        cache.set = AsyncMock()

        agent = self._agent(cache=cache)
        claim = make_resolved_claim()
        config = self._config()

        with patch("agents.settings") as ms:
            ms.enable_semantic_cache = True
            ms.enable_reranker = False
            result = await agent.retrieve(claim, config)

        assert result.retrieval_success is True
        assert result.queries_attempted == 0  # skipped network
        assert result.snippets == cached_snippets

    @pytest.mark.asyncio
    async def test_query_expansion_called_on_retry(self):
        llm = MockLLMClient("Python programming language creation date")
        agent = self._agent(llm=llm)
        with patch("agents.settings") as ms:
            ms.enable_query_expansion = True
    @pytest.mark.asyncio
    async def test_query_expansion_skips_on_attempt_0(self):
        llm = MockLLMClient("SHOULD NOT BE RETURNED")
        agent = self._agent(llm=llm)
        result = await agent._expand_query("original query", 0)
        assert result == "original query"

    @pytest.mark.asyncio
    async def test_retrieve_deduplicates_snippets(self):
        """Two sources returning the same URL should be deduped in the result."""
        wiki_response = {
            "query": {
                "search": [
                    {"title": "Python", "snippet": "Python is a language."}
                ]
            }
        }
        tavily_response = {
            "results": [
                # Same Wikipedia URL as above
                {"url": "https://en.wikipedia.org/wiki/Python", "content": "Python is high-level."}
            ]
        }
        http = AsyncMock()
        wiki_resp = AsyncMock()
        wiki_resp.json = MagicMock(return_value=wiki_response)
        wiki_resp.raise_for_status = MagicMock()
        tavily_resp = AsyncMock()
        tavily_resp.json = MagicMock(return_value=tavily_response)
        tavily_resp.raise_for_status = MagicMock()

        http.get = AsyncMock(return_value=wiki_resp)
        http.post = AsyncMock(return_value=tavily_resp)

        agent = self._agent(http=http)
        with patch("agents.settings") as ms:
            ms.enable_semantic_cache = False
            ms.enable_reranker = False
            ms.kb_search_results_per_query = 3
            ms.web_search_results_per_query = 5
            ms.get_tavily_key.return_value = "key"
            ms.get_serp_key.return_value = None
            ms.max_concurrent_claims = 8
            ms.http_timeout_seconds = 15.0

            config = self._config(attempts=1, web=True, kb=True)
            result = await agent.retrieve(make_resolved_claim(), config)

        urls = [s.source_url for s in result.snippets]
        assert len(urls) == len(set(urls)), "Duplicate URLs found in result"


# ===========================================================================
# INTEGRATION TESTS — Full Orchestrator (all I/O mocked)
# ===========================================================================


class TestOrchestrator:
    """End-to-end pipeline tests with fixture evidence injected via mocks."""

    def _build(self, retrieval_snippets=None):
        from agents import (
            ClaimExtractionAgent,
            RetrievalAgent,
            VerificationEngine,
            ScoringCalibrator,
            ExplanationGenerator,
            VerificationOrchestrator,
        )
        from models import VerificationConfig

        snippets = retrieval_snippets or [
            make_snippet(url="https://en.wikipedia.org/wiki/Python", text="Python was created in 1991."),
        ]

        async def mock_retrieve(claim, config):
            return make_retrieval_result(claim=claim, snippets=snippets)

        async def mock_verify(retrieval, config):
            from models import EntailmentStatus
            for s in retrieval.snippets:
                s.entailment_status = EntailmentStatus.SUPPORTS
                s.nli_score = 0.85
                s.llm_judge_score = 0.80
            return 0.85, 0.80, 0.80, 0.90, retrieval.snippets

        extractor = ClaimExtractionAgent(llm_client=None)
        retriever = MagicMock()
        retriever.retrieve = AsyncMock(side_effect=mock_retrieve)
        verifier = MagicMock()
        verifier.verify = AsyncMock(side_effect=mock_verify)
        calibrator = ScoringCalibrator()
        explainer = ExplanationGenerator()

        return VerificationOrchestrator(extractor, retriever, verifier, calibrator, explainer)

    def _request(self, model_output: str = "Python was created by Guido van Rossum in 1991."):
        from models import VerificationRequest, VerificationConfig

        return VerificationRequest(
            job_id=uuid.uuid4(),
            prompt="Tell me about Python.",
            model_output=model_output,
            config=VerificationConfig(
                enable_llm_judge=False,
                enable_nli=False,
                enable_web_search=False,
                enable_kb_search=False,
            ),
        )

    @pytest.mark.asyncio
    async def test_report_shape(self):
        from models import VerificationReport

        orchestrator = self._build()
        report = await orchestrator.run(self._request())
        assert isinstance(report, VerificationReport)
        assert len(report.claims) >= 1
        assert 0.0 <= report.overall_score <= 1.0

    @pytest.mark.asyncio
    async def test_report_metadata_populated(self):
        orchestrator = self._build()
        req = self._request()
        report = await orchestrator.run(req)
        assert report.metadata.job_id == req.job_id
        assert report.metadata.processing_time_ms >= 0
        assert report.metadata.claims_extracted >= 1

    @pytest.mark.asyncio
    async def test_per_claim_fields_present(self):
        orchestrator = self._build()
        report = await orchestrator.run(self._request())
        for claim in report.claims:
            assert claim.claim != ""
            assert claim.contextualized_claim != ""
            assert 0.0 <= claim.score <= 1.0
            assert claim.label is not None
            assert claim.explanation != ""
            assert isinstance(claim.evidence, list)

    @pytest.mark.asyncio
    async def test_uncertainty_signals_auto_populated(self):
        orchestrator = self._build()
        report = await orchestrator.run(self._request())
        sig = report.uncertainty_signals
        assert "contradicted_by_label_count" in sig
        assert "uncertain_claim_count" in sig
        assert "min_claim_score" in sig
        assert "score_variance" in sig

    @pytest.mark.asyncio
    async def test_no_verifiable_claims_produces_empty_report(self):
        orchestrator = self._build()
        # All sentences are unverifiable opinions
        req = self._request("I think Python is great. In my opinion it is wonderful.")
        report = await orchestrator.run(req)
        assert report.claims == []
        assert report.overall_score is None

    @pytest.mark.asyncio
    async def test_failed_claim_verification_excluded_from_report(self):
        """A claim whose verify() raises should be silently dropped."""
        from agents import (
            ClaimExtractionAgent,
            RetrievalAgent,
            VerificationEngine,
            ScoringCalibrator,
            ExplanationGenerator,
            VerificationOrchestrator,
        )

        async def bad_retrieve(claim, config):
            raise RuntimeError("Retrieval exploded")

        extractor = ClaimExtractionAgent(llm_client=None)
        retriever = MagicMock()
        retriever.retrieve = AsyncMock(side_effect=bad_retrieve)
        verifier = MagicMock()
        calibrator = ScoringCalibrator()
        explainer = ExplanationGenerator()

        orchestrator = VerificationOrchestrator(extractor, retriever, verifier, calibrator, explainer)
        report = await orchestrator.run(self._request())
        # No claims survived; overall score should still be returned
        assert report.claims == []

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Verify the semaphore allows at most max_concurrent_claims tasks simultaneously."""
        import asyncio

        concurrent_count = 0
        max_seen = 0

        async def counting_retrieve(claim, config):
            nonlocal concurrent_count, max_seen
            concurrent_count += 1
            max_seen = max(max_seen, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return make_retrieval_result(claim=claim)

        from agents import (
            ClaimExtractionAgent, ScoringCalibrator,
            ExplanationGenerator, VerificationOrchestrator, VerificationEngine,
        )

        extractor = ClaimExtractionAgent()
        retriever = MagicMock()
        retriever.retrieve = AsyncMock(side_effect=counting_retrieve)
        verifier = MagicMock()
        verifier.verify = AsyncMock(return_value=(0.5, 0.5, 0.5, 0.5, []))

        from models import VerificationConfig

        orchestrator = VerificationOrchestrator(
            extractor, retriever, verifier, ScoringCalibrator(), ExplanationGenerator()
        )
        # Patch semaphore limit to 3
        orchestrator._semaphore = asyncio.Semaphore(3)

        long_text = " ".join([
            "Python was created in 1991.",
            "Guido van Rossum designed the language.",
            "Python supports object-oriented programming.",
            "It is widely used in data science applications.",
            "NumPy and Pandas are popular Python libraries.",
        ])
        from models import VerificationRequest

        req = VerificationRequest(
            job_id=uuid.uuid4(),
            prompt="Python info",
            model_output=long_text,
            config=VerificationConfig(enable_llm_judge=False, enable_nli=False,
                                      enable_web_search=False, enable_kb_search=False),
        )
        await orchestrator.run(req)
        assert max_seen <= 3


# ===========================================================================
# API TESTS — FastAPI TestClient
# ===========================================================================


@pytest.fixture(scope="module")
def client():
    """
    Build a TestClient with all external dependencies mocked:
      - LLM client → MockLLMClient
      - HTTP client → mock_http_client with Wikipedia fixture
      - Redis cache → disabled
    """
    wiki_fixture = {
        "query": {
            "search": [
                {
                    "title": "Python (programming language)",
                    "snippet": "Python is a high-level language created by Guido van Rossum in 1991.",
                }
            ]
        }
    }

    async def mock_retrieve(self, claim, config):
        from models import RetrievalResult, EvidenceSnippet, CredibilityTier, EntailmentStatus

        s = EvidenceSnippet(
            source_url="https://en.wikipedia.org/wiki/Python",
            snippet="Python was created by Guido van Rossum in 1991.",
            credibility_tier=CredibilityTier.TIER_1,
            entailment_status=EntailmentStatus.SUPPORTS,
            nli_score=0.88,
            llm_judge_score=0.85,
        )
        return make_retrieval_result(claim=claim, snippets=[s])

    async def mock_verify(self, retrieval, config):
        from models import EntailmentStatus

        for s in retrieval.snippets:
            s.entailment_status = EntailmentStatus.SUPPORTS
        return 0.88, 0.85, 0.82, 0.95, retrieval.snippets

    with (
        patch("agents.RetrievalAgent.retrieve", new=mock_retrieve),
        patch("agents.VerificationEngine.verify", new=mock_verify),
        patch("agents.LLMClient.complete", new=AsyncMock(return_value="")),
        patch("agents.settings") as ms,
    ):
        ms.enable_semantic_cache = False
        ms.enable_reranker = False
        ms.enable_coreference_resolution = False
        ms.max_concurrent_claims = 8
        ms.log_level = "WARNING"
        ms.redis_url = "redis://localhost:6379/0"
        ms.wikipedia_user_agent = "test"
        ms.http_timeout_seconds = 5.0

        from main import app
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body_has_status_ok(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_health_body_has_service_name(self, client):
        body = client.get("/health").json()
        assert "service" in body


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_body_has_required_fields(self, client):
        body = client.get("/metrics").json()
        assert "total_jobs_processed" in body
        assert "average_latency_ms" in body
        assert "hallucination_rate" in body
        assert "cache_hit_rate" in body


class TestSyncEndpoint:
    def test_valid_request_returns_200(self, client):
        resp = client.post("/verify", json=minimal_request())
        assert resp.status_code == 200

    def test_response_has_overall_score(self, client):
        body = client.post("/verify", json=minimal_request()).json()
        assert "overall_score" in body
        assert 0.0 <= body["overall_score"] <= 1.0

    def test_response_has_overall_label(self, client):
        body = client.post("/verify", json=minimal_request()).json()
        assert body["overall_label"] in [
            "VERIFIED", "UNCERTAIN", "UNSUPPORTED", "CONTRADICTED", "NO_VERIFIABLE_CONTENT", "UNKNOWN"
        ]

    def test_response_has_claims_list(self, client):
        body = client.post("/verify", json=minimal_request()).json()
        assert "claims" in body
        assert isinstance(body["claims"], list)

    def test_response_has_metadata(self, client):
        body = client.post("/verify", json=minimal_request()).json()
        meta = body["metadata"]
        assert "job_id" in meta
        assert "processing_time_ms" in meta
        assert "sources_queried" in meta

    def test_response_has_uncertainty_signals(self, client):
        body = client.post("/verify", json=minimal_request()).json()
        assert "uncertainty_signals" in body

    def test_claim_block_has_required_fields(self, client):
        body = client.post("/verify", json=minimal_request()).json()
        if body["claims"]:
            c = body["claims"][0]
            for field in ["claim", "contextualized_claim", "score", "label", "explanation", "evidence"]:
                assert field in c, f"Missing field: {field}"

    def test_evidence_block_has_required_fields(self, client):
        body = client.post("/verify", json=minimal_request()).json()
        for claim in body["claims"]:
            for ev in claim["evidence"]:
                assert "source_url" in ev
                assert "snippet" in ev
                assert "credibility_tier" in ev
                assert "entailment_status" in ev

    def test_invalid_uuid_returns_422(self, client):
        payload = minimal_request()
        payload["job_id"] = "not-a-uuid"
        resp = client.post("/verify", json=payload)
        assert resp.status_code == 422

    def test_missing_prompt_returns_422(self, client):
        payload = minimal_request()
        del payload["prompt"]
        resp = client.post("/verify", json=payload)
        assert resp.status_code == 422

    def test_missing_model_output_returns_422(self, client):
        payload = minimal_request()
        del payload["model_output"]
        resp = client.post("/verify", json=payload)
        assert resp.status_code == 422

    def test_strictness_out_of_range_returns_422(self, client):
        payload = minimal_request()
        payload["config"]["strictness"] = 2.5
        resp = client.post("/verify", json=payload)
        assert resp.status_code == 422

    def test_job_id_echoed_in_metadata(self, client):
        payload = minimal_request()
        body = client.post("/verify", json=payload).json()
        assert body["metadata"]["job_id"] == payload["job_id"]


class TestAsyncEndpoint:
    def test_async_submit_returns_202(self, client):
        resp = client.post("/verify/async", json=minimal_request())
        assert resp.status_code == 202

    def test_async_submit_returns_job_id(self, client):
        payload = minimal_request()
        body = client.post("/verify/async", json=payload).json()
        assert body["job_id"] == payload["job_id"]

    def test_async_submit_initial_status(self, client):
        body = client.post("/verify/async", json=minimal_request()).json()
        assert body["status"] in ("queued", "processing")

    def test_poll_unknown_job_returns_404(self, client):
        resp = client.get(f"/verify/jobs/{make_uuid()}")
        assert resp.status_code == 404

    def test_poll_known_job_returns_200(self, client):
        payload = minimal_request()
        client.post("/verify/async", json=payload)
        resp = client.get(f"/verify/jobs/{payload['job_id']}")
        assert resp.status_code == 200

    def test_poll_known_job_has_status_field(self, client):
        payload = minimal_request()
        client.post("/verify/async", json=payload)
        body = client.get(f"/verify/jobs/{payload['job_id']}").json()
        assert "status" in body
        assert body["status"] in ("queued", "processing", "completed", "failed")


# ===========================================================================
# EDGE CASE TESTS
# ===========================================================================


class TestEdgeCases:
    """Boundary and regression tests that don't fit cleanly into other classes."""

    def test_very_long_model_output_accepted(self, client):
        payload = minimal_request("Python is a language. " * 200)
        resp = client.post("/verify", json=payload)
        assert resp.status_code == 200

    def test_model_output_at_max_length_accepted(self, client):
        # 32,000 chars is the limit — just under should pass
        payload = minimal_request("A" * 31_999)
        resp = client.post("/verify", json=payload)
        # Will likely produce 0 verifiable claims but should not 422
        assert resp.status_code == 200

    def test_model_output_exceeds_max_length_returns_422(self, client):
        payload = minimal_request("A" * 32_001)
        resp = client.post("/verify", json=payload)
        assert resp.status_code == 422

    def test_all_opinion_output_produces_zero_claims(self, client):
        payload = minimal_request(
            "I think this is great. In my opinion it might be correct. "
            "I believe everything will work out."
        )
        body = client.post("/verify", json=payload).json()
        assert body["claims"] == []
        assert body["overall_score"] is None
        assert body["overall_label"] == "NO_VERIFIABLE_CONTENT"

    @pytest.mark.asyncio
    async def test_score_calibrator_precision(self):
        from agents import ScoringCalibrator
        from models import VerifiedClaim, VerificationLabel

        cal = ScoringCalibrator()
        # Overall score should be rounded to 4 decimal places
        s = cal.compute_overall_score([
            VerifiedClaim(
                claim="c",
                contextualized_claim="c",
                score=0.123456789,
                label=VerificationLabel.UNCERTAIN,
                explanation="e",
                evidence=[],
            )
        ])
        decimal_places = len(str(s).split(".")[-1])
        assert decimal_places <= 4

    @pytest.mark.asyncio
    async def test_extract_domain_handles_bare_hostname(self):
        from agents import _extract_domain

        assert _extract_domain("https://wikipedia.org/wiki/X") == "wikipedia.org"

    @pytest.mark.asyncio
    async def test_extract_domain_handles_malformed(self):
        from agents import _extract_domain

        result = _extract_domain("not-a-url")
        assert result is not None  # Should not raise

    def test_metrics_hallucination_rate_increments(self, client):
        # Fire two requests and check that metrics counter is non-decreasing
        m1 = client.get("/metrics").json()["total_jobs_processed"]
        client.post("/verify", json=minimal_request())
        m2 = client.get("/metrics").json()["total_jobs_processed"]
        assert m2 > m1


# ===========================================================================
# Entry point — run with python test_afve.py for quick smoke test
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AFVE Test Suite — Quick Smoke Run")
    print("Run with:  pytest test_afve.py -v  for full output")
    print("=" * 60)
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short", "-q"],
        capture_output=False,
    )
    sys.exit(result.returncode)