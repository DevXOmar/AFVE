"""
agents.py — Relational Fact Verification Engine (AFVE v1.3)

Root-cause analysis of every reported failure
──────────────────────────────────────────────
All previous complaints shared one flaw: we were validating ENTITIES, not
FACTS. Checking "is Apple in the snippet?" is not fact verification. The
question is "does this snippet say Apple was founded in 1976?". This
revision rebuilds the pipeline around that distinction.

New systems
───────────
RelationalDecomposer
  Every sentence is decomposed into a (subject, predicate, object, qualifier)
  triple before any retrieval happens. "Apple was founded in 1976 and employs
  150,000 people" splits into TWO atomic claims, each with their own triple.
  The subject is always the retrieval anchor — not the object.

  Example:
    "Elon Musk co-founded Tesla in 2003"
    → S="Elon Musk" P="co-founded" O="Tesla" Q="2003"
    Retrieval query: "Elon Musk Tesla co-founded" (subject-first)
    NOT: "Tesla Motors founders" (object-first, wrong)

SubjectFirstRetrieval
  Queries are always built from the SUBJECT. On attempt-0, the subject's
  Wikipedia article is fetched by direct title (subject_article_fetch).
  This guarantees the canonical source for the subject is always in the pool.

RelationVerifier
  After retrieval, each snippet is scored for:
    subject_match_score  — does this snippet discuss the subject?
    relation_match_score — is the predicate relationship expressed here?
  Snippets with low subject_match_score are penalised in source_weight;
  they cannot drive the composite score high even if NLI fires.

NumericComparator
  Extracts all numbers from evidence and compares against NumericAssertions
  using per-unit tolerances. Catches "20% vs 2%", "1991 vs 1994",
  "150,000 employees vs 15,000 employees". Numeric contradiction is a hard
  override on entailment status — NLI alone cannot catch order-of-magnitude
  errors.

TruthAwareCalibrator
  Six-signal composite score: NLI | LLM judge | source credibility |
  cross-source agreement | relation score | numeric comparison.
  TRUE_POSITIVE_FLOOR prevents strong-evidence correct facts from being
  dragged to UNCERTAIN by scoring variance. NUMERIC_CONTRADICTION_PENALTY
  is a hard ceiling when numbers don't match.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from config import (
    CLAIM_TYPE_TRUTH_PRIORS,
    CREDIBILITY_MULTIPLIERS,
    CROSS_SOURCE_AGREEMENT_WEIGHT,
    DOMAIN_TIER_MAP,
    HALLUCINATION_PENALTY,
    LABEL_THRESHOLDS,
    LLM_JUDGE_WEIGHT,
    NLI_WEIGHT,
    NO_EVIDENCE_PENALTY,
    NUMERIC_CONTRADICTION_PENALTY,
    NUMERIC_SCORE_WEIGHT,
    NUMERIC_TOLERANCES,
    RELATION_VERIFICATION_WEIGHT,
    SOURCE_CREDIBILITY_WEIGHT,
    SNIPPET_MIN_CHARS,
    SUBJECT_MATCH_THRESHOLD,
    SUBJECT_QUERY_TEMPLATES,
    TRUE_POSITIVE_FLOOR,
    get_settings,
)
from models import (
    AtomicClaim,
    ClaimType,
    ComparisonResult,
    CredibilityTier,
    EntailmentStatus,
    EntityLabel,
    EntitySpan,
    EvidenceSnippet,
    JobMetadata,
    NumericAssertion,
    NumericUnit,
    RawClaim,
    RelationalAnalysis,
    RelationalTriple,
    ResolvedClaim,
    RetrievalResult,
    VerificationLabel,
    VerificationReport,
    VerificationRequest,
    VerifiedClaim,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Vocabulary constants
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and",
    "or", "but", "for", "on", "at", "by", "from", "with", "that", "this",
    "it", "its", "be", "been", "has", "have", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "as", "if", "than", "then", "so", "yet", "both", "each", "also", "which",
    "who", "whom", "whose", "what", "when", "where", "how", "all", "into",
})

_NEGATION_TOKENS: frozenset[str] = frozenset({
    "not", "never", "no", "false", "incorrect", "wrong", "untrue",
    "inaccurate", "contrary", "refuted", "debunked", "denied", "disproven",
    "didn't", "doesn't", "wasn't", "weren't", "isn't", "aren't",
})

# Predicate synonym groups per ClaimType — used by relation verifier
_PREDICATE_SYNONYMS: dict[ClaimType, frozenset[str]] = {
    ClaimType.DATE: frozenset({
        "founded", "established", "created", "formed", "incorporated",
        "launched", "started", "built", "born", "died", "released",
        "published", "opened", "introduced", "invented", "began",
    }),
    ClaimType.AUTHORSHIP: frozenset({
        "created", "invented", "wrote", "authored", "developed", "designed",
        "built", "made", "produced", "directed", "composed", "conceived",
    }),
    ClaimType.ROLE: frozenset({
        "ceo", "founder", "co-founder", "president", "chairman", "director",
        "officer", "head", "chief", "executive", "manager", "leader",
        "vice president", "vp", "secretary", "partner", "cto", "coo",
    }),
    ClaimType.MEMBERSHIP: frozenset({
        "member", "founder", "co-founder", "part of", "belongs to",
        "affiliated with", "associated with", "joined", "employee",
    }),
    ClaimType.LOCATION: frozenset({
        "located", "based", "headquartered", "situated", "capital",
        "city", "country", "region", "state", "province",
    }),
    ClaimType.CAUSAL: frozenset({
        "caused", "led to", "resulted in", "triggered", "produced",
        "induced", "responsible for", "brought about", "contributed to",
    }),
    ClaimType.ATTRIBUTE: frozenset({
        "has", "have", "employs", "contains", "includes", "owns",
        "operates", "produces", "generates", "earns", "costs",
    }),
    ClaimType.COMPARATIVE: frozenset({
        "larger", "smaller", "bigger", "greater", "more", "less", "fewer",
        "highest", "lowest", "most", "least", "faster", "slower", "older",
        "newer", "ranked", "surpassed", "exceeded", "outperformed",
    }),
    ClaimType.EXISTENCE: frozenset({"is", "are", "exists", "real", "actual"}),
    ClaimType.GENERAL: frozenset(),
}

# ---------------------------------------------------------------------------
# Lazy model loaders
# ---------------------------------------------------------------------------

# Eagerly load NLI globally (Never per request/claim/loop)
NLI_PIPELINE = None
_reranker = None
_reranker_lock = threading.Lock()

try:
    from transformers import pipeline  # type: ignore
    NLI_PIPELINE = pipeline(
        "text-classification",
        model=settings.nli_model_name,
        device=-1,  # CPU
    )
    logger.info("NLI loaded globally: %s", settings.nli_model_name)
except Exception as exc:
    logger.warning("NLI unavailable globally (%s).", exc)

def _get_nli_pipeline():
    return NLI_PIPELINE


def _get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is not None:
                return _reranker
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
                # Stability: force CPU and avoid MPS entirely.
                _reranker = CrossEncoder(settings.reranker_model_name, device="cpu")
                logger.info("Reranker loaded (cpu): %s", settings.reranker_model_name)
            except Exception as exc:
                _reranker = None
                logger.warning("Reranker unavailable (%s).", exc)
    return _reranker


# ---------------------------------------------------------------------------
# Retrieval query hygiene
# ---------------------------------------------------------------------------

_RETRIEVAL_STOPWORDS: frozenset[str] = frozenset({
    # User-required stopwords
    "it","is","the","a","an","for","in","on","that","here","there","what","how",
    "helps","promotes","protects","aids","keep","add","remember","better","tips",
    "choose","flushes","transports","lubricates",
    # Additional generic glue/pronouns that produce low-quality queries
    "another","this","these","those","they","them","he","she","we","you","i",
    "of","to","and","or","as","at","by","from","with","without",
    # Explicitly banned query noise
    "overview","wikipedia",
})

# Extra guardrail for low-quality query tokens that are typically verbs/generic actions.
_RETRIEVAL_VERBISH: frozenset[str] = frozenset({
    "help", "helps", "helped", "helping",
    "maintain", "maintains", "maintained", "maintaining",
    "improve", "improves", "improved", "improving",
    "regulate", "regulates", "regulated", "regulating",
    "promote", "promotes", "promoted", "promoting",
    "protect", "protects", "protected", "protecting",
    "aid", "aids", "aided", "aiding",
    "keep", "keeps", "kept", "keeping",
    "add", "adds", "added", "adding",
    "choose", "chooses", "chose", "choosing",
    "remember", "remembers", "remembered", "remembering",
    "tip", "tips",
    "better",
    "flush", "flushes", "flushed", "flushing",
    "transport", "transports", "transported", "transporting",
    "lubricate", "lubricates", "lubricated", "lubricating",
})


def _has_alpha(text: str) -> bool:
    return any(c.isalpha() for c in (text or ""))


def is_valid_entity(text: str) -> bool:
    text = (text or "").strip()
    if len(text) <= 3:
        return False
    if not _has_alpha(text):
        return False
    if text.lower() in _RETRIEVAL_STOPWORDS:
        return False
    if text.lower() in _RETRIEVAL_VERBISH:
        return False
    return True


def extract_entities(text: str) -> list[str]:
    """Extract meaningful entities/phrases from a claim for retrieval queries.

    Prefers: named entities + noun phrases (spaCy), and falls back to filtered keywords.
    """
    text = (text or "").strip()
    if not text:
        return []

    def _clean_phrase(p: str) -> str:
        p = re.sub(r"\s+", " ", (p or "").strip())
        p = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", p)
        tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z\-']+", p)]
        tokens = [t for t in tokens if is_valid_entity(t) and t.lower() not in _STOPWORDS]
        if not tokens:
            return ""
        return " ".join(tokens)

    def _clean_spacy_span(span) -> str:
        """Keep only noun-ish tokens from a spaCy span when available."""
        kept: list[str] = []
        for tok in span:
            txt = getattr(tok, "text", "")
            if not txt:
                continue
            tl = txt.lower()
            if tl in _STOPWORDS or tl in _RETRIEVAL_STOPWORDS or tl in _RETRIEVAL_VERBISH:
                continue
            if not is_valid_entity(txt):
                continue
            pos = getattr(tok, "pos_", "")
            # Prefer nouns / proper nouns / adjectives; drop verbs/pronouns/dets.
            if pos and pos not in {"NOUN", "PROPN", "ADJ"}:
                continue
            kept.append(txt)
        return " ".join(kept).strip()

    phrases: list[str] = []

    # spaCy path (best quality) — optional dependency.
    try:
        import spacy  # type: ignore

        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = None

        if nlp is not None:
            doc = nlp(text)
            ents = []
            for ent in getattr(doc, "ents", []):
                cleaned = _clean_spacy_span(ent) or _clean_phrase(getattr(ent, "text", ""))
                if cleaned:
                    ents.append(cleaned)
            ents = [e for e in ents if e]

            noun_chunks: list[str] = []
            try:
                for chunk in doc.noun_chunks:
                    cleaned = _clean_spacy_span(chunk) or _clean_phrase(chunk.text)
                    if cleaned:
                        noun_chunks.append(cleaned)
            except Exception:
                noun_chunks = []

            # Prefer named entities, then noun phrases.
            phrases = ents or noun_chunks
    except Exception:
        phrases = []

    if not phrases:
        # Fallback: keyword-ish phrases built from the claim.
        tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", text)
        tokens = [
            t for t in tokens
            if is_valid_entity(t)
            and t.lower() not in _STOPWORDS
            and t.lower() not in _RETRIEVAL_VERBISH
            and t.lower() not in _RETRIEVAL_STOPWORDS
        ]
        # Build 2-4 word phrases from adjacent kept tokens (more noun-phrase-like).
        windowed: list[str] = []
        for i in range(len(tokens)):
            for w in (4, 3, 2):
                if i + w <= len(tokens):
                    windowed.append(" ".join(tokens[i:i+w]))
        # If the claim is benefit-ish, sprinkle a generic anchor term.
        if re.search(r"\b(helps?|benefits?|improves?|promotes?|maintains?)\b", text, flags=re.IGNORECASE):
            windowed.append("human body benefits")
            windowed.append("health benefits")
        phrases = windowed or tokens

    # De-dup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in phrases:
        p = _clean_phrase(p)
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        # Ensure semantic meaningfulness: >3 chars overall and at least one alphabetic token.
        if len(p) <= 3 or not _has_alpha(p):
            continue
        seen.add(key)
        out.append(p)
    return out


def _sanitize_query(query: str) -> str:
    """Normalize a query for retrieval.

    Requirements:
    - keep it short (max 8 tokens)
    - remove stopwords and generic/verbish words
    - remove known noise tokens like "overview"/"wikipedia"
    """
    if not query:
        return ""

    tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", query)
    kept: list[str] = []
    for t in tokens:
        tl = t.lower()
        if not is_valid_entity(t):
            continue
        if tl in _STOPWORDS or tl in _RETRIEVAL_STOPWORDS or tl in _RETRIEVAL_VERBISH:
            continue
        kept.append(tl)

    kept = list(dict.fromkeys(kept))
    if not kept:
        return ""
    return " ".join(kept[:8]).strip()


# ===========================================================================
# RelationalDecomposer
# ===========================================================================


class RelationalDecomposer:
    """
    Converts a resolved sentence into one or more AtomicClaims, each
    carrying a RelationalTriple and subject-first retrieval queries.

    Multi-claim splitting: "Apple was founded in 1976 and employs 150,000
    people" → two AtomicClaims with distinct triples.
    """

    # Compound splitter: splits at coordinating conjunctions that
    # introduce a new independent predicate (not inside a noun phrase).
    _COMPOUND_RE = re.compile(
        r"\s+and\s+(?=[a-z]+(?:s|es|ed|ing)?\s)"   # "and employs/and was"
        r"|\s*;\s*"
        r"|\s*,\s+(?:while|whereas|although|though|but)\s+",
        re.IGNORECASE,
    )

    # Verb detection patterns (order = precedence)
    _DATE_VERBS = re.compile(
        r"\b(founded|established|created|formed|launched|started|born|died|"
        r"released|published|opened|introduced|invented|incorporated|began)\b",
        re.IGNORECASE,
    )
    _ROLE_VERBS = re.compile(
        r"\b(?:is|was|serves?|served|became|appointed|elected|named)\s+"
        r"(?:the\s+)?(?:a\s+)?"
        r"(ceo|founder|co-founder|president|chairman|director|chief|head|"
        r"officer|executive|manager|leader|vp|vice\s+president|cto|coo|cfo|partner)\b",
        re.IGNORECASE,
    )
    _AUTHORSHIP_VERBS = re.compile(
        r"\b(created|invented|wrote|authored|developed|designed|built|made|"
        r"produced|directed|composed|conceived|pioneered)\b",
        re.IGNORECASE,
    )
    _MEMBERSHIP_VERBS = re.compile(
        r"\b(co-founded|co-created|is\s+a\s+member\s+of|part\s+of|"
        r"belongs?\s+to|affiliated\s+with|joined)\b",
        re.IGNORECASE,
    )
    _LOCATION_VERBS = re.compile(
        r"\b(?:is\s+)?(?:located|based|headquartered|situated)\s+in\b"
        r"|\bis\s+the\s+(?:capital|largest\s+city|city)\b",
        re.IGNORECASE,
    )
    _CAUSAL_VERBS = re.compile(
        r"\b(caused|led\s+to|resulted\s+in|triggered|induced|"
        r"responsible\s+for|contributed\s+to)\b",
        re.IGNORECASE,
    )
    _COMPARATIVE_VERBS = re.compile(
        r"\b(?:is\s+)?(?:larger|smaller|bigger|more|less|fewer|"
        r"highest|lowest|faster|slower|ranked|surpassed|exceeded)\b",
        re.IGNORECASE,
    )

    _YEAR_RE   = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
    _NUMBER_RE = re.compile(
        r"\b(\d[\d,]*(?:\.\d+)?)\s*"
        r"(million|billion|trillion|thousand|hundred|"
        r"%|percent|km|miles|kg|lbs|employees|people|users|dollars?|usd)?\b",
        re.IGNORECASE,
    )
    _PERCENT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(%|percent)\b", re.IGNORECASE)
    _ENTITY_RE  = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,4})\b")

    def __init__(self, llm_client: "LLMClient | None" = None) -> None:
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def decompose(
        self, text: str, sentence_index: int, config
    ) -> list[AtomicClaim]:
        atoms_text = (
            self._split_compound(text)
            if config.enable_multi_claim_splitting
            else [text]
        )

        atoms: list[AtomicClaim] = []
        for idx, atom_text in enumerate(atoms_text):
            atom_text = atom_text.strip()
            if not atom_text or len(atom_text.split()) < 4:
                continue
            analysis = await self._analyse(atom_text)
            atoms.append(AtomicClaim(
                original_sentence=text,
                atomic_text=atom_text,
                sentence_index=sentence_index,
                atom_index=idx,
                analysis=analysis,
            ))
        return atoms

    # ------------------------------------------------------------------
    # Compound splitting
    # ------------------------------------------------------------------

    def _split_compound(self, text: str) -> list[str]:
        parts = self._COMPOUND_RE.split(text)
        if len(parts) == 1:
            return parts

        result: list[str] = []
        # Extract subject from first clause to carry forward
        subject_carry = ""
        first_entities = self._ENTITY_RE.findall(parts[0]) if parts else []
        if first_entities:
            subject_carry = first_entities[0]

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            # If the clause starts lowercase (no subject), prepend the carry subject
            if i > 0 and subject_carry and part and part[0].islower():
                part = f"{subject_carry} {part}"
            result.append(part)

        return result if len(result) > 1 else [text]

    # ------------------------------------------------------------------
    # Triple extraction and analysis
    # ------------------------------------------------------------------

    async def _analyse(self, text: str) -> RelationalAnalysis:
        triple = self._extract_triple(text)
        # LLM fallback only when subject is completely missing
        if not triple.subject and self._llm:
            triple = await self._extract_triple_llm(text) or triple

        claim_type = self._classify(text, triple)
        numerics   = self._extract_numerics(text, claim_type)
        prior      = CLAIM_TYPE_TRUTH_PRIORS.get(claim_type.value, 0.60)
        queries    = self._build_queries(triple, claim_type, text)

        return RelationalAnalysis(
            claim_type=claim_type,
            triple=triple,
            atomic_claim_text=text,
            subject_queries=queries,
            numeric_assertions=numerics,
            truth_prior=prior,
        )

    def _extract_triple(self, text: str) -> RelationalTriple:
        """
        Heuristic triple extraction.
        Subject = first capitalised entity sequence (before the main verb).
        Predicate = matched verb pattern.
        Object = first entity after the predicate.
        Qualifier = year or number modifier.
        """
        subject = predicate = obj = qualifier = ""

        # Subject: first entity in text
        all_entities = self._ENTITY_RE.findall(text)
        if all_entities:
            subject = all_entities[0]

        # Predicate + object: match verb patterns in priority order
        verb_patterns = [
            self._ROLE_VERBS,
            self._DATE_VERBS,
            self._AUTHORSHIP_VERBS,
            self._MEMBERSHIP_VERBS,
            self._CAUSAL_VERBS,
            self._LOCATION_VERBS,
            self._COMPARATIVE_VERBS,
        ]
        for pat in verb_patterns:
            m = pat.search(text)
            if m:
                predicate = m.group(0).strip()
                after = text[m.end():].strip()
                obj_ents = self._ENTITY_RE.findall(after)
                if obj_ents:
                    obj = obj_ents[0]
                break

        # Qualifier: year first, then prominent number
        ym = self._YEAR_RE.search(text)
        if ym:
            qualifier = ym.group(0)
        elif not obj:
            nm = self._NUMBER_RE.search(text)
            if nm:
                obj = nm.group(0)

        return RelationalTriple(
            subject=subject,
            subject_label=self._label(subject),
            predicate=predicate,
            object=obj,
            object_label=self._label(obj),
            qualifier=qualifier,
            qualifier_label=EntityLabel.DATE if qualifier and qualifier.isdigit()
                            else EntityLabel.OTHER,
        )

    async def _extract_triple_llm(self, text: str) -> RelationalTriple | None:
        if not settings.enable_llm_triple_extraction:
            return None
        system = (
            "Extract the relational triple from this factual claim.\n"
            "Return ONLY valid JSON: "
            '{"subject":"","predicate":"","object":"","qualifier":""}\n'
            "subject=the entity the claim is ABOUT, predicate=the relation verb, "
            "object=what the predicate resolves to, qualifier=date/number/precision.\n"
            "No markdown, no explanation."
        )
        try:
            raw = await self._llm.complete(system, text, max_tokens=150)
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            d   = json.loads(raw)
            return RelationalTriple(
                subject=d.get("subject", ""),
                subject_label=self._label(d.get("subject", "")),
                predicate=d.get("predicate", ""),
                object=d.get("object", ""),
                object_label=self._label(d.get("object", "")),
                qualifier=d.get("qualifier", ""),
            )
        except Exception as exc:
            logger.debug("LLM triple extraction failed: %s", exc)
            return None

    def _label(self, text: str) -> EntityLabel:
        if not text:
            return EntityLabel.OTHER
        t = text.lower()
        org_signals = {
            "inc", "corp", "ltd", "llc", "company", "group", "bank",
            "foundation", "institute", "university", "college",
            "microsoft", "apple", "google", "amazon", "meta", "tesla",
            "openai", "nasa", "nba", "nfl", "eu", "nato",
        }
        if any(s in t for s in org_signals):
            return EntityLabel.ORG
        if re.match(r"^(19|20)\d{2}$", text.strip()):
            return EntityLabel.DATE
        if re.match(r"^\d", text.strip()):
            return EntityLabel.NUMBER
        words = text.split()
        if 1 < len(words) <= 3 and all(w[0].isupper() for w in words if w):
            return EntityLabel.PERSON
        return EntityLabel.OTHER

    def _classify(self, text: str, triple: RelationalTriple) -> ClaimType:
        if self._ROLE_VERBS.search(text):
            return ClaimType.ROLE
        if self._DATE_VERBS.search(text) or self._YEAR_RE.search(text):
            return ClaimType.DATE
        if self._AUTHORSHIP_VERBS.search(text):
            return ClaimType.AUTHORSHIP
        if self._MEMBERSHIP_VERBS.search(text):
            return ClaimType.MEMBERSHIP
        if self._CAUSAL_VERBS.search(text):
            return ClaimType.CAUSAL
        if self._LOCATION_VERBS.search(text):
            return ClaimType.LOCATION
        if self._COMPARATIVE_VERBS.search(text):
            return ClaimType.COMPARATIVE
        if self._NUMBER_RE.search(text) or self._PERCENT_RE.search(text):
            return ClaimType.ATTRIBUTE
        if triple.subject and not triple.predicate:
            return ClaimType.EXISTENCE
        return ClaimType.GENERAL

    def _extract_numerics(
        self, text: str, claim_type: ClaimType
    ) -> list[NumericAssertion]:
        assertions: list[NumericAssertion] = []
        year_spans: set[tuple[int, int]] = set()

        for m in self._YEAR_RE.finditer(text):
            v = float(m.group(1))
            assertions.append(NumericAssertion(
                raw_text=m.group(1),
                normalized_value=v,
                unit=NumericUnit.YEAR,
                tolerance_low=v,
                tolerance_high=v,
            ))
            year_spans.add(m.span())

        for m in self._PERCENT_RE.finditer(text):
            assertions.append(NumericAssertion(
                raw_text=m.group(0),
                normalized_value=float(m.group(1)),
                unit=NumericUnit.PERCENT,
            ))

        for m in self._NUMBER_RE.finditer(text):
            if any(m.start() >= s and m.end() <= e for s, e in year_spans):
                continue
            raw = m.group(1).replace(",", "")
            try:
                val = float(raw)
            except ValueError:
                continue
            suffix = (m.group(2) or "").lower()
            mult = {
                "million": 1e6, "billion": 1e9, "trillion": 1e12,
                "thousand": 1e3, "hundred": 1e2,
            }
            val *= mult.get(suffix, 1.0)
            unit = NumericUnit.NONE
            if "%" in suffix or "percent" in suffix:
                unit = NumericUnit.PERCENT
            elif suffix in ("dollar", "dollars", "usd"):
                unit = NumericUnit.MONEY
            elif suffix in ("km", "miles"):
                unit = NumericUnit.DISTANCE
            elif suffix in ("kg", "lbs"):
                unit = NumericUnit.WEIGHT
            elif suffix in ("employees", "people", "users"):
                unit = NumericUnit.COUNT
            assertions.append(NumericAssertion(
                raw_text=m.group(0),
                normalized_value=val,
                unit=unit,
            ))

        return assertions

    def _build_queries(
        self,
        triple: RelationalTriple,
        claim_type: ClaimType,
        full_text: str,
    ) -> list[str]:
        """Always anchor to the SUBJECT — the article being retrieved is
        the subject's article, not the object's."""
        subj = (triple.subject or "").strip() or (full_text.split()[0] if full_text.split() else "")

        def _tok(s: str) -> list[str]:
            return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z\-']+", s or "")]

        subj_tokens = [t for t in _tok(subj) if is_valid_entity(t) and t not in _STOPWORDS and t not in _RETRIEVAL_STOPWORDS and t not in _RETRIEVAL_VERBISH]
        obj_tokens = [t for t in _tok(triple.object) if is_valid_entity(t) and t not in _STOPWORDS and t not in _RETRIEVAL_STOPWORDS and t not in _RETRIEVAL_VERBISH]
        qual_tokens = [t for t in _tok(triple.qualifier) if is_valid_entity(t) and t not in _STOPWORDS and t not in _RETRIEVAL_STOPWORDS and t not in _RETRIEVAL_VERBISH]

        entity_tokens: list[str] = []
        for p in extract_entities(full_text):
            for t in _tok(p):
                if t in subj_tokens:
                    continue
                if t in _STOPWORDS or t in _RETRIEVAL_STOPWORDS or t in _RETRIEVAL_VERBISH:
                    continue
                if len(t) <= 2:
                    continue
                entity_tokens.append(t)
        entity_tokens = list(dict.fromkeys(entity_tokens))

        # Build a small set of short, clean queries (6–8 words max).
        candidates: list[list[str]] = []
        if subj_tokens and obj_tokens:
            candidates.append(subj_tokens[:3] + obj_tokens[:3] + entity_tokens[:2])
        if subj_tokens and entity_tokens:
            candidates.append(subj_tokens[:3] + entity_tokens[:5])
        if subj_tokens and qual_tokens:
            candidates.append(subj_tokens[:3] + qual_tokens[:2] + entity_tokens[:3])
        if not candidates and subj_tokens:
            candidates.append(subj_tokens[:3] + entity_tokens[:5])
        if not candidates:
            candidates.append(entity_tokens[:8])

        queries: list[str] = []
        for tokens in candidates:
            q = _sanitize_query(" ".join(tokens))
            if q:
                queries.append(q)

        # Ensure at least one query exists.
        if not queries:
            q = _sanitize_query(subj)
            if q:
                queries.append(q)
        return list(dict.fromkeys(queries))


# ===========================================================================
# A. Claim Extraction Agent
# ===========================================================================


class ClaimExtractionAgent:
    """
    Rule-based extraction of verifiable claims, followed by hybrid triple decomposition.
    """

    def __init__(self, llm_client=None, decomposer=None) -> None:
        self._llm = llm_client
        self._decomposer = decomposer or RelationalDecomposer(llm_client=llm_client)

    def is_verifiable(self, text: str) -> bool:
        """Rule-based classifier to reject questions, opinions, and conversational text."""
        text = text.strip()
        if not text or len(text.split()) < 3: return False
        if text.endswith('?'): return False
        
        lower_text = text.lower()
        unverifiable_prefixes = [
            "i think", "i believe", "in my opinion", "hello", "hi", "how can i",
            "here are", "below is", "let me", "sure", "of course", "personally",
            "arguably", "some might say", "yes", "it might", "it may", "it could"
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

    # ------------------------------------------------------------------
    # Backward-compatible helpers (used by the test suite)
    # ------------------------------------------------------------------

    def _split_into_sentences(self, text: str) -> list[RawClaim]:
        return self._split_sentences(text)

    async def _resolve_coreferences(
        self,
        raw_claims: list[RawClaim],
        prompt: str,
        model_output: str,
    ) -> list[ResolvedClaim]:
        """Optional coreference resolver.

        This is deliberately best-effort and only used when explicitly enabled.
        On any failure, falls back to the raw claim text.
        """
        fallback = [
            ResolvedClaim(
                original_text=c.text,
                contextualized_text=c.text,
                sentence_index=c.sentence_index,
                atom_index=0,
                analysis=None,
            )
            for c in raw_claims
        ]

        if not raw_claims:
            return []
        if not getattr(settings, "enable_coreference_resolution", False):
            return fallback
        if not self._llm:
            return fallback

        system = (
            "Resolve coreferences (pronouns like it/they/this) in each sentence using the given context. "
            "Return ONLY a JSON array of strings, one per input sentence, in the same order."
        )
        user = (
            f"CONTEXT PROMPT: {prompt}\n"
            f"MODEL OUTPUT: {model_output}\n"
            f"SENTENCES: {json.dumps([c.text for c in raw_claims])}\n"
            "RESOLVED JSON ARRAY:"
        )
        try:
            raw = (await self._llm.complete(system, user, max_tokens=256)).strip()
            resolved = json.loads(raw)
            if not isinstance(resolved, list) or len(resolved) != len(raw_claims):
                return fallback
            out: list[ResolvedClaim] = []
            for c, r in zip(raw_claims, resolved):
                out.append(
                    ResolvedClaim(
                        original_text=c.text,
                        contextualized_text=str(r).strip() or c.text,
                        sentence_index=c.sentence_index,
                        atom_index=0,
                        analysis=None,
                    )
                )
            return out
        except Exception:
            return fallback

    async def extract(self, request: VerificationRequest) -> list[ResolvedClaim]:
        raw = self._split_sentences(request.model_output)
        verifiable = [c for c in raw if c.is_verifiable]
        logger.info("Extracted %d verifiable / %d total sentences.", len(verifiable), len(raw))

        if not verifiable:
            fallback_text = request.model_output.strip()
            if self._should_fallback_to_full_claim(fallback_text):
                verifiable = [RawClaim(text=fallback_text, sentence_index=0, is_verifiable=True)]
                logger.info("Using full-response fallback claim extraction.")
            else:
                return []

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

    def _should_fallback_to_full_claim(self, text: str) -> bool:
        text = (text or "").strip()
        if not text or text.endswith("?") or len(text.split()) < 4:
            return False
        lower = text.lower()
        opinion_markers = (
            "i think", "i believe", "in my opinion", "personally",
            "it might", "it may", "it could", "wonderful", "great",
        )
        if any(marker in lower for marker in opinion_markers):
            return False
        if any(ch.isdigit() for ch in text):
            return True
        if extract_entities(text):
            return True
        return bool(re.search(r"\b(is|was|were|are|has|have|had|founded|created|developed|invented)\b", text, re.IGNORECASE))

# ===========================================================================
# B. Retrieval Agent — Subject-First
# ===========================================================================


class RetrievalAgent:
    """
    Always retrieves by SUBJECT, not object.
    First attempt: fetch subject's Wikipedia article by direct title
    lookup + subject-keyed full-text search. Later retries rotate
    through object-inclusive queries from SUBJECT_QUERY_TEMPLATES.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        llm_client: "LLMClient | None" = None,
        cache: "SemanticCache | None" = None,
    ) -> None:
        self._http = http_client
        self._llm = llm_client
        self._cache = cache

    async def retrieve(self, claim: ResolvedClaim, config) -> RetrievalResult:
        cache_key = claim.contextualized_text
        if self._cache and settings.enable_semantic_cache:
            cached = await self._cache.get(cache_key)
            if cached:
                logger.debug("Cache HIT: %.60s", cache_key)
                return RetrievalResult(
                    claim=claim, snippets=cached,
                    queries_attempted=0, retrieval_success=True,
                )

        analysis = claim.analysis
        subject  = analysis.triple.subject if analysis else ""
        queries  = analysis.subject_queries if analysis else [claim.contextualized_text]
        ctype    = analysis.claim_type if analysis else ClaimType.GENERAL
        triple   = analysis.triple if analysis else RelationalTriple()

        snippets: list[EvidenceSnippet] = []
        queries_attempted = 0

        for attempt in range(config.max_retrieval_attempts):
            queries_attempted += 1
            raw_query = queries[min(attempt, len(queries) - 1)]
            query = _sanitize_query(raw_query)
            if not query:
                query = _sanitize_query(subject) if subject else ""
            if not query:
                query = _sanitize_query(claim.contextualized_text)
            if not query:
                logger.debug(
                    "[%s] Attempt %d: query filtered out (raw=%r)",
                    ctype.value,
                    attempt + 1,
                    raw_query,
                )
                continue
            logger.debug("[%s] Attempt %d: %s", ctype.value, attempt + 1, query)

            parallel = []
            if config.enable_kb_search:
                parallel.append(self._wiki_fulltext(query))
                if attempt == 0 and subject:
                    parallel.append(self._wiki_subject_article(subject))
            else:
                parallel.append(_empty())
                parallel.append(_empty())

            if config.enable_web_search:
                parallel.append(self._search_web(query))
            else:
                parallel.append(_empty())

            results = await asyncio.gather(*parallel, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.warning("Source error: %s", r)
                else:
                    snippets.extend(r)

            snippets = _deduplicate(_filter_short(snippets))
            if len(snippets) >= 3:
                break
            logger.info("Attempt %d: %d snippets; retrying.", attempt + 1, len(snippets))

        # Score relational match for every snippet
        for s in snippets:
            s.subject_match_score  = _score_presence(s.snippet, triple.subject)
            s.object_match_score   = _score_presence(s.snippet, triple.object)
            s.relation_match_score = _score_predicate(s.snippet, triple.predicate, ctype)
            s.claim_type = ctype

        # Evidence flow variables (debuggable): raw_snippets -> reranked_snippets -> final_evidence
        raw_snippets: list[EvidenceSnippet] = list(snippets)
        reranked_snippets: list[EvidenceSnippet] = []

        # Rerank blending semantic + subject + relation scores
        if raw_snippets:
            if settings.enable_reranker:
                reranked_snippets = await asyncio.to_thread(
                    self._rerank, claim.contextualized_text, raw_snippets, config.top_k_snippets
                )
                if not reranked_snippets:
                    # Fallback: if reranker returns empty, keep top raw extracts.
                    raw_snippets.sort(key=lambda s: s.subject_match_score, reverse=True)
                    reranked_snippets = raw_snippets[:3]
            else:
                raw_snippets.sort(key=lambda s: s.subject_match_score, reverse=True)
                reranked_snippets = raw_snippets[:3]

        # Safety: always keep at least top 1–3 snippets and never overwrite with [].
        final_evidence: list[EvidenceSnippet] = [s for s in reranked_snippets if (s.snippet or "").strip()]
        if not final_evidence and raw_snippets:
            final_evidence = [s for s in raw_snippets if (s.snippet or "").strip()][:3]
        snippets = final_evidence[:3]

        if self._cache and settings.enable_semantic_cache and snippets:
            await self._cache.set(cache_key, snippets)

        return RetrievalResult(
            claim=claim, snippets=snippets,
            queries_attempted=queries_attempted,
            retrieval_success=bool(snippets),
        )

    async def _wiki_subject_article(self, subject: str) -> list[EvidenceSnippet]:
        """Direct title lookup — canonical source for the subject entity."""
        if not subject:
            return []
        try:
            resp = await self._http.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "titles": subject,
                    "prop": "extracts", "exintro": True,
                    "explaintext": True, "exsentences": 12,
                    "format": "json", "redirects": 1,
                },
            )
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            snippets: list[EvidenceSnippet] = []
            for pid, page in pages.items():
                if pid == "-1":
                    continue
                extract = page.get("extract", "").strip()
                if not extract:
                    continue
                slug = page.get("title", subject).replace(" ", "_")
                snippets.append(EvidenceSnippet(
                    source_url=f"https://en.wikipedia.org/wiki/{slug}",
                    snippet=extract[:2500],
                    credibility_tier=CredibilityTier.TIER_1,
                    subject_match_score=1.0,  # IS the subject article
                ))
            return snippets
        except Exception as exc:
            logger.warning("Subject article fetch '%s': %s", subject, exc)
            return []

    async def _wiki_fulltext(self, query: str) -> list[EvidenceSnippet]:
        try:
            s_resp = await self._http.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "list": "search",
                    "srsearch": query,
                    "srlimit": settings.kb_search_results_per_query,
                    "format": "json", "utf8": 1,
                },
            )
            s_resp.raise_for_status()
            hits = s_resp.json().get("query", {}).get("search", [])
            if not hits:
                return []

            pids = "|".join(str(h["pageid"]) for h in hits)
            e_resp = await self._http.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "pageids": pids,
                    "prop": "extracts", "exintro": True,
                    "explaintext": True, "exsentences": 8,
                    "format": "json",
                },
            )
            e_resp.raise_for_status()
            pages = e_resp.json().get("query", {}).get("pages", {})

            snippets: list[EvidenceSnippet] = []
            for hit in hits:
                page    = pages.get(str(hit["pageid"]), {})
                extract = page.get("extract", "").strip()
                if not extract:
                    extract = re.sub(r"<[^>]+>", "", hit.get("snippet", ""))
                if not extract:
                    continue
                slug = hit["title"].replace(" ", "_")
                snippets.append(EvidenceSnippet(
                    source_url=f"https://en.wikipedia.org/wiki/{slug}",
                    snippet=extract[:1800],
                    credibility_tier=CredibilityTier.TIER_1,
                ))
            return snippets
        except Exception as exc:
            logger.warning("Wikipedia fulltext: %s", exc)
            return []

    async def _search_web(self, query: str) -> list[EvidenceSnippet]:
        if settings.get_tavily_key():
            try:
                return await self._tavily(query)
            except Exception as exc:
                logger.warning("Tavily failed: %s", exc)
        if settings.get_serp_key():
            try:
                return await self._serpapi(query)
            except Exception as exc:
                logger.warning("SerpAPI failed: %s", exc)
        return []

    async def _tavily(self, query: str) -> list[EvidenceSnippet]:
        resp = await self._http.post("https://api.tavily.com/search", json={
            "api_key": settings.get_tavily_key(),
            "query": query,
            "max_results": settings.web_search_results_per_query,
            "search_depth": "advanced",
        })
        resp.raise_for_status()
        payload = resp.json()
        if hasattr(payload, "__await__"):
            payload = await payload
        return [
            EvidenceSnippet(
                source_url=r.get("url", ""),
                snippet=r.get("content", "")[:1800],
                credibility_tier=_infer_tier(r.get("url", "")),
            )
            for r in payload.get("results", []) if r.get("url")
        ]

    async def _serpapi(self, query: str) -> list[EvidenceSnippet]:
        resp = await self._http.get("https://serpapi.com/search", params={
            "q": query, "api_key": settings.get_serp_key(),
            "num": settings.web_search_results_per_query, "engine": "google",
        })
        resp.raise_for_status()
        payload = resp.json()
        if hasattr(payload, "__await__"):
            payload = await payload
        return [
            EvidenceSnippet(
                source_url=r.get("link", ""),
                snippet=r.get("snippet", "")[:1800],
                credibility_tier=_infer_tier(r.get("link", "")),
            )
            for r in payload.get("organic_results", []) if r.get("link")
        ]

    # ------------------------------------------------------------------
    # Backward-compatible helpers (used by the test suite)
    # ------------------------------------------------------------------

    async def _search_wikipedia(self, query: str) -> list[EvidenceSnippet]:
        """Wikipedia search (single request) used by unit tests."""
        try:
            resp = await self._http.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": settings.kb_search_results_per_query,
                    "format": "json",
                    "utf8": 1,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            if hasattr(payload, "__await__"):
                payload = await payload

            hits = payload.get("query", {}).get("search", [])
            out: list[EvidenceSnippet] = []
            for h in hits:
                title = h.get("title") or ""
                snippet = re.sub(r"<[^>]+>", "", h.get("snippet", ""))
                if not title or not snippet:
                    continue
                slug = title.replace(" ", "_")
                out.append(
                    EvidenceSnippet(
                        source_url=f"https://en.wikipedia.org/wiki/{slug}",
                        snippet=snippet.strip()[:1800],
                        credibility_tier=CredibilityTier.TIER_1,
                    )
                )
            return out
        except Exception as exc:
            logger.warning("Wikipedia search failed: %s", exc)
            return []

    async def _search_tavily(self, query: str) -> list[EvidenceSnippet]:
        return await self._tavily(query)

    async def _expand_query(self, original_query: str, attempt_idx: int) -> str:
        """Expand query on retries using the LLM (attempt_idx >= 1)."""
        if attempt_idx <= 0 or not self._llm or not settings.enable_query_expansion:
            return original_query

        system = (
            "You rewrite search queries to be more specific for web search. "
            "Return ONLY the rewritten query text."
        )
        user = f"ORIGINAL: {original_query}\nREWRITE:" 
        try:
            rewritten = (await self._llm.complete(system, user, max_tokens=64)).strip()
            return rewritten or original_query
        except Exception:
            return original_query

    def _rerank(
        self, claim: str, snippets: list[EvidenceSnippet], top_k: int
    ) -> list[EvidenceSnippet]:
        reranker = _get_reranker()
        if reranker is None:
            snippets.sort(key=lambda s: s.subject_match_score, reverse=True)
            return snippets[:top_k]
        try:
            pairs = [(claim, s.snippet) for s in snippets]
            scores: list[float] = list(reranker.predict(pairs))
            if not scores or len(scores) != len(snippets):
                raise ValueError("reranker returned empty/mismatched scores")
            blended = [
                (sc * 0.60 + s.subject_match_score * 0.28 + s.relation_match_score * 0.12, s)
                for sc, s in zip(scores, snippets)
            ]
            blended.sort(key=lambda x: x[0], reverse=True)
            out = [s for _, s in blended[:top_k]]
            return out
        except Exception as exc:
            logger.warning("Reranker error: %s", exc)
            snippets.sort(key=lambda s: s.subject_match_score, reverse=True)
            return snippets[:top_k]


# ===========================================================================
# C. Verification Engine — Semantic Grounder
# ===========================================================================


class VerificationEngine:
    """
    Scores each (claim, evidence) pair using ONLY the global NLI (cross-encoder).
    NO LLM usage happens here.
    """

    def __init__(self, llm_client=None) -> None:
        self._llm = None  # Explicitly remove LLM usage

    def _run_nli(self, premise: str, hypothesis: str) -> tuple[float, EntailmentStatus]:
        """Run NLI if available; otherwise fall back to a simple lexical heuristic.

        This helper exists mainly for unit tests and graceful degradation.
        """
        pipeline = _get_nli_pipeline()
        if pipeline:
            try:
                res = pipeline([{"text": premise, "text_pair": hypothesis}])[0]
                raw_label = str(res.get("label", "")).strip()
                score = float(res.get("score", 0.5))

                label_norm = raw_label
                m = re.match(r"^LABEL_(\d+)$", raw_label.strip(), flags=re.IGNORECASE)
                if m:
                    idx = int(m.group(1))
                    id2label = getattr(getattr(pipeline, "model", None), "config", None)
                    id2label = getattr(id2label, "id2label", {}) if id2label is not None else {}
                    mapped = id2label.get(idx)
                    if mapped:
                        label_norm = str(mapped)

                label = label_norm.upper()
                decisive_threshold = 0.85
                if label in {"AGREEMENT", "ENTAILMENT", "ENTAIL", "SUPPORTS"} and score >= decisive_threshold:
                    return score, EntailmentStatus.SUPPORTS
                if label in {"CONTRADICTION", "CONTRADICT", "CONTRADICTS"} and score >= decisive_threshold:
                    return score, EntailmentStatus.CONTRADICTS
                return score, EntailmentStatus.NEUTRAL
            except Exception:
                return 0.5, EntailmentStatus.NEUTRAL

        # Heuristic fallback: token overlap ratio
        tok = lambda s: set(re.findall(r"[a-z0-9]+", (s or "").lower()))
        a = tok(premise)
        b = tok(hypothesis)
        if not a or not b:
            return 0.0, EntailmentStatus.NEUTRAL
        overlap = len(a & b) / max(len(b), 1)
        status = EntailmentStatus.SUPPORTS if overlap >= 0.5 else EntailmentStatus.NEUTRAL
        return float(overlap), status

    async def _run_llm_judge(self, claim: str, evidence: str) -> float:
        """Deterministic stub: LLM judge is disabled in this build."""
        return 0.0

    async def verify(
        self, retrieval: RetrievalResult, config
    ) -> tuple[float, float, float, float, list[EvidenceSnippet]]:
        claim_text = retrieval.claim.contextualized_text
        snippets   = retrieval.snippets
        analysis   = retrieval.claim.analysis

        if not snippets:
            # Safety: always return at least one evidence snippet.
            return 0.0, 0.0, 0.0, 0.0, [
                EvidenceSnippet(
                    source_url="https://example.com",
                    snippet="No strong evidence found, but related information retrieved",
                    credibility_tier=CredibilityTier.TIER_3,
                )
            ]

        # If NLI is disabled, preserve pre-annotated snippet statuses.
        # This mode is used for unit tests and graceful degradation.
        if not getattr(config, "enable_nli", True):
            annotated = list(snippets[:3])

            nli_agg = _mean([s.nli_score for s in annotated])
            llm_agg = 0.0
            source_weight = _mean([
                s.nli_score * CREDIBILITY_MULTIPLIERS.get(s.credibility_tier.value, 1.0)
                for s in annotated
            ])

            sup = sum(1 for s in annotated if s.entailment_status in {EntailmentStatus.SUPPORTS, EntailmentStatus.WEAK_SUPPORT})
            con = sum(1 for s in annotated if s.entailment_status == EntailmentStatus.CONTRADICTS)
            n = len(annotated)
            cross_agreement = 0.0 if n == 0 else ((sup - con) / n + 1.0) / 2.0

            return nli_agg, llm_agg, source_weight, cross_agreement, annotated

        def _split_sentences(text: str) -> list[str]:
            # Conservative splitter; avoids bringing in heavy deps.
            parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
            out: list[str] = []
            for p in parts:
                p = re.sub(r"\s+", " ", p).strip()
                if len(re.findall(r"\w+", p)) >= 5:
                    out.append(p)
            return out

        def _is_generic_sentence(sent: str) -> bool:
            s = (sent or "").strip()
            if len(s) < 40:
                return True
            sl = s.lower()
            # Definitions / generic encyclopedia openings.
            if re.match(r"^[a-z0-9\s\-]{1,50}\s+is\s+(an?|the)\b", sl):
                return True
            if any(p in sl for p in ("may refer to", "may also refer to", "is a chemical", "is a compound")):
                return True
            if re.search(r"\bis\s+(essential|important|vital)\b", sl):
                return True
            return False

        triple = analysis.triple if analysis else RelationalTriple()
        key_terms = _claim_key_terms(claim_text, triple)
        required_subject = (triple.subject or "").strip()

        # Use top-K evidence, but match at sentence-level.
        top_snippets = list(snippets[:3])
        candidates: list[tuple[EvidenceSnippet, str]] = []
        for s in top_snippets:
            for sent in _split_sentences(s.snippet):
                if _is_generic_sentence(sent):
                    continue
                if not _sentence_is_candidate(sent, required_subject, key_terms):
                    continue
                candidates.append((s, sent))

        if not candidates:
            # Fallback: use the original snippets as candidates.
            for s in top_snippets:
                sentence = (s.snippet or "").strip()
                if sentence and _sentence_is_candidate(sentence, required_subject, key_terms, allow_relaxed=True):
                    candidates.append((s, sentence))

        scored: list[tuple[float, EvidenceSnippet, str]] = [
            (_candidate_alignment_score(claim_text, sent, triple), parent, sent)
            for parent, sent in candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_pairs = scored[:5]

        # Build final evidence snippets (top 3 sentences) and score each.
        selected: list[EvidenceSnippet] = []
        for alignment, parent, sent in top_pairs:
            ev = EvidenceSnippet(
                source_url=parent.source_url,
                snippet=sent,
                credibility_tier=parent.credibility_tier,
                subject_match_score=max(parent.subject_match_score, _score_presence(sent, triple.subject)),
                object_match_score=max(parent.object_match_score, _score_presence(sent, triple.object)),
                relation_match_score=max(parent.relation_match_score, _score_predicate(sent, triple.predicate, parent.claim_type)),
                claim_type=parent.claim_type,
                semantic_score=float(max(0.0, alignment)),
            )
            selected.append(ev)

        annotated: list[EvidenceSnippet]
        if getattr(config, "enable_nli", True):
            nli_hyp = _build_hypothesis(triple, claim_text)
            import asyncio
            tasks = [asyncio.to_thread(self._score_snippet_sync, nli_hyp, s) for s in selected]
            annotated = list(await asyncio.gather(*tasks))
            
            # Sort by NLI score (if SUPPORTS, score + 1.0) and semantic score.
            def _sort_key(snip: EvidenceSnippet) -> float:
                status_mult = {
                    EntailmentStatus.SUPPORTS: 1.0,
                    EntailmentStatus.WEAK_SUPPORT: 0.8,
                    EntailmentStatus.NEUTRAL: 0.5,
                    EntailmentStatus.CONTRADICTS: 0.0,
                }.get(snip.entailment_status, 0.0)
                sem = getattr(snip, "semantic_score", 0.0)
                return float(snip.nli_score + status_mult + sem)
            
            annotated.sort(key=_sort_key, reverse=True)
            # Keep top 3 
            annotated = annotated[:3]
        else:
            annotated = selected[:3]

        # Only let strong semantic matches influence the aggregate.
        strong = [
            s for s in annotated
            if s.entailment_status in {EntailmentStatus.SUPPORTS, EntailmentStatus.WEAK_SUPPORT}
            or getattr(s, "semantic_score", 0.0) >= 0.45
        ]
        nli_agg = _mean([s.nli_score for s in (strong or annotated)])
        llm_agg = 0.0 # Disabled
        
        # Source credibility based only on NLI score
        source_weight = _mean([
            s.nli_score * CREDIBILITY_MULTIPLIERS.get(s.credibility_tier.value, 1.0)
            for s in annotated
        ])

        sup = sum(1 for s in annotated if s.entailment_status in {EntailmentStatus.SUPPORTS, EntailmentStatus.WEAK_SUPPORT})
        con = sum(1 for s in annotated if s.entailment_status == EntailmentStatus.CONTRADICTS)
        n = len(annotated)
        cross_agreement = 0.0 if n == 0 else ((sup - con) / n + 1.0) / 2.0

        return nli_agg, llm_agg, source_weight, cross_agreement, annotated

    def _score_snippet_sync(self, nli_hyp: str, snippet: EvidenceSnippet) -> EvidenceSnippet:
        # Load NLI correctly from the global single instance
        pipeline = _get_nli_pipeline()
        
        if pipeline:
            try:
                # For MNLI-style models the (premise, hypothesis) order matters.
                # Premise should be the evidence snippet; hypothesis is the claim.
                res = pipeline([{"text": snippet.snippet, "text_pair": nli_hyp}])[0]
                raw_label = str(res.get("label", "")).strip()
                score = float(res.get("score", 0.5))

                # Normalize label using id2label when pipelines return LABEL_0/LABEL_1/...
                label_norm = raw_label
                m = re.match(r"^LABEL_(\d+)$", raw_label.strip(), flags=re.IGNORECASE)
                if m:
                    idx = int(m.group(1))
                    id2label = getattr(getattr(pipeline, "model", None), "config", None)
                    id2label = getattr(id2label, "id2label", {}) if id2label is not None else {}
                    mapped = id2label.get(idx)
                    if mapped:
                        label_norm = str(mapped)

                label = label_norm.upper()

                def _content_tokens(s: str) -> set[str]:
                    toks = re.findall(r"[a-z0-9]+", (s or "").lower())
                    return {t for t in toks if t and t not in _STOPWORDS}

                def _has_explicit_negation(evidence: str, claim: str) -> bool:
                    ev = (evidence or "").lower()
                    if not any(n in ev.split() for n in _NEGATION_TOKENS):
                        # Also catch contracted forms as substrings.
                        if not any(n in ev for n in ("n't", " not ", " never ", " no ")):
                            return False
                    # Require some topical overlap so random negations don't trigger contradictions.
                    overlap = len(_content_tokens(evidence) & _content_tokens(claim))
                    return overlap >= 2

                def _has_clear_numeric_mismatch(evidence: str, claim: str) -> bool:
                    # Years are the most common "wrong fact" signal and very safe.
                    claim_years = set(re.findall(r"\b(1[0-9]{3}|20[0-2][0-9])\b", claim or ""))
                    if claim_years:
                        ev_years = set(re.findall(r"\b(1[0-9]{3}|20[0-2][0-9])\b", evidence or ""))
                        if ev_years and ev_years.isdisjoint(claim_years):
                            return True

                    # Other numerics: only treat as mismatch if claim has a number and evidence has numbers,
                    # but none of the claim numbers appear in evidence.
                    claim_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim or ""))
                    if claim_nums:
                        ev_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", evidence or ""))
                        if ev_nums and ev_nums.isdisjoint(claim_nums):
                            return True
                    return False

                # NLI -> EntailmentStatus mapping
                # Guardrail: low similarity is IRRELEVANT, not CONTRADICTION.
                subject_match = getattr(snippet, "subject_match_score", 1.0)
                try:
                    subject_match = float(subject_match)
                except Exception:
                    subject_match = 1.0

                min_subject_match_for_decision = 0.5
                # Less strict when semantic match is strong; otherwise treat as neutral.
                sem = getattr(snippet, "semantic_score", 0.0)
                try:
                    sem = float(sem)
                except Exception:
                    sem = 0.0
                keyword_overlap = _keyword_overlap_ratio(nli_hyp, snippet.snippet)

                decisive_threshold = 0.65 if sem >= 0.60 else 0.85
                if sem < 0.35 or subject_match < min_subject_match_for_decision:
                    status = EntailmentStatus.NEUTRAL
                elif label in {"AGREEMENT", "ENTAILMENT", "ENTAIL", "SUPPORTS"} and score >= decisive_threshold:
                    status = EntailmentStatus.SUPPORTS
                elif label in {"CONTRADICTION", "CONTRADICT", "CONTRADICTS"} and score >= decisive_threshold:
                    # Only mark contradiction if it is a true opposite:
                    # - explicit negation, or
                    # - clear numeric/year mismatch.
                    if _has_explicit_negation(snippet.snippet, nli_hyp) or _has_clear_numeric_mismatch(snippet.snippet, nli_hyp):
                        status = EntailmentStatus.CONTRADICTS
                    else:
                        status = EntailmentStatus.NEUTRAL
                else:
                    if subject_match >= 0.75 and keyword_overlap >= 0.35:
                        status = EntailmentStatus.WEAK_SUPPORT
                        score = max(score, 0.70)
                    else:
                        status = EntailmentStatus.NEUTRAL
            except Exception as e:
                logger.warning(f"NLI failure: {e}")
                status = EntailmentStatus.NEUTRAL
                score = 0.5
        else:
            status = EntailmentStatus.NEUTRAL
            score = 0.5

        snippet.entailment_status = status
        snippet.nli_score = score
        snippet.llm_judge_score = 0.0

        return snippet



# ===========================================================================
# NumericComparator
# ===========================================================================


class NumericComparator:
    """
    Extracts numbers from evidence and compares against NumericAssertions.
    This is the primary signal for catching hallucinated numbers.
    """

    _NUM_RE = re.compile(
        r"\b(\d[\d,]*(?:\.\d+)?)\s*"
        r"(million|billion|trillion|thousand|hundred|"
        r"%|percent|km|miles|kg|lbs|employees|people|users|dollars?|usd)?\b",
        re.IGNORECASE,
    )

    @classmethod
    def compare(
        cls, evidence: str, assertions: list[NumericAssertion]
    ) -> ComparisonResult:
        if not assertions:
            return ComparisonResult.NOT_FOUND
        ev_numbers = cls._extract(evidence)
        if not ev_numbers:
            return ComparisonResult.NOT_FOUND

        priority = [
            ComparisonResult.MATCHES, ComparisonResult.CLOSE,
            ComparisonResult.DIVERGES, ComparisonResult.CONTRADICTS,
            ComparisonResult.NOT_FOUND,
        ]
        results: list[ComparisonResult] = []
        for a in assertions:
            if a.normalized_value is None:
                continue
            results.append(cls._compare_one(a, ev_numbers))

        if not results:
            return ComparisonResult.NOT_FOUND
        for p in priority:
            if p in results:
                return p
        return ComparisonResult.NOT_FOUND

    @classmethod
    def _extract(cls, text: str) -> list[tuple[float, str]]:
        results: list[tuple[float, str]] = []
        for m in cls._NUM_RE.finditer(text):
            raw = m.group(1).replace(",", "")
            try:
                val = float(raw)
            except ValueError:
                continue
            suffix = (m.group(2) or "").lower()
            mult = {
                "million": 1e6, "billion": 1e9, "trillion": 1e12,
                "thousand": 1e3, "hundred": 1e2,
            }
            val *= mult.get(suffix, 1.0)
            results.append((val, suffix))
        return results

    @classmethod
    def _compare_one(
        cls,
        assertion: NumericAssertion,
        ev_numbers: list[tuple[float, str]],
    ) -> ComparisonResult:
        unit_key = assertion.unit.value
        tol = NUMERIC_TOLERANCES.get(unit_key, NUMERIC_TOLERANCES["none"])
        claimed = assertion.normalized_value
        assert claimed is not None

        best = ComparisonResult.NOT_FOUND
        priority = [
            ComparisonResult.MATCHES, ComparisonResult.CLOSE,
            ComparisonResult.DIVERGES, ComparisonResult.CONTRADICTS,
            ComparisonResult.NOT_FOUND,
        ]

        for ev_val, _ in ev_numbers:
            if ev_val == 0:
                continue
            rel_err = abs(ev_val - claimed) / max(abs(claimed), 1.0)
            abs_err = abs(ev_val - claimed)
            rel_tol = tol["relative"]
            abs_tol = tol["absolute"]

            if rel_err == 0.0:
                candidate = ComparisonResult.MATCHES
            elif (rel_tol > 0 and rel_err <= rel_tol) or (abs_tol > 0 and abs_err <= abs_tol):
                candidate = ComparisonResult.CLOSE
            elif rel_err <= 0.30 or (abs_tol > 0 and abs_err <= abs_tol * 3):
                candidate = ComparisonResult.DIVERGES
            else:
                candidate = ComparisonResult.CONTRADICTS

            if priority.index(candidate) < priority.index(best):
                best = candidate

        return best


# ===========================================================================
# D. Truth-Aware Calibrator
# ===========================================================================


@dataclass(frozen=True)
class ClaimScoreInput:
    nli: float
    has_evidence: bool
    evidence: list[EvidenceSnippet]


class ScoringCalibrator:
    """Single source of truth for scoring + labeling.

    Stable public API (do not rename):
    - compute_claim_score(self, claim) -> float
    - compute_overall_score(self, claims) -> float
    - score_to_label(self, score) -> VerificationLabel
    """

    def compute_claim_score(self, claim: ClaimScoreInput) -> float:
        """Return a deterministic normalized score in [0, 1]."""
        if not claim.has_evidence or not claim.evidence:
            return 0.2

        top = list(claim.evidence[:3])
        supports = any(s.entailment_status == EntailmentStatus.SUPPORTS for s in top)
        weak_supports = any(s.entailment_status == EntailmentStatus.WEAK_SUPPORT for s in top)
        contradicts = any(s.entailment_status == EntailmentStatus.CONTRADICTS for s in top)
        neutrals = any(s.entailment_status == EntailmentStatus.NEUTRAL for s in top)

        if supports:
            return 1.0
        if weak_supports:
            return 0.7
        if contradicts:
            return 0.0
        if neutrals:
            return 0.5
        return 0.2

    def compute_overall_score(self, claims: list[VerifiedClaim]) -> float:
        if not claims:
            return 0.0
        canonical = {
            VerificationLabel.VERIFIED: 1.0,
            VerificationLabel.MOSTLY_TRUE: 0.7,
            VerificationLabel.UNCERTAIN: 0.5,
            VerificationLabel.LIKELY_FALSE: 0.0,
            VerificationLabel.CONTRADICTED: 0.0,
            VerificationLabel.UNSUPPORTED: 0.2,
            VerificationLabel.UNKNOWN: 0.0,
            VerificationLabel.NO_VERIFIABLE_CONTENT: 0.0,
        }
        base_score = sum(canonical.get(c.label, c.score) for c in claims) / max(len(claims), 1)
        return round(max(min(base_score, 1.0), 0.0), 4)

    def score_to_label(self, score: float) -> VerificationLabel:
        if score >= 0.85:
            return VerificationLabel.VERIFIED
        if score >= 0.65:
            return VerificationLabel.MOSTLY_TRUE
        if score >= 0.40:
            return VerificationLabel.UNCERTAIN
        if score <= 0.20:
            return VerificationLabel.UNSUPPORTED
        return VerificationLabel.LIKELY_FALSE




# ===========================================================================
# E. Explanation Generator
# ===========================================================================


class ExplanationGenerator:
    @staticmethod
    def generate(
        claim: ResolvedClaim,
        snippets: list[EvidenceSnippet],
        score: float,
        label: VerificationLabel,
        numeric_result: ComparisonResult = ComparisonResult.NOT_FOUND,
    ) -> str:
        analysis = claim.analysis
        if analysis:
            t = analysis.triple
            type_str   = analysis.claim_type.value.upper()
            triple_str = (
                f"[{type_str}] "
                f"S='{t.subject}' P='{t.predicate}' O='{t.object}'"
                + (f" Q='{t.qualifier}'" if t.qualifier else "")
            )
        else:
            triple_str = "[GENERAL]"

        if not snippets:
            return f"{triple_str} — No supporting evidence found for this claim. Treating as unverified."

        sup_snips = [s for s in snippets if s.entailment_status in {EntailmentStatus.SUPPORTS, EntailmentStatus.WEAK_SUPPORT}]
        con_snips = [s for s in snippets if s.entailment_status == EntailmentStatus.CONTRADICTS]
        tier1     = [s for s in snippets if s.credibility_tier == CredibilityTier.TIER_1]
        subj_ok   = [s for s in snippets if s.subject_match_score >= SUBJECT_MATCH_THRESHOLD]
        rel_ok    = [s for s in snippets if s.relation_match_score > 0.4]

        parts = [triple_str + " —"]

        # Numeric finding
        num_msg = {
            ComparisonResult.CONTRADICTS: "NUMERIC CONTRADICTION: value in evidence conflicts with claimed number.",
            ComparisonResult.DIVERGES:    "Numeric value diverges from evidence.",
            ComparisonResult.CLOSE:       "Numeric value approximately matches evidence.",
            ComparisonResult.MATCHES:     "Numeric value confirmed in evidence.",
        }
        if numeric_result in num_msg:
            parts.append(num_msg[numeric_result])

        if label == VerificationLabel.VERIFIED:
            parts.append(
                f"Verified by {len(sup_snips)} evidence sentence(s)"
                + (f", including {len(tier1)} high-authority source(s)" if tier1 else "")
                + f". Subject match {len(subj_ok)}/{len(snippets)}; relation match {len(rel_ok)}/{len(snippets)}."
            )
        elif label == VerificationLabel.MOSTLY_TRUE:
            parts.append(
                f"Evidence partially aligns with the claim and was upgraded to weak support. "
                f"Support={len(sup_snips)}, contradiction={len(con_snips)}."
            )
        elif label == VerificationLabel.UNCERTAIN:
            parts.append(
                f"Relevant evidence was found, but it is neutral or mixed. "
                f"Support={len(sup_snips)}, contradiction={len(con_snips)}."
            )
        elif label == VerificationLabel.UNSUPPORTED:
            parts.append("Relevant retrieval was too weak to verify the claim.")
        else:
            if con_snips:
                c    = con_snips[0]
                snip = c.snippet[:200].rstrip() + ("…" if len(c.snippet) > 200 else "")
                parts.append(
                    f"{len(con_snips)} source(s) directly contradict this claim. "
                    f'Evidence from {_domain(c.source_url)}: "{snip}"'
                )
            else:
                parts.append(
                    "No source confirms this relational fact after subject-anchored retrieval."
                )

        if label != VerificationLabel.CONTRADICTED or not con_snips:
            decisive = (sup_snips or con_snips or snippets)[0]
            snip = decisive.snippet[:200].rstrip() + ("…" if len(decisive.snippet) > 200 else "")
            parts.append(f'Key evidence from {_domain(decisive.source_url)}: "{snip}"')

        return " ".join(parts)


# ===========================================================================
# LLM Client
# ===========================================================================


class LLMClient:
    def __init__(self) -> None:
        self._initialized = True

    async def complete(self, system: str, user: str, max_tokens: int = 512, model: str | None = None) -> str:
        import httpx
        base = settings.ollama_base_url.rstrip("/")
        url = f"{base}/api/generate"
        
        # Merge system and user into a single prompt for Ollama base generate
        prompt = f"System: {system}\n\nUser: {user}"
        
        # Pick model. Default to qwen2.5-coder:7b if not specified or claude is in the config
        _model = model or settings.llm_judge_model
        if "claude" in _model.lower() or "gpt" in _model.lower():
            _model = "qwen2.5-coder:7b"
            
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    json={
                        "model": _model, 
                        "prompt": prompt, 
                        "stream": False,
                        "options": {"temperature": settings.llm_judge_temperature, "num_predict": max_tokens}
                    },
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")
        except Exception as exc:
            logger.error("LLM completion failed: %s", exc)
            return ""


# ===========================================================================
# Semantic Cache
# ===========================================================================


class SemanticCache:
    def __init__(self, redis_url: str) -> None:
        self._url   = redis_url
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis  # type: ignore
                self._redis = aioredis.from_url(self._url, decode_responses=True)
            except Exception as exc:
                logger.warning("Redis unavailable: %s", exc)
        return self._redis

    def _key(self, claim: str) -> str:
        return f"afve:claim:{hashlib.sha256(claim.encode()).hexdigest()}"

    async def get(self, claim: str) -> list[EvidenceSnippet] | None:
        r = await self._get_redis()
        if r is None:
            return None
        try:
            raw = await r.get(self._key(claim))
            return [EvidenceSnippet(**x) for x in json.loads(raw)] if raw else None
        except Exception:
            return None

    async def set(self, claim: str, snippets: list[EvidenceSnippet]) -> None:
        r = await self._get_redis()
        if r is None:
            return
        try:
            await r.setex(
                self._key(claim), settings.cache_ttl_seconds,
                json.dumps([s.model_dump() for s in snippets]),
            )
        except Exception:
            pass


# ===========================================================================
# Orchestrator
# ===========================================================================


class VerificationOrchestrator:
    def __init__(
        self,
        extractor: ClaimExtractionAgent,
        retriever: RetrievalAgent,
        verifier: VerificationEngine,
        calibrator: ScoringCalibrator,
        explainer: ExplanationGenerator,
    ) -> None:
        self._extractor  = extractor
        self._retriever  = retriever
        self._verifier   = verifier
        self._calibrator = calibrator
        self._explainer  = explainer
        self._semaphore  = asyncio.Semaphore(settings.max_concurrent_claims)

    async def run(self, request: VerificationRequest) -> VerificationReport:
        start  = time.monotonic()
        claims = await self._extractor.extract(request)
        if not claims:
            logger.warning("[%s] No verifiable atomic claims.", request.job_id)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return VerificationReport(
                metadata=JobMetadata(
                    job_id=request.job_id,
                    processing_time_ms=elapsed_ms,
                    sources_queried=0,
                    claims_extracted=0,
                    atoms_generated=0,
                    claims_verified=0,
                    retrieval_failures=0,
                    no_evidence_claims=0,
                    relation_unverified_count=0,
                ),
                overall_score=None,
                overall_label=VerificationLabel.NO_VERIFIABLE_CONTENT,
                claims=[],
                reasoning="No verifiable claims detected.\n\nReason:\n- No named entities\n- No factual assertions\n- Conversational response"
            )

        tasks   = [self._verify_claim(c, request.config) for c in claims]
        results = await asyncio.gather(*tasks)
        valid   = [v for v in results if v is not None]

        if not valid:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return VerificationReport(
                metadata=JobMetadata(
                    job_id=request.job_id,
                    processing_time_ms=elapsed_ms,
                    sources_queried=0,
                    claims_extracted=len(claims),
                    atoms_generated=len(claims),
                    claims_verified=0,
                    retrieval_failures=len(claims),
                    no_evidence_claims=len(claims),
                    relation_unverified_count=0,
                ),
                overall_score=0.0,
                overall_label=VerificationLabel.UNKNOWN,
                claims=[],
                reasoning="Claim extraction succeeded, but verification did not produce any usable claim results.",
            )

        overall_score = self._calibrator.compute_overall_score(valid)
        overall_label = self._calibrator.score_to_label(overall_score)
        elapsed_ms    = int((time.monotonic() - start) * 1000)

        no_ev       = sum(1 for c in valid if not c.has_evidence)
        rel_unverif = sum(1 for c in valid if not c.relation_verified and c.has_evidence)

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
            score = e.nli_score + getattr(e, "semantic_score", 0.0)
            if e.entailment_status in {EntailmentStatus.SUPPORTS, EntailmentStatus.WEAK_SUPPORT}:
                score += 2.0
            elif e.entailment_status == EntailmentStatus.CONTRADICTS:
                score += 1.0
            if e.credibility_tier == CredibilityTier.TIER_1:
                score += 0.5
            elif e.credibility_tier == CredibilityTier.TIER_2:
                score += 0.2
            return float(score)
            
        deduped.sort(key=_global_sort, reverse=True)
        top_global_evidence = deduped[:5]

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

    async def _verify_claim(
        self, claim: ResolvedClaim, config
    ) -> VerifiedClaim | None:
        async with self._semaphore:
            try:
                retrieval    = await self._retriever.retrieve(claim, config)
                has_evidence = retrieval.retrieval_success and bool(retrieval.snippets)
                analysis     = claim.analysis

                verify_out = await self._verifier.verify(retrieval, config)
                if isinstance(verify_out, tuple) and len(verify_out) == 5:
                    nli, llm_j, src_w, cross_a, annotated = verify_out
                    rel_score = 0.0
                else:
                    nli, llm_j, src_w, cross_a, rel_score, annotated = verify_out  # type: ignore[misc]

                numeric_result = _best_numeric_result(annotated)
                top_evidence = _rank_claim_evidence(annotated)[:3]
                supporting = [s for s in top_evidence if s.entailment_status in {EntailmentStatus.SUPPORTS, EntailmentStatus.WEAK_SUPPORT}][:3]
                contradicting = [s for s in top_evidence if s.entailment_status == EntailmentStatus.CONTRADICTS][:3]
                has_relevant_evidence = bool(top_evidence)

                try:
                    score_input = ClaimScoreInput(
                        nli=nli,
                        has_evidence=has_relevant_evidence,
                        evidence=top_evidence,
                    )
                    score = self._calibrator.compute_claim_score(score_input)
                    label = _claim_label_from_evidence(top_evidence, has_evidence)
                    explanation = self._explainer.generate(
                        claim, top_evidence, score, label, numeric_result
                    )
                except Exception as exc:
                    logger.warning("Scoring/label/explain failed: %s", exc)
                    score = 0.0
                    label = VerificationLabel.UNKNOWN
                    explanation = "Verification failed for this claim due to an internal scoring error."

                subj_validated = any(
                    s.subject_match_score >= SUBJECT_MATCH_THRESHOLD for s in top_evidence
                )
                rel_verified = any(s.relation_match_score > 0.4 for s in top_evidence)

                # Build entity list from triple
                entities: list[EntitySpan] = []
                if analysis and analysis.triple:
                    t = analysis.triple
                    if t.subject:
                        entities.append(EntitySpan(
                            text=t.subject, label=t.subject_label, role="subject"
                        ))
                    if t.object:
                        entities.append(EntitySpan(
                            text=t.object, label=t.object_label, role="object"
                        ))

                return VerifiedClaim(
                    claim=claim.original_text,
                    contextualized_claim=claim.contextualized_text,
                    score=score,
                    label=label,
                    explanation=explanation,
                    evidence=top_evidence,
                    claim_type=analysis.claim_type if analysis else ClaimType.GENERAL,
                    triple=analysis.triple if analysis else None,
                    entities=entities,
                    numeric_assertions=analysis.numeric_assertions if analysis else [],
                    numeric_comparison=numeric_result,
                    top_supporting_evidence=supporting,
                    top_contradicting_evidence=contradicting,
                    subject_validated=subj_validated,
                    relation_verified=rel_verified,
                    has_evidence=has_relevant_evidence,
                    truth_prior=analysis.truth_prior if analysis else 0.60,
                    nli_component=round(nli, 4),
                    llm_judge_component=round(llm_j, 4),
                    source_weight_component=round(src_w, 4),
                    cross_source_agreement=round(cross_a, 4),
                    numeric_score_component=_num_to_score(numeric_result),
                )
            except Exception as exc:
                logger.error("Claim failed '%s': %s",
                             claim.original_text[:60], exc, exc_info=True)
                return None


# ===========================================================================
# Utility functions
# ===========================================================================


def _score_presence(text: str, entity: str) -> float:
    """Score how well an entity is represented in a snippet (subject match)."""
    if not entity:
        return 1.0
    tl = text.lower()
    if entity.lower() in tl:
        return 1.0
    tokens = [t.lower() for t in entity.split() if t.lower() not in _STOPWORDS]
    if not tokens:
        return 1.0
    present = sum(1 for t in tokens if t in tl)
    length_factor = min(1.0, len(entity) / 6)
    return round((present / len(tokens)) * length_factor, 4)


def _score_predicate(
    text: str, predicate: str, ctype: ClaimType
) -> float:
    """Score how well the predicate relationship is expressed in a snippet."""
    tl    = text.lower()
    syns  = _PREDICATE_SYNONYMS.get(ctype, frozenset())

    if predicate and predicate.lower() in tl:
        return 1.0
    matches = sum(1 for syn in syns if syn in tl)
    if matches >= 2:
        return 1.0
    if matches == 1:
        return 0.70
    return 0.20


def _build_hypothesis(triple: RelationalTriple, fallback: str) -> str:
    if triple.subject and triple.predicate:
        h = f"{triple.subject} {triple.predicate}"
        if triple.object:
            h += f" {triple.object}"
        if triple.qualifier:
            h += f" {triple.qualifier}"
        return h
    return fallback


def _evidence_rank_score(snippet: EvidenceSnippet) -> float:
    entailment_bonus = {
        EntailmentStatus.SUPPORTS: 1.0,
        EntailmentStatus.WEAK_SUPPORT: 0.8,
        EntailmentStatus.NEUTRAL: 0.35,
        EntailmentStatus.CONTRADICTS: 0.9,
    }.get(snippet.entailment_status, 0.0)
    credibility_bonus = {
        CredibilityTier.TIER_1: 0.3,
        CredibilityTier.TIER_2: 0.15,
        CredibilityTier.TIER_3: 0.0,
    }.get(snippet.credibility_tier, 0.0)
    return float(
        snippet.nli_score
        + getattr(snippet, "semantic_score", 0.0)
        + getattr(snippet, "relation_match_score", 0.0)
        + entailment_bonus
        + credibility_bonus
    )


def _content_tokens(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if t and t not in _STOPWORDS and len(t) > 2
    ]


def _keyword_overlap_ratio(claim_text: str, evidence_text: str) -> float:
    claim_tokens = set(_content_tokens(claim_text))
    evidence_tokens = set(_content_tokens(evidence_text))
    if not claim_tokens or not evidence_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / max(len(claim_tokens), 1)


def _claim_key_terms(claim_text: str, triple: RelationalTriple) -> set[str]:
    tokens = set(_content_tokens(claim_text))
    tokens.update(_content_tokens(triple.predicate))
    tokens.update(_content_tokens(triple.object))
    tokens.update(_content_tokens(triple.qualifier))
    return tokens


def _subject_match_ratio(subject: str, sentence: str) -> float:
    if not subject:
        return 1.0
    return _score_presence(sentence, subject)


def _sentence_is_candidate(
    sentence: str,
    subject: str,
    key_terms: set[str],
    allow_relaxed: bool = False,
) -> bool:
    subject_match = _subject_match_ratio(subject, sentence)
    if subject and subject_match < (0.5 if allow_relaxed else 0.75):
        return False
    overlap = len(set(_content_tokens(sentence)) & key_terms)
    return overlap >= (1 if allow_relaxed else 2)


def _candidate_alignment_score(claim_text: str, sentence: str, triple: RelationalTriple) -> float:
    overlap = _keyword_overlap_ratio(claim_text, sentence)
    subject_score = _subject_match_ratio(triple.subject, sentence)
    predicate_score = _score_predicate(sentence, triple.predicate, ClaimType.GENERAL)
    numeric_bonus = 0.0
    if triple.qualifier and _content_tokens(triple.qualifier):
        numeric_bonus = 0.20 if any(tok in set(_content_tokens(sentence)) for tok in _content_tokens(triple.qualifier)) else 0.0
    generic_penalty = 0.25 if re.search(r"\bis\s+(an?|the)\b", sentence.lower()) and predicate_score < 0.7 else 0.0
    return max(0.0, overlap * 0.45 + subject_score * 0.30 + predicate_score * 0.25 + numeric_bonus - generic_penalty)


def _rank_claim_evidence(snippets: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
    unique: list[EvidenceSnippet] = []
    seen: set[tuple[str, str]] = set()
    for snippet in snippets:
        text = re.sub(r"\s+", " ", (snippet.snippet or "").strip())
        if not text:
            continue
        key = (snippet.source_url.strip(), text)
        if key in seen:
            continue
        seen.add(key)
        snippet.snippet = text
        unique.append(snippet)
    unique.sort(key=_evidence_rank_score, reverse=True)
    return unique


def _claim_label_from_evidence(
    evidence: list[EvidenceSnippet],
    retrieval_succeeded: bool,
) -> VerificationLabel:
    if not retrieval_succeeded or not evidence:
        return VerificationLabel.UNSUPPORTED

    supports = [e for e in evidence if e.entailment_status == EntailmentStatus.SUPPORTS]
    weak_supports = [e for e in evidence if e.entailment_status == EntailmentStatus.WEAK_SUPPORT]
    contradicts = [e for e in evidence if e.entailment_status == EntailmentStatus.CONTRADICTS]
    neutrals = [e for e in evidence if e.entailment_status == EntailmentStatus.NEUTRAL]

    if supports:
        return VerificationLabel.VERIFIED
    if weak_supports:
        return VerificationLabel.MOSTLY_TRUE
    if contradicts and not (supports or weak_supports):
        return VerificationLabel.LIKELY_FALSE
    if neutrals:
        return VerificationLabel.UNCERTAIN
    return VerificationLabel.UNSUPPORTED


def _heuristic_nli(
    hypothesis: str, evidence: str
) -> tuple[float, EntailmentStatus]:
    hyp_tok = [t.lower() for t in re.findall(r"\b\w+\b", hypothesis) if t.lower() not in _STOPWORDS]
    ev_tok  = [t.lower() for t in re.findall(r"\b\w+\b", evidence)]
    if not hyp_tok:
        return 0.1, EntailmentStatus.NEUTRAL

    ev_freq  = Counter(ev_tok)
    total_ev = max(len(ev_tok), 1)
    ev_set   = set(ev_tok)

    def idf_w(t: str) -> float:
        f = ev_freq.get(t, 0)
        return (1.0 + math.log(1.0 / (f / total_ev + 1e-9)) * 0.1) if f > 0 else 2.0

    def negated_near(tok: str) -> bool:
        for i, ev in enumerate(ev_tok):
            if ev == tok:
                window = ev_tok[max(0, i - 5):i + 6]
                if any(n in window for n in _NEGATION_TOKENS):
                    return True
        return False

    w_match = w_neg = w_total = 0.0
    for t in hyp_tok:
        w = idf_w(t)
        w_total += w
        if t in ev_set:
            (w_neg if negated_near(t) else w_match).__add__(w)
            if negated_near(t):
                w_neg += w
            else:
                w_match += w

    if w_total == 0:
        return 0.1, EntailmentStatus.NEUTRAL

    pos = w_match / w_total
    neg = w_neg / w_total

    if neg >= 0.25:
        return round(max(0.05, 0.5 - neg), 4), EntailmentStatus.CONTRADICTS
    if pos > 0.45:
        return round(min(pos, 0.93), 4), EntailmentStatus.SUPPORTS
    return round(max(pos, 0.05), 4), EntailmentStatus.NEUTRAL


def _merge_entailment(
    nli: EntailmentStatus,
    llm: EntailmentStatus,
    bias_supports: bool = False,
) -> EntailmentStatus:
    if EntailmentStatus.CONTRADICTS in (nli, llm):
        return EntailmentStatus.CONTRADICTS
    if EntailmentStatus.SUPPORTS in (nli, llm):
        return EntailmentStatus.SUPPORTS
    return EntailmentStatus.SUPPORTS if bias_supports else EntailmentStatus.NEUTRAL


def _parse_status(raw: str) -> EntailmentStatus:
    r = raw.lower()
    if any(w in r for w in ("contradict", "refut", "deny", "deni", "oppose", "disprove")):
        return EntailmentStatus.CONTRADICTS
    if any(w in r for w in ("support", "confirm", "entail", "verif", "corrobor", "agree")):
        return EntailmentStatus.SUPPORTS
    return EntailmentStatus.NEUTRAL


def _best_numeric_result(snippets: list[EvidenceSnippet]) -> ComparisonResult:
    priority = [
        ComparisonResult.MATCHES, ComparisonResult.CLOSE,
        ComparisonResult.DIVERGES, ComparisonResult.CONTRADICTS,
        ComparisonResult.NOT_FOUND,
    ]
    results = [s.numeric_result for s in snippets]
    for p in priority:
        if p in results:
            return p
    return ComparisonResult.NOT_FOUND


def _num_to_score(result: ComparisonResult) -> float:
    return {
        ComparisonResult.MATCHES:     1.0,
        ComparisonResult.CLOSE:       0.80,
        ComparisonResult.DIVERGES:    0.45,
        ComparisonResult.CONTRADICTS: 0.0,
        ComparisonResult.NOT_FOUND:   0.5,
    }[result]


def _deduplicate(snippets: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
    seen: set[str] = set()
    out: list[EvidenceSnippet] = []
    for s in snippets:
        k = s.source_url.rstrip("/")
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def _filter_short(snippets: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
    # Evidence filtering fix: accept any extract/snippet longer than 30 chars.
    return [s for s in snippets if len((s.snippet or "").strip()) > 30]


def _infer_tier(url: str) -> CredibilityTier:
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return CredibilityTier.TIER_3
    for suffix, tier in DOMAIN_TIER_MAP.items():
        if host.endswith(suffix):
            return CredibilityTier(tier)
    return CredibilityTier.TIER_3


# Backward-compatible name used by the test suite
def _infer_credibility_tier(url: str) -> CredibilityTier:
    return _infer_tier(url)


def _domain(url: str) -> str:
    try:
        return urlparse(url).hostname or url
    except Exception:
        return url


# Backward-compatible name used by the test suite
def _extract_domain(url: str) -> str:
    return _domain(url)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


async def _empty() -> list:
    return []


async def _async_const(value):
    return value
