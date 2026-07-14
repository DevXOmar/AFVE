# AFVE Pipeline Feature Handoff

This document is a technical handoff for improving the current AFVE pipeline. It describes what the system does today, how it calculates scores, what heuristics are active, and where the design has TODO-shaped gaps.

Primary implementation files:

- `main.py`: FastAPI app, dependency wiring, endpoints, metrics.
- `agents.py`: claim extraction, relational decomposition, retrieval, NLI verification, numeric comparison, scoring, explanations, orchestration.
- `models.py`: Pydantic schemas and enums.
- `config.py`: settings, weights, thresholds, query templates, domain credibility tiers.

## High-Level Flow

1. A user submits a `VerificationRequest` with `prompt`, `model_output`, `job_id`, and `config`.
2. `main.py` builds a `VerificationOrchestrator`.
3. `ClaimExtractionAgent` splits the model output into verifiable sentences.
4. `RelationalDecomposer` splits compound sentences into atomic claims and extracts a relational triple for each claim.
5. `RetrievalAgent` performs subject-first retrieval from Wikipedia and optionally Tavily/SerpAPI.
6. Retrieved snippets are filtered, scored for subject/object/relation match, reranked, and reduced to the top evidence.
7. `VerificationEngine` selects evidence sentences and runs NLI against a hypothesis built from the relational triple.
8. Numeric comparison, unit validation, temporal validation, entity-existence heuristics, and cross-claim checks are applied.
9. `ScoringCalibrator` turns evidence statuses into per-claim scores and labels.
10. Claim scores are weighted by prompt relevance to produce an overall report.

## Public API Surface

### Main Verification Endpoint

`POST /verify`

Input:

- `job_id`: UUID.
- `prompt`: original user prompt.
- `model_output`: text to verify.
- `config`: optional `VerificationConfig`.

Output:

- `VerificationReport`, including claim-level scores, labels, evidence, metadata, global evidence, uncertainty signals, and scoring note.

### Ollama Generation Endpoints

`POST /generate`

- Calls local Ollama `/api/generate`.
- Returns raw model output without verification.

`POST /run`

- Calls Ollama to generate text, then verifies it.

Default Ollama base URL is `http://127.0.0.1:11434`.

## Config Flags

`VerificationConfig` supports:

- `strictness`: default `0.8`, currently not heavily used in scoring.
- `max_retrieval_attempts`: `1..5`, default `3`.
- `top_k_snippets`: `1..20`, default `5`.
- `enable_llm_judge`: default `True`, but the current `VerificationEngine` has the LLM judge disabled/stubbed.
- `enable_nli`: default `True`.
- `enable_web_search`: default `True`.
- `enable_kb_search`: default `True`.
- `enable_relational_decomposition`: default `True`, but current extraction path still calls the decomposer.
- `enable_multi_claim_splitting`: default `True`.

Environment/settings flags:

- `AFVE_ENABLE_SEMANTIC_CACHE`: default `True`.
- `AFVE_ENABLE_RERANKER`: default `True`.
- `AFVE_ENABLE_COREFERENCE_RESOLUTION`: default `False`.
- `AFVE_ENABLE_QUERY_EXPANSION`: default `False`.
- `AFVE_ENABLE_LLM_TRIPLE_EXTRACTION`: default `False`.

## Core Data Model

### RelationalTriple

Every atomic claim is represented as:

- `subject`: entity the claim is about.
- `predicate`: relation or verb phrase.
- `object`: relation target.
- `qualifier`: date, number, or precision modifier.
- labels for subject/object/qualifier.

Example:

`"Elon Musk co-founded Tesla in 2003"` becomes:

- subject: `Elon Musk`
- predicate: `co-founded`
- object: `Tesla`
- qualifier: `2003`

The retrieval philosophy is subject-first: retrieve the subject article/context, then verify whether the predicate/object/qualifier is stated there.

### Claim Types

Current `ClaimType` values:

- `date`
- `membership`
- `attribute`
- `authorship`
- `role`
- `location`
- `causal`
- `comparative`
- `existence`
- `general`

### Claim Categories

- `factual`
- `conceptual`
- `abstract`

Conceptual/explanation prompts get more relaxed scoring in some cases, especially science-like domains.

### Entailment Statuses

- `supports`
- `weak_support`
- `contradicts`
- `neutral`

### Verification Labels

- `VERIFIED`
- `MOSTLY_TRUE`
- `LIKELY_FALSE`
- `UNCERTAIN`
- `UNSUPPORTED`
- `CONTRADICTED`
- `UNKNOWN`
- `NO_VERIFIABLE_CONTENT`

## Claim Extraction

`ClaimExtractionAgent.extract()`:

1. Detects verification mode from prompt:
   - explanation mode if prompt contains markers like `explain`, `describe`, `how`, `what is`, `why does`, `tell me about`.
   - otherwise factual mode.
2. Splits model output into sentences using regex.
3. Filters non-verifiable content:
   - rejects very short text.
   - rejects questions.
   - rejects conversational/meta openings like `I think`, `hello`, `sure`, `hope this helps`, etc.
   - rejects hedged non-answers such as `exact figure is not available`, `cannot confirm`, `I don't have current data`, and `as of my last update`.
4. If nothing is verifiable, may fallback to using the full response if it has entities, digits, or factual verbs.
5. Decomposes each verifiable sentence into atomic claims.
6. Drops abstract analogy/stylistic atoms.
7. Calculates prompt relevance.
8. Deduplicates claims.
9. Sorts by claim weight, relevance, qualifier presence, subject presence.
10. Keeps at most 6 claims.

### Prompt Relevance

`_claim_relevance_score(prompt, claim_text, analysis)`:

- Tokenizes prompt and claim, removing stopwords.
- Measures overlap between prompt focus terms and claim terms.
- Adds direct bonuses:
  - subject terms overlap prompt: `+0.35`
  - object terms overlap prompt: `+0.20`
  - predicate terms overlap prompt: `+0.15`
  - qualifier terms overlap prompt: `+0.20`
  - prompt contains `when` and claim has qualifier terms: `+0.20`
  - prompt contains `who` and subject terms overlap: `+0.20`
- Clamps at `1.0`.

Claims below relevance `0.20` are filtered out.

### Claim Weight

`_claim_weight_from_relevance()`:

- relevance `>= 0.75`: weight `1.0`
- relevance `>= 0.40`: weight `0.6`
- otherwise: weight `0.3`

Overall score is a weighted average using these weights.

## Relational Decomposition

`RelationalDecomposer` is mostly regex/heuristic-based.

### Multi-Claim Splitting

If enabled, compound claims are split on:

- coordinating `and` before likely new predicate.
- semicolons.
- contrastive clauses like `while`, `whereas`, `although`, `though`, `but`.

If later split clauses start lowercase and the first clause had a capitalized subject, the subject is prepended.

Example:

`"Apple was founded in 1976 and employs 150,000 people"`

becomes:

- `Apple was founded in 1976`
- `Apple employs 150,000 people`

### Triple Extraction

Heuristic extraction:

- subject: first meaningful capitalized entity sequence after stopword/query-noise filtering.
- predicate: first matched verb pattern.
- object: first capitalized entity after predicate.
- qualifier: first year, otherwise a prominent number if no object.

Verb pattern priority:

1. role verbs
2. date verbs
3. authorship verbs
4. membership verbs
5. causal verbs
6. location verbs
7. comparative verbs

Optional LLM triple extraction exists but only runs when:

- no subject was found,
- `settings.enable_llm_triple_extraction` is true,
- an LLM client is available.

Current default is disabled.

Sentence-initial conjunctions and discourse markers such as `While`, `However`, and `Although` are explicitly blocked from becoming subjects.

### Claim Classification

Claim type is chosen by regex order:

- role pattern -> `ROLE`
- date verbs or year -> `DATE`
- authorship verbs -> `AUTHORSHIP`
- membership verbs -> `MEMBERSHIP`
- causal verbs -> `CAUSAL`
- location verbs -> `LOCATION`
- comparative verbs -> `COMPARATIVE`
- numeric/percent -> `ATTRIBUTE`
- subject with no predicate -> `EXISTENCE`
- fallback -> `GENERAL`

### Numeric Assertions

The decomposer extracts:

- years as `NumericUnit.YEAR`
- percentages as `NumericUnit.PERCENT`
- scaled numbers with million/billion/trillion/thousand/hundred
- money markers: dollar/dollars/usd
- distance markers: km/miles
- weight markers: kg/lbs
- counts: employees/people/users

Years get exact tolerance bounds.

### Truth Priors

Configured in `CLAIM_TYPE_TRUTH_PRIORS`:

- date: `0.52`
- membership: `0.65`
- attribute: `0.50`
- authorship: `0.68`
- role: `0.60`
- location: `0.72`
- causal: `0.60`
- comparative: `0.55`
- existence: `0.74`
- general: `0.60`

Current scoring uses priors only as a mild floor condition: if prior `>= 0.72` and the claim has support/strong support, score is at least `0.8`.

## Retrieval

`RetrievalAgent.retrieve()`:

1. Checks semantic cache by contextualized claim text if enabled.
2. Gets subject queries from `RelationalAnalysis`.
3. For each retrieval attempt:
   - sanitizes query.
   - runs Wikipedia full-text search if KB search is enabled.
   - on first attempt, fetches direct subject Wikipedia article if subject exists.
   - runs web search if enabled and API keys exist.
4. Deduplicates snippets by URL.
5. Filters short snippets.
6. Stops early after at least 3 snippets.
7. Scores each snippet for:
   - subject match
   - object match
   - relation match
   - claim type
8. Applies evidence quality filter.
9. Reranks snippets.
10. Keeps final top 3 evidence snippets.
11. Writes result to cache if enabled.

If the normal subject-first attempts return no snippets, retrieval performs one fallback search from meaningful full-claim entities/key terms before giving up.

### Wikipedia Retrieval

Direct subject article:

- calls MediaWiki API with `titles=subject`.
- gets intro extract, plaintext, up to 12 sentences.
- returns Tier 1 evidence.
- sets `subject_match_score = 1.0`.

Wikipedia full-text:

- searches up to `kb_search_results_per_query`.
- fetches intro extracts for page IDs.
- returns Tier 1 evidence.

### Web Retrieval

Uses Tavily if `AFVE_TAVILY_API_KEY` exists.

Uses SerpAPI if `AFVE_SERP_API_KEY` exists.

Otherwise web search returns no snippets.

### Domain Credibility

`DOMAIN_TIER_MAP` includes:

Tier 1:

- Wikipedia, Britannica, Reuters, AP, BBC, Nature, Science, NEJM, PubMed, Scholar, `.gov`, `.edu`, Wikidata, arXiv, WHO, CDC, NIH, UN.

Tier 2:

- NYT, Guardian, Washington Post, Forbes, TechCrunch, Wired, Medium, StackOverflow, GitHub, Economist, FT, WSJ, Bloomberg, Politico, The Atlantic, Vox, Statista, Our World in Data.

Everything else falls to Tier 3.

Credibility multipliers:

- Tier 1: `1.00`
- Tier 2: `0.75`
- Tier 3: `0.45`

### Subject Match Score

`_score_presence(text, entity)`:

- exact entity substring -> `1.0`
- otherwise checks how many non-stopword entity tokens appear.
- multiplied by `min(1.0, len(entity) / 6)`.
- empty entity -> `1.0`.

### Predicate/Relation Match Score

`_score_predicate(text, predicate, claim_type)`:

- exact predicate substring -> `1.0`
- two or more claim-type synonyms in text -> `1.0`
- one synonym -> `0.70`
- otherwise -> `0.20`

### Evidence Quality Filter

`_passes_evidence_quality_filter()`:

- subject match ratio must be at least `0.5`.
- evidence must overlap claim key terms.
- rejects generic disambiguation phrases like `may refer to`.
- rejects dictionary-style snippets such as `X is a word`, `X is a conjunction`, `definition of`, and similar low-evidence entries.

If filtering produces any snippets, filtered snippets replace the original pool.

### Reranking

If `enable_reranker` is true, uses `BAAI/bge-reranker-base` CrossEncoder if loadable.

Reranker blend:

`0.60 * reranker_score + 0.28 * subject_match_score + 0.12 * relation_match_score`

If reranker fails or is disabled, sorts by `subject_match_score`.

## NLI Verification Pipeline

`VerificationEngine.verify()` is the active claim/evidence verifier.

Important current reality:

- The docstring says no LLM usage happens here.
- `self._llm = None`.
- `_run_llm_judge()` is a deterministic stub returning `0.0`.
- `llm_judge_component` is therefore always `0.0` in normal verification.
- `enable_llm_judge` exists in config, but this engine does not currently use it.

### Evidence Sentence Selection

For each retrieved snippet:

1. Split snippet into sentences.
2. Drop generic sentences:
   - too short.
   - definition-like openings.
   - `may refer to`.
   - `is essential/important/vital`.
3. Candidate sentence must:
   - match subject strongly enough.
   - overlap claim key terms.

Relaxed candidate thresholds are used for:

- explanation mode.
- science/physics/biology domains.

Candidate alignment score:

`0.45 * keyword_overlap + 0.30 * subject_score + 0.25 * predicate_score + numeric_bonus + relaxed_bonus - generic_penalty`

Where:

- numeric bonus: `+0.20` if qualifier terms appear.
- relaxed bonus: `+0.10` if relaxed mode and overlap `>= 0.35`.
- generic penalty: `-0.25` for generic `is a/the` sentence with weak predicate.

Top 5 aligned sentence candidates are scored. Final evidence is sorted and trimmed to top 3.

### Hypothesis Construction

NLI hypothesis is built from the triple:

`subject predicate object qualifier`

If no subject/predicate exists, falls back to full claim text.

Example:

`Apple founded 1976`

### NLI Model

Default model:

`cross-encoder/nli-deberta-v3-base`

Loaded as a Hug Face text-classification pipeline.

Input order:

- premise: evidence sentence
- hypothesis: claim hypothesis

The code normalizes `LABEL_0`, `LABEL_1`, etc. through `model.config.id2label` when available.

### NLI Status Mapping

Inside `_score_snippet_sync()`:

- Runs numeric comparison first.
- If numeric result is `CONTRADICTS`, status is forced to `CONTRADICTS`, and NLI score is capped to `<= 0.05`.
- If numeric result is `MATCHES` or `CLOSE` and subject match `>= 0.70`, status is forced to `SUPPORTS`, and score is raised to at least `0.88`.
- If semantic score `< 0.35` or subject match `< 0.5`, status becomes `NEUTRAL`.
- Otherwise, entailment labels become `SUPPORTS` only if model confidence passes threshold.
- Contradiction labels become `CONTRADICTS` only if there is explicit negation or clear numeric/year mismatch.
- Otherwise, if subject match `>= 0.75` and concept match is strong, status becomes `WEAK_SUPPORT`, score at least `0.78`.
- Else status is `NEUTRAL`.

Decision threshold:

- `0.65` if semantic score `>= 0.60`.
- otherwise `0.85`.

Strong concept match:

- relation match `>= 0.65`, or
- semantic score `>= 0.55`, or
- keyword overlap `>= 0.45`.

Contradiction guardrails:

- Low similarity is treated as irrelevant/neutral, not contradiction.
- Model contradiction is trusted only with explicit negation or numeric/year mismatch.

### NLI Fallbacks

If NLI pipeline cannot load:

- numeric match/close -> `SUPPORTS` with score `0.88`.
- numeric contradiction -> `CONTRADICTS` with score `0.05`.
- otherwise -> `NEUTRAL` with score `0.5`.

If retrieval produces no real snippets, verification returns no evidence. It no longer fabricates a placeholder evidence snippet.

There is also `_heuristic_nli()`, but the primary active path relies mostly on `_score_snippet_sync()` fallback behavior.

## Numeric Comparison

There are two numeric comparison paths.

### NumericComparator Class

`NumericComparator.compare(evidence, assertions)` compares decomposer-created `NumericAssertion` values against extracted numbers from evidence.

It supports:

- million/billion/trillion/thousand/hundred multipliers.
- percent.
- km/miles.
- kg/lbs.
- employees/people/users.
- dollars/usd.

Result priority:

1. `MATCHES`
2. `CLOSE`
3. `DIVERGES`
4. `CONTRADICTS`
5. `NOT_FOUND`

### Active Text-to-Text Numeric Comparison

The active NLI path calls `_compare_numeric_texts(claim_hypothesis, evidence_sentence)`.

It:

1. extracts numeric quantities from claim and evidence.
2. normalizes units through aliases.
3. compares only compatible units.
4. applies configured tolerances.

Numeric tolerances:

- year: exact.
- count: `10%` relative or `1000` absolute.
- percent: `2` percentage points absolute.
- money: `15%` relative.
- rank: exact.
- distance: `5%` relative.
- weight: `5%` relative.
- none: `10%` relative.

Numeric pair result:

- exact relative error `0.0` -> `MATCHES`
- within normal tolerance -> `CLOSE`
- within roughly 3x tolerance or `0.12` relative -> `DIVERGES`
- otherwise -> `CONTRADICTS`

### Numeric Score Component

`_num_to_score()`:

- `MATCHES`: `1.0`
- `CLOSE`: `0.80`
- `DIVERGES`: `0.45`
- `CONTRADICTS`: `0.0`
- `NOT_FOUND`: `0.5`

Note: `NUMERIC_SCORE_WEIGHT` exists in config but is currently `0.0` and not part of the active `ScoringCalibrator.compute_claim_score()` formula. Numeric still affects entailment hard overrides.

## Unit Validation

After scoring evidence, `_validate_unit_consistency()` checks strict unit consistency for speed, temperature, distance, and weight.

Supported normalizations include:

- mph, miles/hour, miles per hour -> m/s
- m/s, meters per second -> m/s
- km/s -> m/s
- Celsius/Fahrenheit/Kelvin -> Celsius
- km/m/meters/miles -> meters
- kg/g/lb/lbs -> kg

If claim and evidence quantities of same family do not match within `5%`, a validation error is added:

- `UNIT ERROR: value corresponds to ...`
- or `UNIT ERROR: ... does not match the referenced value within tolerance`

Any validation error caps score to `0.3` and label becomes `CONTRADICTED`.

## Temporal Validation

Temporal requirement is inferred from the prompt:

- before-patterns -> `before`
- after-patterns -> `after`
- equality-patterns or exactly one prompt year -> `equal`

Current active strict validation mainly handles equality:

- if prompt asks for a specific year and claim year differs, add `TEMPORAL ERROR`.
- if claim year differs from evidence reference year, add `TEMPORAL ERROR`.

Any temporal validation error caps score to `0.3` and label becomes `CONTRADICTED`.

### Cross-Claim Temporal Validation

After all claims are verified:

- If prompt implies equality and multiple verified claims contain years,
- and claim years disagree,
- all year claims are marked `CONTRADICTED`,
- each score is capped to `0.3`,
- explanation is prepended with `TEMPORAL ERROR: events occur in different years`.

## Entity Existence Heuristics

The pipeline tries to handle existence/non-existence claims.

Polarity detection:

- negative: `there is no`, `does not exist`, `no such`, `never existed`, etc.
- positive: `there is a`, `there are`, `exists`, `is real`.
- otherwise neutral.

Target entity extraction:

- regexes for `there is no X`, `X does not exist`, etc.
- otherwise triple object/subject fallback.

Impossible entity heuristics:

- political role assigned to a celestial body.
- fictional setting treated as a real-world institution.
- historical figure paired with modern tech invention/creation.

Credible negative evidence:

- Tier 1 evidence mentioning target and phrases like `does not exist`, `there is no`, `never existed`, `fictional`.

Existence override:

- non-existent target + negative claim -> score at least `0.95`, label `VERIFIED`.
- non-existent target + positive claim -> score `0.0`, label `CONTRADICTED`.
- unknown existence + existence/general positive or negative claim + no evidence -> score `0.5`, label `UNCERTAIN`.

## Scoring

Despite older config comments about a six-signal composite, the current active `ScoringCalibrator.compute_claim_score()` is rule-based over evidence statuses. It does not currently use the imported linear weights directly.

### Per-Claim Score

Special cases:

- Abstract claim:
  - any contradiction -> `0.0`
  - otherwise -> `0.6`
- No evidence -> `0.2`

For top evidence snippets:

- If contradiction exists and no support/weak support/strong numeric -> `0.0`
- If any support -> `1.0`
- If strong support -> `0.85`
- If weak support -> `0.7`
- If neutral -> `0.5`
- Else -> `0.2`

Strong support means:

- direct support, or
- numeric `MATCHES`/`CLOSE`, or
- weak support with semantic score `>= 0.65` or relation match `>= 0.7`.

Score floors/upgrades:

- conceptual/abstract claims in explanation mode can be floored around `0.62-0.68`.
- science domain with semantic match can be floored around `0.65-0.72`.
- subject validated + relation verified + semantic score `>= 0.45` -> at least `0.75`.
- strong support without contradiction -> at least `0.8`.
- truth prior `>= 0.72` plus support/strong support -> at least `0.8`.

Score is rounded to 4 decimals and clamped to `[0, 1]`.

### Claim Label From Score and Evidence

For claim-level label:

- no evidence -> `UNSUPPORTED`
- contradiction and no support -> `CONTRADICTED`
- score `>= 0.85` -> `VERIFIED`
- score `>= 0.70` -> `MOSTLY_TRUE`
- score `>= 0.50` -> `UNCERTAIN`
- otherwise -> `LIKELY_FALSE`

After entity/temporal/unit overrides, label may change.

### Overall Score

Overall score is weighted average:

`sum(claim.score * claim.claim_weight) / sum(claim.claim_weight)`

No evidence penalty or hallucination penalty is currently applied at overall-score level.

### Overall Label

`ScoringCalibrator.score_to_label()`:

- score `>= 0.85` -> `VERIFIED`
- score `>= 0.70` -> `MOSTLY_TRUE`
- score `>= 0.50` -> `UNCERTAIN`
- score `<= 0.20` -> `UNSUPPORTED`
- otherwise -> `LIKELY_FALSE`

## Evidence Ranking

Claim-level evidence ranking:

`nli_score + semantic_score + relation_match_score + entailment_bonus + credibility_bonus`

Entailment bonus:

- supports: `1.0`
- weak support: `0.8`
- neutral: `0.35`
- contradicts: `0.9`

Credibility bonus:

- Tier 1: `0.3`
- Tier 2: `0.15`
- Tier 3: `0.0`

The orchestrator keeps top 2 claim evidence snippets for final claim display.

Global evidence ranking:

- starts with `nli_score + semantic_score`
- supports/weak support: `+2.0`
- contradiction: `+1.0`
- Tier 1: `+0.5`
- Tier 2: `+0.2`

Top 5 global evidence snippets are returned.

## Explanation Generation

Each claim explanation includes:

- claim type and triple:
  - `[DATE] S='Apple' P='founded' O='' Q='1976'`
- numeric finding if present:
  - numeric contradiction/divergence/close/match message.
- validation errors if any.
- support/contradiction counts.
- high-authority source count.
- subject match count.
- relation match count.
- a key evidence quote truncated to roughly 200 chars.

## Report Metadata and Uncertainty Signals

`VerificationReport` includes:

- processing time.
- number of sources queried.
- claims extracted.
- atoms generated.
- claims verified.
- retrieval failures.
- no-evidence claims.
- relation-unverified count.
- verification mode.

Model validator populates uncertainty signals:

- contradicted-by-label count.
- uncertain claim count.
- no-evidence claim count.
- relation-unverified count.
- numeric contradiction count.
- minimum claim score.
- score variance.
- contradicted-by-evidence count.
- claim type breakdown.
- factuality percentage and rating.

Rating thresholds:

- `>= 85%`: Highly Reliable
- `>= 70%`: Mostly True
- `>= 50%`: Acceptable / Mixed
- `<= 20%`: Insufficient Evidence
- otherwise: Likely Incorrect

## Metrics

`main.py` tracks:

- total requests.
- total claims.
- total hallucinated/contradicted claims.
- total no-evidence claims.
- total relation-verified claims.
- total numeric-contradiction claims.
- total processing time.

`GET /metrics` exposes rates:

- hallucination rate.
- no-evidence rate.
- contradiction rate.
- relation verification rate.
- numeric contradiction rate.
- average latency.

## Caching

`SemanticCache` uses Redis if available.

Key:

`afve:claim:{sha256(claim_text)}`

Value:

- JSON-dumped `EvidenceSnippet` list.

TTL:

- `cache_ttl_seconds`, default `86400`.

Despite the name, the current cache key is exact hash of claim text, not semantic nearest-neighbor lookup. `cache_embedding_model` and `cache_similarity_threshold` exist in settings but are not active in the shown cache implementation.

## LLM Usage

There is an `LLMClient` that calls local Ollama `/api/generate`.

Used for:

- optional coreference resolution if `enable_coreference_resolution` is true.
- optional LLM triple extraction if `enable_llm_triple_extraction` is true.
- optional query expansion if `enable_query_expansion` is true.

Not currently active in verification:

- `VerificationEngine` disables LLM judge.
- `_run_llm_judge()` returns `0.0`.
- `enable_llm_judge` exists but does not affect current verification results.

If `settings.llm_judge_model` contains `claude` or `gpt`, `LLMClient` substitutes `qwen2.5-coder:7b` for local Ollama.

## Known Design Mismatches and Improvement Targets

These are important for another AI trying to improve the pipeline:

- The config comments describe a six-signal weighted composite, but the active scorer is rule-based and does not use `NLI_WEIGHT`, `LLM_JUDGE_WEIGHT`, `SOURCE_CREDIBILITY_WEIGHT`, `CROSS_SOURCE_AGREEMENT_WEIGHT`, `RELATION_VERIFICATION_WEIGHT`, or `NUMERIC_SCORE_WEIGHT` directly.
- `LLM_JUDGE_WEIGHT` is `0.30` in config, but the LLM judge is disabled in the active engine.
- `enable_llm_judge` is currently misleading unless the engine is changed to actually call a judge.
- `RELATION_VERIFICATION_WEIGHT` and `NUMERIC_SCORE_WEIGHT` default to `0.0`; relation/numeric signals still affect rule-based scores through status overrides and floors.
- `strictness` is exposed but not deeply integrated into thresholds.
- `enable_relational_decomposition` is exposed but not respected as a full off switch in extraction.
- Semantic cache is exact text hash, not semantic similarity.
- Retrieval is heavily Wikipedia-biased unless Tavily/SerpAPI keys are configured.
- Triple extraction is regex-heavy and can fail on lowercase entities, non-English names, app/product names, and claims with implicit subjects.
- NLI contradiction is deliberately conservative; it may miss contradictions without explicit negation or numeric mismatch.
- Numeric comparison chooses the best matching numeric pair by priority, which can over-credit evidence containing many unrelated numbers.
- The direct subject article assumption is good for entities but weaker for claims about events, processes, abstract concepts, or multi-hop relations.
- Comparative claims still need true dual-subject retrieval and numeric comparison across both entities; current fixes only prevent obvious bad anchors such as sentence-initial conjunctions.
- Cross-claim temporal validation currently handles equality-style prompts, not full before/after reasoning.
- Unit validation handles some physical units but is incomplete for finance, dates/ranges, rates, and compound units.
- Claim extraction caps at 6 claims, which may under-verify long answers.
- Evidence finalization keeps only 3 snippets before verification and 2 after ranking, which can lose contradictory evidence.

## Recommended Improvement Directions

Potential high-value work:

- Make scoring consistent: either implement the six-signal weighted calibrator or update config/docs to match the rule-based scorer.
- Re-enable an explicit LLM judge as a separate, auditable component that outputs status, confidence, and rationale in JSON.
- Add calibrated contradiction detection with better contradiction datasets or pairwise evidence comparison.
- Use structured numeric assertions directly in active numeric comparison, not only text-to-text quantity extraction.
- Add semantic cache with embeddings if keeping the name `SemanticCache`.
- Respect `strictness` by shifting thresholds for NLI, subject match, relation match, and label cutoffs.
- Improve triple extraction with dependency parsing or a constrained LLM extraction step.
- Add source diversity: Wikidata, DBpedia, domain APIs, news APIs, PubMed/arXiv for scientific claims.
- Add date range and temporal relation reasoning for before/after/during claims.
- Improve evidence selection to preserve both strongest support and strongest contradiction.
- Add per-claim debug traces showing raw snippets, filtered snippets, reranked snippets, candidates, NLI inputs, NLI raw labels, numeric pairs, and final overrides.
- Add calibration tests with known true/false/unsupported claims across claim types.
