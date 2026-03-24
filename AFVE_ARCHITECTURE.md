# AFVE: Relational Fact Verification Engine - Architecture & Design

This document details the architecture, data flow, and core components of the AFVE (Relational Fact Verification Engine) system, with a specific focus on its Natural Language Inference (NLI) and explanation generation capabilities.

## 1. System Overview and Data Flow

The AFVE pipeline processes verification tasks linearly through several distinct agent components, orchestrated by the `VerificationOrchestrator` (located in `agents.py`).

The overall flow of data is as follows:

1.  **Ingestion & Orchestration (`main.py` -> `agents.py`)**: A `VerificationRequest`, typically containing a user prompt and model-generated text, is passed to the `VerificationOrchestrator`.
2.  **Claim Extraction & Decomposition (`ClaimExtractionAgent`)**: The raw model output is split into atomic, verifiable sentences using an LLM. Through the `RelationalDecomposer`, each `AtomicClaim` is strictly structured into a `RelationalTriple`. This triple consists of a subject, predicate, object, and an optional qualifier (e.g., *Subject="Elon Musk"*, *Predicate="co-founded"*, *Object="Tesla"*, *Qualifier="in 2003"*).
3.  **Retrieval (`RetrievalAgent`)**: For each atomic claim, a search is performed (against the web or a knowledge base) anchored on the **subject** of the parsed triple. Retrieved documents are cached using Redis (`SemanticCache`) to optimize repeated lookups and parsed into `EvidenceSnippet` objects, which are prioritized by source credibility.
4.  **Fact Verification (`VerificationEngine` & NLI)**: Each retrieved `EvidenceSnippet` is evaluated against the source claim utilizing multiple distinct comparison mechanisms running in parallel (detailed in the NLI section below).
5.  **Score Calibration (`ScoringCalibrator`)**: Evaluates all annotated snippets per claim. The score uses a mix of the verification engine outputs, weighting factors tied to source credibility tiers (e.g., highly credible domains map to Tier 1), and inter-source agreement metrics to calculate a final Bayesian estimation and cast a qualitative `VerificationLabel` (e.g., `HIGHLY_FACTUAL`, `UNCERTAIN`, `LIKELY_HALLUCINATED`).
6.  **Explanation Generation (`ExplanationGenerator`)**: Produces a readable justification text summarizing exactly *why* a discrete label was given (detailed below).
7.  **Report Compilation**: The elements are packed back into a `VerificationReport` providing rich debugging analytics, signals of uncertainty, and performance metrics (e.g., processing latency).

---

## 2. Natural Language Inference (NLI) Pipeline

Within the `VerificationEngine._score_snippet` method, verifying a claim leverages **three concurrent components**, with the NLI being the focal point of linguistic comparison:

### 2.1 Primary NLI Pipeline (`_run_nli`)
The system constructs a plain-text hypothesis string dynamically from the `RelationalTriple` (e.g., *"{subject} {predicate} {object} {qualifier}"*). It uses a Hugging Face `transformers` pipeline initialized for `text-classification` (usually a cross-encoder model like DeBERTa) to calculate an entailment probability score, labeling the text as `SUPPORTS`, `CONTRADICTS`, or `NEUTRAL`.

### 2.2 Fallback Heuristic NLI (`_heuristic_nli`)
If the primary transformer pipeline is unavailable, the system fails over to a robust TF-IDF similarity technique. It normalizes both the hypothesis and the evidence string, measures weighted token overlap, and includes a check for nearby negation words to approximate inference probabilistically.

### 2.3 LLM Judge (`_run_llm_judge`) & Numeric Comparator
While the primary NLI focuses on textual syntax and semantics, an auxiliary LLM Judge attempts semantic context-entailment. Simultaneously, a `NumericComparator` extracts numbers to ensure values aren't mathematically out of bounds, protecting against NLI models that may struggle with mathematical reasoning.

### 2.4 Merging Operations (`_merge_entailment`)
The entailment evaluations represent a pessimistic aggregate. If the `NumericComparator` flags a number constraint violation, or if *either* the LLM Judge or NLI engine triggers a `CONTRADICTS` status, the entire snippet resolves to `CONTRADICTS`.

---

## 3. Explanation Generation (`ExplanationGenerator`)

Located in `agents.py`, the `ExplanationGenerator` acts dynamically on the back-end to serialize technical verification artifacts into cohesive, human-readable text intended for UI display.

Its `generate()` method constructs deterministic, template-based summaries leveraging multiple facts:

*   **Triple Highlighting**: It reveals the internal model translation, exposing how the claim was parsed (e.g., outputting `[DATE] S='Apple' P='founded' O='-' Q='1976'`).
*   **Evidence Mathing**: It computes and prints counts for how many sources supported or contradicted the claim, along with mentions of authoritative tiering (e.g., `"Verified by 3 source(s) incl. 1 high-authority"`).
*   **Match Rates**: It highlights the specific confidence of whether the subject and relations mathematically surfaced above the requisite thresholds.
*   **Numeric Callouts**: If the `NumericComparator` flagged anomalies, it safely appends warnings like `"NUMERIC CONTRADICTION: value in evidence conflicts with claimed number."`
*   **Quote Bubbles**: Assuming evidence was found, it extracts a key portion (e.g., ~200 characters) from the most decisive `EvidenceSnippet` string to visually prove the evaluation.
