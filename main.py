"""
main.py — FastAPI entry point for AFVE v1.3 (Relational Fact Verification Engine).

Changes from previous revision
───────────────────────────────
  • build_orchestrator wires RelationalDecomposer into ClaimExtractionAgent.
  • _update_metrics tracks relation_verified_claims and numeric_contradiction_claims.
  • /metrics exposes relation_verification_rate and numeric_contradiction_rate.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agents import (
    ClaimExtractionAgent,
    ExplanationGenerator,
    LLMClient,
    RelationalDecomposer,
    RetrievalAgent,
    ScoringCalibrator,
    SemanticCache,
    VerificationEngine,
    VerificationOrchestrator,
)
from config import get_settings
from models import (
    ComparisonResult,
    EntailmentStatus,
    JobStatus,
    JobStatusResponse,
    MetricsSummary,
    OllamaRunRequest,
    OllamaRunResponse,
    VerificationLabel,
    VerificationReport,
    VerificationRequest,
)

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("afve.main")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_metrics: dict[str, float | int] = {
    "total_jobs_processed": 0,
    "total_latency_ms": 0,
    "total_claims": 0,
    "total_hallucinated_claims": 0,
    "total_no_evidence_claims": 0,
    "total_contradicted_claims": 0,
    "total_relation_verified_claims": 0,
    "total_numeric_contradiction_claims": 0,
    "cache_hits": 0,
    "cache_requests": 0,
}

_job_store: dict[str, JobStatusResponse] = {}
_job_store_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    http_client = httpx.AsyncClient(
        timeout=settings.http_timeout_seconds,
        headers={"User-Agent": settings.wikipedia_user_agent},
        follow_redirects=True,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    app.state.http_client = http_client
    app.state.llm_client  = LLMClient()
    app.state.cache = (
        SemanticCache(settings.redis_url) if settings.enable_semantic_cache else None
    )
    logger.info("AFVE startup complete.")
    yield
    await http_client.aclose()
    logger.info("AFVE shutdown complete.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Agentic Factual Verification Engine — relational triple decomposition, "
        "subject-first retrieval, numeric comparison, and relation-aware scoring."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


def get_http_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_cache(request: Request) -> SemanticCache | None:
    return getattr(request.app.state, "cache", None)


def build_orchestrator(
    http_client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    llm_client: Annotated[LLMClient, Depends(get_llm_client)],
    cache: Annotated[SemanticCache | None, Depends(get_cache)],
) -> VerificationOrchestrator:
    decomposer = RelationalDecomposer(llm_client=llm_client)
    return VerificationOrchestrator(
        extractor=ClaimExtractionAgent(llm_client=llm_client, decomposer=decomposer),
        retriever=RetrievalAgent(http_client=http_client, llm_client=llm_client, cache=cache),
        verifier=VerificationEngine(llm_client=llm_client),
        calibrator=ScoringCalibrator(),
        explainer=ExplanationGenerator(),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/verify",
    response_model=VerificationReport,
    summary="Synchronous relational fact verification",
    status_code=status.HTTP_200_OK,
)
async def verify_sync(
    payload: VerificationRequest,
    orchestrator: Annotated[VerificationOrchestrator, Depends(build_orchestrator)],
) -> VerificationReport:
    logger.info("[%s] Sync verification started.", payload.job_id)
    start  = time.monotonic()
    report = await orchestrator.run(payload)
    elapsed = int((time.monotonic() - start) * 1000)
    _update_metrics(report, elapsed)
    logger.info(
        "[%s] Done %dms score=%.3f label=%s atoms=%d",
        payload.job_id, elapsed, report.overall_score,
        report.overall_label, len(report.claims),
    )
    return report


async def _ollama_generate(
    http_client: httpx.AsyncClient,
    model: str,
    prompt: str,
) -> str:
    """Generate a completion from a local Ollama server."""

    base = settings.ollama_base_url.rstrip("/")
    url = f"{base}/api/generate"
    try:
        resp = await http_client.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ollama is not reachable at {base} ({exc}).",
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ollama error: {exc.response.status_code}",
        ) from exc

    text = (data or {}).get("response", "")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Ollama returned an empty response.",
        )
    return text


@app.post(
    "/run",
    response_model=OllamaRunResponse,
    summary="Run an Ollama model then verify with AFVE",
    status_code=status.HTTP_200_OK,
)
async def run_ollama_then_verify(
    payload: OllamaRunRequest,
    http_client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    orchestrator: Annotated[VerificationOrchestrator, Depends(build_orchestrator)],
) -> OllamaRunResponse:
    job_id = payload.job_id or uuid.uuid4()

    start = time.monotonic()
    model_output = await _ollama_generate(http_client, payload.model, payload.prompt)

    req = VerificationRequest(
        job_id=job_id,
        prompt=payload.prompt,
        model_output=model_output,
        config=payload.config,
    )
    report = await orchestrator.run(req)
    elapsed = int((time.monotonic() - start) * 1000)
    _update_metrics(report, elapsed)

    return OllamaRunResponse(
        job_id=job_id,
        model=payload.model,
        prompt=payload.prompt,
        model_output=model_output,
        report=report,
    )


@app.post(
    "/verify/async",
    response_model=JobStatusResponse,
    summary="Async job submission",
    status_code=status.HTTP_202_ACCEPTED,
)
async def verify_async(
    payload: VerificationRequest,
    background_tasks: BackgroundTasks,
    orchestrator: Annotated[VerificationOrchestrator, Depends(build_orchestrator)],
) -> JobStatusResponse:
    job_id = payload.job_id
    async with _job_store_lock:
        _job_store[str(job_id)] = JobStatusResponse(
            job_id=job_id, status=JobStatus.QUEUED
        )
    background_tasks.add_task(_run_bg, str(job_id), payload, orchestrator)
    return _job_store[str(job_id)]


@app.get(
    "/verify/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll async job status",
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    resp = _job_store.get(job_id)
    if resp is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return resp


@app.get("/health", summary="Liveness probe")
async def health() -> dict:
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}


@app.get("/metrics", response_model=MetricsSummary, summary="Operational metrics")
async def metrics() -> MetricsSummary:
    total   = int(_metrics["total_jobs_processed"])
    total_c = int(_metrics["total_claims"])
    return MetricsSummary(
        total_jobs_processed=total,
        average_latency_ms=round(
            float(_metrics["total_latency_ms"]) / max(total, 1), 2
        ),
        hallucination_rate=round(
            int(_metrics["total_hallucinated_claims"]) / max(total_c, 1), 4
        ),
        cache_hit_rate=round(
            int(_metrics["cache_hits"]) / max(int(_metrics["cache_requests"]), 1), 4
        ),
        no_evidence_rate=round(
            int(_metrics["total_no_evidence_claims"]) / max(total_c, 1), 4
        ),
        contradiction_rate=round(
            int(_metrics["total_contradicted_claims"]) / max(total_c, 1), 4
        ),
        relation_verification_rate=round(
            int(_metrics["total_relation_verified_claims"]) / max(total_c, 1), 4
        ),
        numeric_contradiction_rate=round(
            int(_metrics["total_numeric_contradiction_claims"]) / max(total_c, 1), 4
        ),
    )


# ---------------------------------------------------------------------------
# Background runner
# ---------------------------------------------------------------------------


async def _run_bg(
    job_id: str,
    payload: VerificationRequest,
    orchestrator: VerificationOrchestrator,
) -> None:
    async with _job_store_lock:
        _job_store[job_id] = JobStatusResponse(
            job_id=payload.job_id, status=JobStatus.PROCESSING
        )
    start = time.monotonic()
    try:
        report  = await orchestrator.run(payload)
        elapsed = int((time.monotonic() - start) * 1000)
        _update_metrics(report, elapsed)
        async with _job_store_lock:
            _job_store[job_id] = JobStatusResponse(
                job_id=payload.job_id, status=JobStatus.COMPLETED, result=report
            )
        logger.info("[%s] Background job done in %dms.", job_id, elapsed)
    except Exception as exc:
        logger.error("[%s] Background job failed: %s", job_id, exc, exc_info=True)
        async with _job_store_lock:
            _job_store[job_id] = JobStatusResponse(
                job_id=payload.job_id, status=JobStatus.FAILED, error=str(exc)
            )


# ---------------------------------------------------------------------------
# Metrics update
# ---------------------------------------------------------------------------


def _update_metrics(report: VerificationReport, elapsed_ms: int) -> None:
    _metrics["total_jobs_processed"] = int(_metrics["total_jobs_processed"]) + 1
    _metrics["total_latency_ms"]     = float(_metrics["total_latency_ms"]) + elapsed_ms
    _metrics["total_claims"]         = int(_metrics["total_claims"]) + len(report.claims)

    _metrics["total_hallucinated_claims"] = (
        int(_metrics["total_hallucinated_claims"])
        + sum(1 for c in report.claims if c.label == VerificationLabel.LIKELY_HALLUCINATED)
    )
    _metrics["total_no_evidence_claims"] = (
        int(_metrics["total_no_evidence_claims"])
        + sum(1 for c in report.claims if not c.has_evidence)
    )
    _metrics["total_contradicted_claims"] = (
        int(_metrics["total_contradicted_claims"])
        + sum(
            1 for c in report.claims
            if any(e.entailment_status == EntailmentStatus.CONTRADICTS for e in c.evidence)
        )
    )
    _metrics["total_relation_verified_claims"] = (
        int(_metrics["total_relation_verified_claims"])
        + sum(1 for c in report.claims if c.relation_verified)
    )
    _metrics["total_numeric_contradiction_claims"] = (
        int(_metrics["total_numeric_contradiction_claims"])
        + sum(
            1 for c in report.claims
            if c.numeric_comparison == ComparisonResult.CONTRADICTS
        )
    )


# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Check server logs for details."},
    )


from models import OllamaGenerateRequest, OllamaGenerateResponse

@app.post(
    "/generate",
    response_model=OllamaGenerateResponse,
    summary="Generate text using Ollama without verification",
)
async def generate_ollama_text(
    payload: OllamaGenerateRequest,
    http_client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
) -> OllamaGenerateResponse:
    model_output = await _ollama_generate(http_client, payload.model, payload.prompt)
    return OllamaGenerateResponse(
        model=payload.model,
        prompt=payload.prompt,
        model_output=model_output
    )
