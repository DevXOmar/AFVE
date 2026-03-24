# AFVE — Scalability & Production Architecture Notes

## Current Architecture (Phase 1)

```
Client → FastAPI (uvicorn, 2 workers) → asyncio pipeline
                                      ↘ Redis (semantic cache)
```

FastAPI's `BackgroundTasks` handles async job submission within the same
process. This is suitable for moderate load (~50–200 RPM) but has a single
point of failure if the process crashes.

---

## Phase 2: Celery + RabbitMQ Worker Queue

For high-throughput or long-running verification jobs (large model outputs,
many claims), decouple the API layer from the compute layer:

```
Client
  │ POST /verify/async
  ▼
FastAPI (API layer — stateless, horizontally scalable)
  │ publish(job_payload)
  ▼
RabbitMQ / Redis (message broker)
  │ consume
  ▼
Celery Workers (N replicas — scale independently)
  │  ┌──────────────────────────────────┐
  │  │  ClaimExtractionAgent            │
  │  │  RetrievalAgent (async I/O)      │
  │  │  VerificationEngine (GPU/CPU)    │
  │  └──────────────────────────────────┘
  │ store result
  ▼
Redis / PostgreSQL (job result store)
  ▲
  │ GET /verify/jobs/{id}
Client
```

### Implementation steps

1. **Add Celery**
   ```python
   # tasks.py
   from celery import Celery
   celery_app = Celery("afve", broker="amqp://rabbitmq:5672//", backend="redis://redis:6379/1")

   @celery_app.task(bind=True, max_retries=3)
   def verify_task(self, payload_dict: dict) -> dict:
       import asyncio
       payload = VerificationRequest(**payload_dict)
       report = asyncio.run(orchestrator.run(payload))
       return report.model_dump()
   ```

2. **Update the async endpoint** to call `.delay()` instead of `BackgroundTasks`.

3. **Scale workers** independently of the API:
   ```bash
   docker compose up --scale afve-worker=8
   ```

4. **Use Flower** for Celery task monitoring dashboard.

---

## Phase 3: Full Kubernetes Deployment

```yaml
# Horizontal Pod Autoscaler on CPU / custom metric (queue depth)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  scaleTargetRef:
    name: afve-worker
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: External
      external:
        metric:
          name: rabbitmq_queue_messages_ready
        target:
          type: AverageValue
          averageValue: "10"
```

Key components:
- **afve-api**: FastAPI deployment (2–4 replicas, low CPU, high I/O)
- **afve-worker**: Celery workers (2–20 replicas, high CPU for NLI inference)
- **redis**: Semantic cache + Celery backend
- **rabbitmq**: Message broker (clustered for HA)
- **prometheus + grafana**: Observability stack

---

## NLI Model Scaling

The DeBERTa NLI model is CPU-bound and the primary scaling bottleneck.
Options ranked by cost/performance:

| Option | Throughput | Cost |
|--------|-----------|------|
| CPU (current) | ~10 claims/s | $ |
| GPU (T4 node) | ~200 claims/s | $$ |
| Triton Inference Server | ~500 claims/s | $$$ |
| Batch inference (offline) | Unlimited | $$ |

For very high throughput, consider replacing DeBERTa with a distilled
MiniLM NLI model (5× faster, ~5% accuracy drop).

---

## Semantic Cache Scaling

Replace SHA-256 key lookup with Redis VSS (Vector Similarity Search):

```python
# Store claim embedding → evidence list
embedding = model.encode(claim)          # float32 [384]
redis.execute_command("HSET", key, "embedding", embedding.tobytes(), "data", json_payload)

# Query
results = redis.execute_command(
    "FT.SEARCH", "idx:claims",
    f"*=>[KNN 1 @embedding $vec AS score]",
    "PARAMS", "2", "vec", query_embedding.tobytes(),
    "SORTBY", "score", "LIMIT", "0", "1"
)
# If score > threshold → cache hit
```

This enables fuzzy matching: "Apple's Q3 revenue" matches
"Apple revenue third quarter" without re-running the full pipeline.

---

## Observability in Production

Replace the mock `/metrics` endpoint with real prometheus_client counters:

```python
from prometheus_client import Counter, Histogram, make_asgi_app

VERIFICATION_LATENCY = Histogram("afve_verification_latency_seconds", "End-to-end latency")
HALLUCINATION_RATE   = Counter("afve_hallucinated_claims_total", "Hallucinated claim count")
CACHE_HITS           = Counter("afve_cache_hits_total", "Semantic cache hits")

# Mount the metrics app
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

Recommended Grafana dashboard panels:
- P50/P95/P99 verification latency
- Hallucination rate over time
- Claims processed per second
- Cache hit rate
- Retrieval failure rate per source (Wikipedia / Tavily / SerpAPI)
- LLM judge error rate
