"""HTTP endpoint for the cascade.

    uvicorn serve.app:app --app-dir src --port 8000
    CASCADE_PROVIDER=anthropic uvicorn serve.app:app --app-dir src --port 8000

    curl -s localhost:8000/health | jq
    curl -s localhost:8000/classify -H 'content-type: application/json' \
      -d '{"title":"AmazonBasics Modern Euro Toilet Paper Holder - Satin Nickel"}' | jq

WEIGHTS LOAD ONCE, AT STARTUP, NOT PER REQUEST.
DistilBERT, the cross-encoder and Qwen3-Embedding-0.6B are built in the lifespan handler
and held for the process lifetime. Loading them per request is the standard way this kind
of demo ends up reporting seconds of latency that belong to `from_pretrained`, not to the
model. /health reports `warm` so a latency number taken before the first deferred item --
the retriever is lazy, by design, because 83.5% of items never reach it -- is visibly not
comparable to one taken after.

THE LATENCY NUMBERS ARE PER TIER, AND THAT IS THE POINT.
Every response carries the per-node trace. An end-to-end number averages an S1-only item
(one DistilBERT forward) with an item that went all the way to a synchronous LLM call,
and those differ by orders of magnitude at a mix that tau controls. Reporting the mean
alone would hide exactly the thing the architecture is about.

THE DEFAULT PROVIDER IS 'cached' AND SPENDS NOTHING.
It replays data/stage3_fused.csv, so it answers only items whose item_id is in that file.
Set CASCADE_PROVIDER=anthropic for live calls; that one costs money per request.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DATA = os.environ.get("CASCADE_DATA", "data")
DEVICE = os.environ.get("CASCADE_DEVICE", "auto")
PROVIDER = os.environ.get("CASCADE_PROVIDER", "cached")
# Second-pass agent on Stage 3 abstentions. Off by default: it makes LIVE model
# calls and needs ANTHROPIC_API_KEY. See serve/agent.py.
AGENT = os.environ.get("CASCADE_AGENT", "") == "1"

STATE: dict = {"rt": None, "warm": False, "load_s": None}


class Item(BaseModel):
    title: str = Field(..., min_length=1)
    brand: str = ""
    bullets: str = ""
    item_id: str | None = None


class BatchRequest(BaseModel):
    items: list[Item] = Field(..., min_length=1, max_length=256)


def get_runtime():
    rt = STATE["rt"]
    if rt is None:
        raise HTTPException(503, "models are still loading")
    return rt


@asynccontextmanager
async def lifespan(app: FastAPI):
    from serve.graph import CascadeRuntime
    t0 = time.perf_counter()
    STATE["rt"] = CascadeRuntime.load(DATA, device=DEVICE, provider=PROVIDER, agent=AGENT)
    STATE["load_s"] = round(time.perf_counter() - t0, 2)
    yield
    STATE["rt"] = None


app = FastAPI(title="Long-tail cascade classifier", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def root():
    """There is no model call here -- it exists because a browser pointed at the port
    returns 404 and that reads as a broken server."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    rt = STATE["rt"]
    if rt is None:
        return {"status": "loading"}
    c = rt.cascade
    return {
        "status": "ok",
        "device": c.device,
        "fp16": c.fp16,
        "load_seconds": STATE["load_s"],
        # The retriever is lazy. Until a deferred item has arrived it is not in memory,
        # and the first one to arrive pays for loading it.
        "retriever_warm": c.retriever is not None,
        "provider": rt.stage3.cfg.provider,
        "model": rt.stage3.cfg.model if rt.stage3.cfg.provider in ("anthropic", "raw")
                 else rt.stage3.cfg.ollama_model if rt.stage3.cfg.provider == "ollama"
                 else None,
        "leaves": len(c.leaves),
        "config": {"tau": c.cfg.tau, "k": c.cfg.k, "shortlist": c.cfg.shortlist,
                   "shortlist_mode": c.cfg.shortlist_mode,
                   "s1_tag": c.cfg.s1_tag, "s2_tag": c.cfg.s2_tag},
    }


@app.post("/classify")
def classify(item: Item):
    rt = get_runtime()
    payload = item.model_dump()
    iid = payload.pop("item_id")
    return rt.classify(payload, item_id=iid)


@app.post("/classify/batch")
def classify_batch(req: BatchRequest):
    """Sequential on purpose. Concurrency here would interleave GPU work across requests
    and make the per-tier timings meaningless, which is the one thing this endpoint is
    for. Throughput work belongs in the offline path, which already batches at 32."""
    rt = get_runtime()
    out, t0 = [], time.perf_counter()
    for it in req.items:
        p = it.model_dump()
        out.append(rt.classify(p, item_id=p.pop("item_id")))
    by_stage: dict[str, int] = {}
    for r in out:
        by_stage[r["stage"]] = by_stage.get(r["stage"], 0) + 1
    return {
        "results": out,
        "summary": {
            "n": len(out),
            "wall_ms": round((time.perf_counter() - t0) * 1000, 2),
            "stage_counts": by_stage,
            "escalated_pct": round(
                sum(v for k, v in by_stage.items() if k != "S1") / len(out), 4),
        },
    }
