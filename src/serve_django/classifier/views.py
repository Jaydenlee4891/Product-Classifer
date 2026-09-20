"""Three JSON endpoints. The contract is identical to src/serve/app.py on purpose.

Same request shape, same response keys, same rounding -- so the two front ends are
interchangeable and a client cannot tell which is serving. That is the point of the
exercise: the framework is a delivery mechanism, and the cascade does not know about it.

Validation is hand-written rather than DRF. These are three endpoints with no models and
no auth, so serializers would be ceremony -- but the checks are shaped the way a serializer
would be, so swapping one in is mechanical. See README.md.
"""
import json
import time

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from classifier import runtime

MAX_BATCH = 256


def _bad(msg: str, status: int = 400) -> JsonResponse:
    return JsonResponse({"detail": msg}, status=status)


def _parse_item(raw: dict) -> tuple[dict | None, str | None, str | None]:
    """-> (item, item_id, error). item_id travels separately: it selects a cached Stage 3
    answer and is not a model input, so it must never reach the prompt."""
    if not isinstance(raw, dict):
        return None, None, "each item must be an object"
    title = raw.get("title")
    if not isinstance(title, str) or not title.strip():
        return None, None, "title is required and must be a non-empty string"
    for k in ("brand", "bullets"):
        if raw.get(k) is not None and not isinstance(raw[k], str):
            return None, None, f"{k} must be a string"
    item = {"title": title, "brand": raw.get("brand") or "", "bullets": raw.get("bullets") or ""}
    iid = raw.get("item_id")
    if iid is not None and not isinstance(iid, str):
        return None, None, "item_id must be a string"
    return item, iid, None


def root(request):
    return JsonResponse({
        "service": "long-tail cascade classifier (Django)",
        "endpoints": ["/health", "/classify", "/classify/batch"],
        "note": "POST JSON to /classify with at least {\"title\": \"...\"}",
    })


def health(request):
    """`warm` is reported separately from `loaded` because the retriever is lazy even
    once the process has models: 82% of items never reach it, so it is not built until the
    first deferred request, which pays ~10.8s for it."""
    if not runtime.is_loaded():
        return JsonResponse({"status": "not loaded", "detail": "models load on first request"})
    rt = runtime.get_runtime()
    c = rt.cascade
    return JsonResponse({
        "status": "ok",
        "framework": "django",
        "device": c.device,
        "fp16": c.fp16,
        "retriever_warm": c.retriever is not None,
        "provider": rt.stage3.cfg.provider,
        "leaves": len(c.leaves),
        "config": {"tau": c.cfg.tau, "k": c.cfg.k, "shortlist": c.cfg.shortlist,
                   "shortlist_mode": c.cfg.shortlist_mode,
                   "s1_tag": c.cfg.s1_tag, "s2_tag": c.cfg.s2_tag},
    })


@csrf_exempt
@require_http_methods(["POST"])
def classify(request):
    try:
        body = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return _bad("body must be valid JSON")
    item, iid, err = _parse_item(body)
    if err:
        return _bad(err, 422)
    return JsonResponse(runtime.get_runtime().classify(item, item_id=iid))


@csrf_exempt
@require_http_methods(["POST"])
def classify_batch(request):
    """Sequential on purpose. Concurrency here would interleave GPU work across requests
    and make the per-tier timings meaningless, which is the one thing this endpoint is
    for."""
    try:
        body = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return _bad("body must be valid JSON")
    if not isinstance(body, dict):
        return _bad("body must be a JSON object", 422)
    items = body.get("items")
    if not isinstance(items, list) or not items:
        return _bad("items must be a non-empty list", 422)
    if len(items) > MAX_BATCH:
        return _bad(f"items must contain at most {MAX_BATCH} entries", 422)

    parsed = []
    for i, raw in enumerate(items):
        item, iid, err = _parse_item(raw)
        if err:
            return _bad(f"items[{i}]: {err}", 422)
        parsed.append((item, iid))

    rt = runtime.get_runtime()
    out, t0 = [], time.perf_counter()
    for item, iid in parsed:
        out.append(rt.classify(item, item_id=iid))
    by_stage: dict[str, int] = {}
    for r in out:
        by_stage[r["stage"]] = by_stage.get(r["stage"], 0) + 1
    return JsonResponse({
        "results": out,
        "summary": {
            "n": len(out),
            "wall_ms": round((time.perf_counter() - t0) * 1000, 2),
            "stage_counts": by_stage,
            "escalated_pct": round(
                sum(v for k, v in by_stage.items() if k != "S1") / len(out), 4),
        },
    })
