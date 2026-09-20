"""Endpoint tests for the Django front end. Stubbed runtime, so no weights, no torch:

    python src/serve_django/test_django.py

These mirror src/serve/test_app.py deliberately. Two front ends over one cascade are only
useful if they behave identically, so the same assertions are made against both -- the same
validation, the same response keys, the same 4dp rounding on escalated_pct. If one drifts,
one of these suites goes red.

Django-specific things checked here that FastAPI gets for free: that GET is rejected on a
POST endpoint, that CSRF does not block an API POST, and that the lazy-loading contract in
runtime.py actually holds (a request builds the runtime; /health before that says so).
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cascade_site.settings")
os.environ.setdefault("CASCADE_DATA", str(HERE.parent.parent / "data"))

import django                                                    # noqa: E402
django.setup()

from django.test import Client                                   # noqa: E402

from classifier import runtime                                   # noqa: E402

FAILURES = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILURES.append(name)


class StubRuntime:
    def __init__(self):
        self.cascade = SimpleNamespace(
            device="cpu", fp16=False, retriever=None, leaves=["L0", "L1"],
            cfg=SimpleNamespace(tau=0.9, k=50, shortlist=10, shortlist_mode="fused",
                                s1_tag="head69", s2_tag="ft2"))
        self.stage3 = SimpleNamespace(cfg=SimpleNamespace(provider="cached"))
        self.seen = []

    def classify(self, item, item_id=None):
        self.seen.append((item, item_id))
        return {"leaf": "L0", "stage": "S1" if "confident" in item["title"] else "S3",
                "s1_confidence": 0.99, "llm_confidence": None, "text": item["title"],
                "shortlist": ["L0", "L1"], "trace": [{"node": "s1", "ms": 4.2}],
                "total_ms": 4.2}


def main():
    c = Client()

    print("\nlazy loading")
    runtime.reset_for_tests(None)
    h = c.get("/health").json()
    check("health reports not-loaded before any request", h["status"] == "not loaded")

    rt = StubRuntime()
    runtime.reset_for_tests(rt)

    print("\nhealth")
    h = c.get("/health").json()
    check("reports ok once loaded", h["status"] == "ok")
    check("names the framework", h["framework"] == "django")
    check("exposes retriever warmth separately", h["retriever_warm"] is False)
    check("echoes the operating point", h["config"]["tau"] == 0.9)

    print("\n/classify")
    r = c.post("/classify", data={"title": "confident widget"},
               content_type="application/json")
    check("200 on a minimal body", r.status_code == 200)
    check("returns the tier that decided", r.json()["stage"] == "S1")
    check("returns the per-node trace", r.json()["trace"][0]["node"] == "s1")
    check("item_id is not forwarded as a model field", "item_id" not in rt.seen[-1][0])
    c.post("/classify", data={"title": "x", "item_id": "B01"}, content_type="application/json")
    check("item_id is forwarded separately", rt.seen[-1][1] == "B01")
    check("CSRF does not block an API POST", r.status_code == 200)
    check("GET is rejected", c.get("/classify").status_code == 405)
    check("empty title is rejected",
          c.post("/classify", data={"title": ""}, content_type="application/json"
                 ).status_code == 422)
    check("missing title is rejected",
          c.post("/classify", data={"brand": "Acme"}, content_type="application/json"
                 ).status_code == 422)
    check("non-string brand is rejected",
          c.post("/classify", data={"title": "x", "brand": 7}, content_type="application/json"
                 ).status_code == 422)
    check("malformed JSON is rejected",
          c.post("/classify", data="{not json", content_type="application/json"
                 ).status_code == 400)

    print("\n/classify/batch")
    body = {"items": [{"title": "confident a"}, {"title": "b"}, {"title": "c"}]}
    b = c.post("/classify/batch", data=body, content_type="application/json").json()
    check("returns one result per item", len(b["results"]) == 3)
    check("counts stages", b["summary"]["stage_counts"] == {"S1": 1, "S3": 2})
    check("escalation share excludes S1, rounded to 4dp",
          b["summary"]["escalated_pct"] == round(2 / 3, 4))
    check("empty batch is rejected",
          c.post("/classify/batch", data={"items": []}, content_type="application/json"
                 ).status_code == 422)
    for label, raw in (("list", "[]"), ("null", "null"), ("string", '"x"')):
        check(f"non-object batch body ({label}) is a 422, not a 500",
              c.post("/classify/batch", data=raw, content_type="application/json"
                     ).status_code == 422)
    check("an invalid item names its index",
          "items[1]" in c.post("/classify/batch",
                               data={"items": [{"title": "ok"}, {"title": ""}]},
                               content_type="application/json").json()["detail"])

    print("\nroot")
    check("/ lists the endpoints", "/health" in c.get("/").json()["endpoints"])

    print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILURES: {FAILURES}"))
    sys.exit(0 if not FAILURES else 1)


if __name__ == "__main__":
    main()
