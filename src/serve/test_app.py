"""Endpoint tests. The runtime is stubbed, so this exercises the HTTP contract only --
validation, the health payload, the batch summary -- and needs no weights:

    python src/serve/test_app.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient                    # noqa: E402

import serve.app as A                                        # noqa: E402

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
        self.stage3 = SimpleNamespace(cfg=SimpleNamespace(
            provider="cached", model="claude-sonnet-4-5", ollama_model="qwen3:8b"))
        self.seen = []

    def classify(self, item, item_id=None):
        self.seen.append((item, item_id))
        stage = "S1" if "confident" in item["title"] else "S3"
        return {"leaf": "L0", "stage": stage, "s1_confidence": 0.99,
                "llm_confidence": None, "shortlist": ["L0", "L1"],
                "trace": [{"node": "s1", "ms": 4.2}], "total_ms": 4.2}


def main():
    rt = StubRuntime()
    A.STATE.update({"rt": rt, "load_s": 1.5})
    c = TestClient(A.app)

    print("\nhealth")
    h = c.get("/health").json()
    check("reports ok once loaded", h["status"] == "ok")
    check("exposes the device", h["device"] == "cpu")
    check("exposes retriever warmth separately", h["retriever_warm"] is False)
    check("echoes the operating point", h["config"]["tau"] == 0.9)
    check("names the provider", h["provider"] == "cached")

    print("\nroot")
    r = c.get("/", follow_redirects=False)
    check("/ redirects rather than 404ing", r.status_code in (302, 307))
    check("/ points at the docs", r.headers.get("location") == "/docs")

    print("\n/classify")
    r = c.post("/classify", json={"title": "confident widget"})
    check("200 on a minimal body", r.status_code == 200)
    check("returns the tier that decided", r.json()["stage"] == "S1")
    check("returns the per-node trace", r.json()["trace"][0]["node"] == "s1")
    check("item_id is not forwarded as a model field",
          "item_id" not in rt.seen[-1][0])
    r = c.post("/classify", json={"title": "x", "item_id": "B01"})
    check("item_id is forwarded separately", rt.seen[-1][1] == "B01")
    check("empty title is rejected",
          c.post("/classify", json={"title": ""}).status_code == 422)
    check("missing title is rejected",
          c.post("/classify", json={"brand": "Acme"}).status_code == 422)

    print("\n/classify/batch")
    body = {"items": [{"title": "confident a"}, {"title": "b"}, {"title": "c"}]}
    b = c.post("/classify/batch", json=body).json()
    check("returns one result per item", len(b["results"]) == 3)
    check("counts stages", b["summary"]["stage_counts"] == {"S1": 1, "S3": 2})
    # The endpoint rounds to 4dp for a readable payload; assert that contract exactly
    # rather than with a tolerance, so a change to the rounding is a visible failure.
    check("escalation share excludes S1",
          b["summary"]["escalated_pct"] == round(2 / 3, 4))
    check("empty batch is rejected",
          c.post("/classify/batch", json={"items": []}).status_code == 422)

    print("\nnot-yet-loaded")
    A.STATE["rt"] = None
    check("503 while models are loading",
          c.post("/classify", json={"title": "x"}).status_code == 503)
    check("health says loading", c.get("/health").json()["status"] == "loading")

    print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILURES: {FAILURES}"))
    sys.exit(0 if not FAILURES else 1)


if __name__ == "__main__":
    main()
