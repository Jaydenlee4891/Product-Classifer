"""Fail the build when the documentation stops describing the code.

    python src/check_docs.py

This exists because of a specific incident. The README claimed 20 and 18 assertions for
the two serving test files when they had 26 and 20; `stage3_agent.py` described a second
agent pass that was never implemented; and two figures in the README were measured from a
dataset build that no longer exists. None of it was caught by a test, because none of it
is code -- and all of it is what a reader sees first.

Documentation drift is invisible to a normal test suite by construction. So the checks
here treat prose as an artefact with invariants:

  1. every assertion count the README states matches the number of check() calls in that
     file  -- the exact failure above
  2. the serving-layer total in the prose equals the sum of the per-file counts
  3. every src/ path listed in the README's repository tree actually exists  -- catches a
     file renamed or deleted without the tree being updated

What it deliberately does NOT do is verify measured results. A number like macro 0.647
can only be checked by re-running the pipeline, which needs weights and hours; claiming
otherwise here would be its own kind of drift. These three are the ones that are cheap,
exact, and have already gone wrong once.
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def count_checks(path: Path) -> int:
    """Calls to check(...) at statement level -- the assertion count these suites print."""
    tree = ast.parse(path.read_text())
    return sum(
        1 for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "check"
    )


def main() -> None:
    readme = (ROOT / "README.md").read_text()

    print("\nassertion counts stated in the README")
    stated = dict(re.findall(r"^(src/\S+\.py)\s+(\d+)\s+\S*\s*assertions", readme, re.M))
    if not stated:
        check("README states assertion counts at all", False, "no 'N assertions' lines found")
    serve_total = 0
    for rel, n in sorted(stated.items()):
        p = ROOT / rel
        if not p.exists():
            check(f"{rel} exists", False)
            continue
        actual = count_checks(p)
        check(f"{rel}: README says {n}", actual == int(n), f"actual check() calls = {actual}")
        if rel.startswith("src/serve/"):
            serve_total += actual

    print("\nserving-layer total quoted in prose")
    # Match against whitespace-collapsed prose: the README hard-wraps at ~90 columns, so
    # any literal multi-word pattern breaks the moment a sentence rewraps. That is a
    # brittleness worth naming -- a doc check that fails on reflow trains you to ignore it.
    flat = re.sub(r"\s+", " ", readme)
    m = re.search(r"(\d+) assertions cover routing", flat)
    if m:
        check(f"prose says {m.group(1)} assertions, files have {serve_total}",
              int(m.group(1)) == serve_total)
    else:
        check("prose states a serving assertion total", False)

    print("\nfiles listed in the repository tree")
    listed = sorted(set(re.findall(r"^(src/[\w/]+\.py)", readme, re.M)))
    missing = [r for r in listed if not (ROOT / r).exists()]
    check(f"all {len(listed)} listed src/ paths exist", not missing, f"missing: {missing}")

    print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILURES: {FAILURES}"))
    sys.exit(0 if not FAILURES else 1)


if __name__ == "__main__":
    main()
