"""Build step 2a: turn each taxonomy leaf into a retrievable document.

Four variants, cumulative, so recall.py can measure what each addition buys:

  v1_name   leaf name, humanised
  v2_path   + browse-node ancestor path
  v3_proto  + k real product titles from TRAIN
  v4_desc   + a generated one-line description   (--describe, needs ANTHROPIC_API_KEY)

Prototypes come from the train split only. Drawing them from val or test would leak the
evaluation set into the thing being evaluated, and the leak would flatter exactly the
strata the project is about.
"""
import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd

SEED = 17
N_PROTOTYPES = 3
DESC_MODEL = "claude-sonnet-4-5"


def humanise(product_type: str) -> str:
    return product_type.replace("_", " ").lower().strip()


def dominant_path(paths: pd.Series) -> str:
    paths = [p for p in paths if p]
    if not paths:
        return ""
    return pd.Series(paths).value_counts().idxmax()


def build(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    rng = random.Random(SEED)
    train = df[df.split == "train"]
    docs: dict[str, dict[str, str]] = {}

    for pt in sorted(df.product_type.unique()):
        name = humanise(pt)
        rows = train[train.product_type == pt]

        path = dominant_path(df[df.product_type == pt].node_path)
        # ABO paths start '/Categories/...'; drop the constant root, keep the hierarchy.
        path = " > ".join(p for p in path.split("/") if p and p != "Categories")

        titles = list(rows.title.unique())
        rng.shuffle(titles)
        protos = titles[:N_PROTOTYPES]

        v1 = name
        v2 = f"{name}. Category: {path}." if path else v1
        v3 = v2 + (f" Examples: {'; '.join(protos)}." if protos else "")
        docs[pt] = {"v1_name": v1, "v2_path": v2, "v3_proto": v3, "v4_desc": v3}

    return docs


def describe(docs: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """One short description per leaf. 576 leaves ≈ well under a dollar."""
    from anthropic import Anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set.")
    client = Anthropic()

    for i, (pt, v) in enumerate(docs.items(), 1):
        msg = client.messages.create(
            model=DESC_MODEL,
            max_tokens=90,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Write one sentence describing what products belong in this "
                        "e-commerce category, and one clause on what is commonly confused "
                        "with it. Be concrete and use the words a shopper would use. "
                        "No preamble.\n\n"
                        f"Category: {v['v2_path'] or v['v1_name']}"
                    ),
                }
            ],
        )
        docs[pt]["v4_desc"] = f"{v['v3_proto']} {msg.content[0].text.strip()}"
        if i % 50 == 0:
            print(f"  described {i}/{len(docs)}")
    return docs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument(
        "--describe",
        action="store_true",
        help="generate v4_desc via the API (~$0.60). Without this, v4 == v3.",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.data / "items.parquet")
    docs = build(df)
    if args.describe:
        docs = describe(docs)

    out = args.data / "label_docs.json"
    out.write_text(json.dumps(docs, ensure_ascii=False, indent=2))
    print(f"Wrote {len(docs)} label documents to {out}")

    empty_path = sum(1 for v in docs.values() if v["v2_path"] == v["v1_name"])
    no_proto = sum(1 for v in docs.values() if v["v3_proto"] == v["v2_path"])
    print(f"  {empty_path} leaves have no browse path (v2 == v1)")
    print(f"  {no_proto} leaves have no train prototypes (v3 == v2) — expect zero-shot leaves here")

    sample = sorted(docs)[len(docs) // 2]
    print(f"\nExample — {sample}")
    for k, v in docs[sample].items():
        print(f"  {k:9s} {v[:150]}")


if __name__ == "__main__":
    main()
