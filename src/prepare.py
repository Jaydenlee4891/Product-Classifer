"""Build step 1: parse ABO listings, cut splits, assign frequency strata.

Target label space is `product_type` (576 flat leaves). The `node` browse paths are
kept as text for the label documents, not as a prediction target.
"""
import argparse
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

SEED = 17
N_ZERO_SHOT_LEAVES = 30
STRATA = [("head", 1000, None), ("torso", 100, 999), ("tail", 10, 99), ("few_shot", 1, 9)]


def find_listing_files(raw: Path) -> list[Path]:
    files = sorted(raw.rglob("listings_*.json.gz"))
    if not files:
        sys.exit(
            f"No listings_*.json.gz found under {raw}.\n"
            "Download ABO listings metadata first — see README."
        )
    return files


def probe_schema(path: Path) -> None:
    """Print the shape of the first record. ABO's schema is the one assumption
    this script cannot verify on its own, so make it visible before parsing 147k rows."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rec = json.loads(f.readline())
    print(f"Schema probe — {path.name}, first record:")
    for k in sorted(rec):
        v = rec[k]
        shape = f"list[{len(v)}] e.g. {v[0]!r}" if isinstance(v, list) and v else repr(v)
        print(f"  {k:20s} {shape[:110]}")
    missing = {"item_id", "item_name", "product_type"} - set(rec)
    if missing:
        sys.exit(
            f"\nExpected fields missing: {sorted(missing)}.\n"
            "ABO's schema differs from what this script assumes. Fix the extractors "
            "in prepare.py against the field names printed above rather than guessing."
        )
    print()


def pick_en(field) -> str | None:
    """ABO stores localised fields as [{language_tag, value}, ...]. Prefer en_US,
    then any English, then nothing — non-English rows are dropped rather than mixed in."""
    if not isinstance(field, list):
        return None
    vals = [e for e in field if isinstance(e, dict) and e.get("value")]
    for want in ("en_US", "en_GB", "en_IN", "en_AU", "en_CA", "en_SG", "en_AE"):
        for e in vals:
            if e.get("language_tag") == want:
                return e["value"].strip()
    for e in vals:
        if str(e.get("language_tag", "")).startswith("en"):
            return e["value"].strip()
    return None


def first_value(field) -> str | None:
    """product_type is [{'value': 'CELLULAR_PHONE_CASE'}] — but tolerate a bare string."""
    if isinstance(field, str):
        return field.strip() or None
    if isinstance(field, list) and field:
        e = field[0]
        if isinstance(e, dict) and e.get("value"):
            return str(e["value"]).strip()
        if isinstance(e, str):
            return e.strip() or None
    return None


def node_path(field) -> str | None:
    """Longest English browse path, e.g. '/Categories/Cell Phones & Accessories/Cases'."""
    if not isinstance(field, list):
        return None
    names = [
        e["node_name"].strip()
        for e in field
        if isinstance(e, dict)
        and e.get("node_name")
        and str(e.get("language_tag", "en_US")).startswith("en")
    ]
    return max(names, key=len) if names else None


def parse(files: list[Path]) -> pd.DataFrame:
    rows, skipped = [], Counter()
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                pt = first_value(rec.get("product_type"))
                name = pick_en(rec.get("item_name"))
                if not pt:
                    skipped["no_product_type"] += 1
                    continue
                if not name:
                    skipped["no_english_name"] += 1
                    continue
                bullets = [
                    b["value"].strip()
                    for b in (rec.get("bullet_point") or [])
                    if isinstance(b, dict)
                    and b.get("value")
                    and str(b.get("language_tag", "")).startswith("en")
                ][:3]
                rows.append(
                    {
                        "item_id": rec.get("item_id"),
                        "product_type": pt,
                        "title": name,
                        "brand": pick_en(rec.get("brand")) or "",
                        "bullets": " ".join(bullets),
                        "node_path": node_path(rec.get("node")) or "",
                        "main_image_id": rec.get("main_image_id") or "",
                    }
                )
    print(f"Parsed {len(rows):,} items. Skipped: {dict(skipped)}\n")
    return pd.DataFrame(rows)


def split(df: pd.DataFrame) -> pd.DataFrame:
    """80/10/10 stratified by product_type. Leaves with <3 items send one item to test
    and keep the rest in train — those become the few-shot stratum, which is the point."""
    rng = random.Random(SEED)
    assign = {}
    for pt, grp in df.groupby("product_type"):
        ids = list(grp["item_id"])
        rng.shuffle(ids)
        if len(ids) < 3:
            assign.update({i: "train" for i in ids[:-1]})
            assign[ids[-1]] = "test"
            continue
        n_val = max(1, int(0.1 * len(ids)))
        n_test = max(1, int(0.1 * len(ids)))
        for i in ids[:n_val]:
            assign[i] = "val"
        for i in ids[n_val : n_val + n_test]:
            assign[i] = "test"
        for i in ids[n_val + n_test :]:
            assign[i] = "train"
    df["split"] = df["item_id"].map(assign)
    return df


def make_zero_shot(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """A true zero-shot stratum does not exist in ABO — every product_type has items.
    Construct one by deleting all TRAIN rows for N leaves while keeping their val/test
    rows and their label documents. Those leaves are then reachable only via label text."""
    rng = random.Random(SEED)
    train_counts = df[df.split == "train"].product_type.value_counts()
    eligible = [
        pt
        for pt, c in train_counts.items()
        if 10 <= c <= 200 and (df[(df.product_type == pt) & (df.split != "train")].shape[0] >= 3)
    ]
    if len(eligible) < N_ZERO_SHOT_LEAVES:
        print(f"WARNING: only {len(eligible)} leaves eligible for zero-shot holdout.")
    chosen = rng.sample(eligible, min(N_ZERO_SHOT_LEAVES, len(eligible)))
    drop = df[(df.product_type.isin(chosen)) & (df.split == "train")].index
    print(f"Zero-shot holdout: {len(chosen)} leaves, {len(drop):,} train rows removed.\n")
    return df.drop(index=drop), chosen


def assign_strata(df: pd.DataFrame, zero_shot: list[str]) -> pd.DataFrame:
    """Stratum is a property of the LEAF, set by its post-holdout train frequency."""
    counts = df[df.split == "train"].product_type.value_counts()

    def stratum(pt: str) -> str:
        if pt in zero_shot:
            return "zero_shot"
        n = int(counts.get(pt, 0))
        if n == 0:
            return "zero_shot"
        for name, lo, hi in STRATA:
            if n >= lo and (hi is None or n <= hi):
                return name
        return "few_shot"

    df["stratum"] = df["product_type"].map(stratum)
    return df


def plot_tail(df: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = df[df.split == "train"].product_type.value_counts().values
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(range(1, len(counts) + 1), counts, lw=1.6, color="#1d5f58")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("leaf rank (log)")
    ax.set_ylabel("training items (log)")
    ax.set_title(f"ABO product_type frequency — {len(counts)} leaves with training data")
    ax.grid(alpha=0.25, which="both", lw=0.5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    files = find_listing_files(args.raw)
    print(f"Found {len(files)} listing files.\n")
    probe_schema(files[0])

    df = parse(files)
    df = split(df)
    df, zero_shot = make_zero_shot(df)
    df = assign_strata(df, zero_shot)

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out / "items.parquet", index=False)
    (args.out / "zero_shot_leaves.json").write_text(json.dumps(sorted(zero_shot), indent=2))
    plot_tail(df, args.out.parent / "figs" / "frequency.png")

    order = ["head", "torso", "tail", "few_shot", "zero_shot"]
    leaves = df.groupby("stratum").product_type.nunique().reindex(order).fillna(0).astype(int)
    summary = pd.DataFrame(
        {
            "leaves": leaves,
            "train_items": df[df.split == "train"].stratum.value_counts().reindex(order).fillna(0).astype(int),
            "val_items": df[df.split == "val"].stratum.value_counts().reindex(order).fillna(0).astype(int),
            "test_items": df[df.split == "test"].stratum.value_counts().reindex(order).fillna(0).astype(int),
        }
    )
    print(f"\n{len(df):,} items · {df.product_type.nunique()} leaves\n")
    print(summary.to_string())
    summary.to_csv(args.out / "strata.csv")

    thin = summary[(summary.leaves > 0) & (summary.val_items < 20)]
    if not thin.empty:
        print(
            f"\nWARNING: {list(thin.index)} have <20 val items — recall on those strata "
            "will be too noisy to draw conclusions from. Consider evaluating on val+test."
        )


if __name__ == "__main__":
    main()
