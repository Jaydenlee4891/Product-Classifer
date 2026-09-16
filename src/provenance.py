"""Which machine and which precision produced each array in data/?

    python src/provenance.py --data data

Arrays written from now on carry a <name>.meta.json sidecar (recall.save_array). Arrays
written before that do not, so for those this reports an INFERRED precision instead,
from a property that survives the cast to float32: a tensor computed in fp16 and stored
as fp32 has every value on an fp16 grid point, which a genuine fp32 computation does
with probability ~5e-4 per element.

That inference is how the mixed provenance in this project was found. probs_*.npy and
scores_*.npy are fp32; every emb_*.npy is fp16. The embeddings were produced on a CUDA
card, the rest on an Apple GPU, and the run_config.json files recorded "device": "auto"
-- the flag passed, not the device it resolved to.

Why it matters, measured rather than assumed: re-encoding the same items on a different
machine moves each vector by ~4e-4 cosine, which is enough to reshuffle a fused top-10
out of 530 labels on 92% of deferred items and to cost 2.5 points of gold-leaf
reachability. See docs/serving-parity.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np

FP32_BASELINE = 5e-4     # chance an unrelated fp32 value lands on an fp16 grid point


def infer_precision(path: Path, sample: int = 2000) -> tuple[str, float]:
    a = np.load(path, mmap_mode="r")
    if not np.issubdtype(a.dtype, np.floating):
        return f"n/a ({a.dtype})", float("nan")
    x = np.asarray(a[:sample]).astype(np.float32)
    frac = float((x.astype(np.float16).astype(np.float32) == x).mean())
    if frac > 0.99:
        return "fp16", frac
    if frac < 0.01:
        return "fp32", frac
    return "mixed/unclear", frac


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()

    paths = sorted(args.data.rglob("*.npy"))
    if not paths:
        raise SystemExit(f"no .npy files under {args.data}")

    print(f"{'array':<36}{'stamped device':<18}{'precision':<10}{'frac':>8}  source")
    print("-" * 88)
    seen = {}
    for p in paths:
        meta_p = p.with_suffix(".meta.json")
        if meta_p.exists():
            m = json.loads(meta_p.read_text())
            dev = str(m.get("device"))
            prec = str(m.get("dtype", "")).replace("torch.", "") or "-"
            src, frac = "sidecar", float("nan")
        else:
            dev = "unrecorded"
            prec, frac = infer_precision(p)
            src = "inferred"
        rel = str(p.relative_to(args.data))
        print(f"{rel:<36}{dev:<18}{prec:<10}{frac:>8.4f}  {src}"
              if src == "inferred" else
              f"{rel:<36}{dev:<18}{prec:<10}{'':>8}  {src}")
        seen.setdefault((dev, prec), []).append(rel)

    print()
    for cfg_name in ("stage1", "stage2"):
        for rc in sorted(args.data.rglob("run_config.json")):
            if cfg_name not in str(rc):
                continue
            m = json.loads(rc.read_text())
            res = repr(m["resolved_device"]) if "resolved_device" in m else \
                "NOT RECORDED (pre-dates stamping)"
            req = repr(m.get("device"))
            print(f"{str(rc.relative_to(args.data)):<44}"
                  f"requested={req} resolved={res}")

    print()
    if len(seen) > 1:
        print(f"{len(seen)} distinct (device, precision) combinations across data/.")
        print("Arrays compared against each other must come from the same one, or the")
        print("difference between them is the machine, not the model:")
        for (dev, prec), names in sorted(seen.items(), key=lambda kv: -len(kv[1])):
            print(f"  {dev:<14} {prec:<8} {len(names):>2} file(s): "
                  f"{', '.join(names[:3])}{' ...' if len(names) > 3 else ''}")
    else:
        print("All arrays share one (device, precision). Nothing to reconcile.")


if __name__ == "__main__":
    main()
