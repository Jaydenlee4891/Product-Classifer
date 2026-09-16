"""Build step 5: Stage 1 — BERT classifier that is allowed to defer.

Two modes.

  --head-cut 0   pilot: softmax over every leaf that has training data (440 of 530).
                 This is also the "S1 alone, forced answer" baseline, and it produces
                 the per-class F1 needed to choose the head cut.

  --head-cut N   production: softmax over the top-N leaves by training frequency plus
                 one OTHER class. Deferral becomes a learned prediction rather than a
                 threshold artefact.

Deferral has two triggers and both matter:
    argmax == OTHER          learned  — "this is not one of mine"
    p_max   <  tau           confidence — catches head-vs-head confusion

The metric that decides everything is DEFERRAL RECALL: of the items S1 would get wrong,
what fraction does it hand to S2? Every point below 100% is an error no downstream stage
can reach, so it is reported per frequency stratum — the aggregate hides the tail.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from recall import item_text, save_array  # same text the retriever saw; they
                                          # must not diverge

OTHER = "__OTHER__"
STRATA = ["head", "torso", "tail", "few_shot", "zero_shot"]
TAUS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_label_space(train: pd.DataFrame, head_cut: int) -> tuple[list[str], dict[str, int]]:
    freq = train.product_type.value_counts()
    if head_cut <= 0:
        classes = sorted(freq.index)
        return classes, {c: i for i, c in enumerate(classes)}
    head = sorted(freq.head(head_cut).index)
    classes = head + [OTHER]
    idx = {c: i for i, c in enumerate(classes)}
    return classes, idx


def encode(tok, texts: list[str], max_len: int):
    enc = tok(texts, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def class_weights(y: np.ndarray, n_classes: int, scheme: str, device):
    """Counter the head's grip on the gradient. With one leaf at 53% of the corpus,
    unweighted cross-entropy gives a 3-example class effectively no signal — which is
    the leading explanation for why a linear TF-IDF model beat this on the tail."""
    if scheme == "none":
        return None
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = len(y) / (n_classes * counts) if scheme == "balanced" else 1.0 / np.sqrt(counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_model(model, loader, device, epochs, lr, use_fp16, weights=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    # No bf16 below compute capability 8.0 (a 1660 Ti is 7.5), so fp16 + GradScaler.
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    lossf = torch.nn.CrossEntropyLoss(weight=weights)
    model.train()
    step = 0
    for ep in range(epochs):
        run = 0.0
        for ids, mask, y in loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss = lossf(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            step += 1
            if step % 200 == 0:
                print(f"  epoch {ep+1} step {step}/{total}  loss {run/200:.4f}", flush=True)
                run = 0.0
    return model


@torch.no_grad()
def predict(model, loader, device, use_fp16) -> np.ndarray:
    model.eval()
    out = []
    for ids, mask, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


def deferral_table(probs: np.ndarray, ev: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """The gate. `answerable` = the gold leaf exists in S1's label space at all;
    items whose gold is outside it (zero-shot leaves, or non-head leaves in head+OTHER
    mode) are wrong by construction unless S1 defers, and must be counted that way.

    Broken out by SPLIT as well as by stratum. tau is fitted on val and reported on
    test; an earlier version pooled the two, which selects the threshold using the very
    set it is then reported on.
    """
    other_i = classes.index(OTHER) if OTHER in classes else -1
    idx = {c: i for i, c in enumerate(classes)}
    gold = ev.product_type.map(idx).fillna(-1).astype(int).to_numpy()
    answerable = (gold >= 0) & (gold != other_i)
    pred = probs.argmax(1)
    pmax = probs.max(1)
    would_be_wrong = ~answerable | (pred != gold)
    splits = {"val": (ev.split == "val").to_numpy(), "test": (ev.split == "test").to_numpy()}

    rows = []
    for tau in TAUS:
        deferred = (pred == other_i) | (pmax < tau)
        for split, s in splits.items():
            for stratum in STRATA + ["ALL"]:
                m = s if stratum == "ALL" else (s & (ev.stratum == stratum).to_numpy())
                if m.sum() == 0:
                    continue
                wrong_m = m & would_be_wrong
                acc_m = m & ~deferred
                rows.append(
                    {
                        "split": split,
                        "tau": tau,
                        "stratum": stratum,
                        "n": int(m.sum()),
                        "coverage": float((~deferred)[m].mean()),
                        "accepted_acc": float((pred[acc_m] == gold[acc_m]).mean()) if acc_m.sum() else float("nan"),
                        "deferral_recall": float(deferred[wrong_m].mean()) if wrong_m.sum() else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def fit_tau(pred, pmax, gold, other_i, mask, target: float):
    """Smallest tau whose accepted stream reaches `target` precision within `mask`.

    Returns None when no threshold reaches it, so the caller prints '--' instead of
    the closest miss, which would read as a hit.
    """
    for tau in np.arange(0.0, 1.0, 0.005):
        acc = mask & (pred != other_i) & (pmax >= tau)
        if acc.sum() == 0:
            return None
        if (pred[acc] == gold[acc]).mean() >= target:
            return float(tau)
    return None


def per_class_f1(probs: np.ndarray, ev: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Per-class F1 for S1, next to the frozen retriever's per-class top-1 where the
    cached embeddings allow it. The crossover between the two columns is what should
    set the head cut — a class belongs to S1 only where S1 beats S2 on it.

    Caveat, stated so it is not forgotten: retrieval top-1 is a LOWER BOUND on S2, which
    also has a cross-encoder. Treat the crossover as provisional until that exists.
    """
    idx = {c: i for i, c in enumerate(classes)}
    gold = ev.product_type.map(idx).fillna(-1).astype(int).to_numpy()
    pred = probs.argmax(1)
    rows = []
    for c, i in idx.items():
        if c == OTHER:
            continue
        tp = int(((pred == i) & (gold == i)).sum())
        fp = int(((pred == i) & (gold != i)).sum())
        fn = int(((pred != i) & (gold == i)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        rows.append({"leaf": c, "s1_f1": f1, "support": int((gold == i).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--model", default="distilbert-base-uncased", help="bert-base-uncased is the upgrade; KLUE-RoBERTa for the Korean arm")
    ap.add_argument("--head-cut", type=int, default=0, help="0 = pilot over all trainable leaves")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128, help="titles are 22 words at p95; 128 covers title+brand+bullets")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit-train", type=int, default=0, help="subsample training rows for a fast smoke run")
    ap.add_argument("--from-probs", action="store_true",
                    help="skip training; recompute the tables from a saved probs_<tag>.npy. "
                         "For fixing a reporting bug without spending a GPU-hour.")
    ap.add_argument("--class-weight", default="none", choices=["none", "sqrt", "balanced"],
                    help="reweight the loss by inverse class frequency. 'none' was the original "
                         "baseline and is the leading confound in the TF-IDF comparison: a linear "
                         "model learns a rare class from one distinctive n-gram, while unweighted "
                         "cross-entropy gives that class almost no gradient. 'sqrt' is the safer "
                         "start; 'balanced' can destabilise a 440-way head.")
    args = ap.parse_args()

    df = pd.read_parquet(args.data / "items.parquet")
    train = df[df.split == "train"]
    ev = df[df.split.isin(["val", "test"])].reset_index(drop=True)
    if args.limit_train:
        train = train.sample(n=min(args.limit_train, len(train)), random_state=17)

    classes, idx = build_label_space(train, args.head_cut)
    mode = "pilot (all trainable leaves)" if args.head_cut <= 0 else f"head {args.head_cut} + OTHER"
    n_leaves = df.product_type.nunique()
    natural = int(n_leaves - df[df.split == "train"].product_type.nunique())
    unreachable = int(n_leaves - train.product_type.nunique())
    print(f"Mode: {mode}")
    print(f"Label space: {len(classes)} classes over {n_leaves} leaves in the data")
    print(f"  {unreachable} leaves have no training rows here — unreachable by S1, they are S2's job")
    if args.limit_train:
        print(f"  WARNING: {unreachable - natural} of those are an artefact of --limit-train "
              f"({natural} genuinely have none). Subsampled numbers are not the real result.")
    print()

    y_train = train.product_type.map(lambda c: idx.get(c, idx.get(OTHER, -1))).to_numpy()
    if (y_train < 0).any():
        raise SystemExit("Unmapped training labels — label space construction is wrong.")

    out = args.data / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    tag = "pilot" if args.head_cut <= 0 else f"head{args.head_cut}"
    if args.class_weight != "none":
        tag += f"_{args.class_weight}"      # keep weighted runs from overwriting the baseline

    if args.from_probs:
        # Re-report a finished run. probs rows align with `ev` by construction, so a
        # reporting fix costs nothing but a reload — no GPU, no retraining.
        p = out / f"probs_{tag}.npy"
        if not p.exists():
            raise SystemExit(f"{p} not found — run without --from-probs first.")
        probs = np.load(p)
        if len(probs) != len(ev):
            raise SystemExit(
                f"{p} has {len(probs)} rows but ev has {len(ev)}. The splits changed "
                "since that run; these probabilities cannot be realigned. Retrain."
            )
        classes = json.loads((out / f"classes_{tag}.json").read_text())
        print(f"Loaded {p} ({len(probs):,} x {probs.shape[1]}) — reporting only, no training.")
    else:
        device = pick_device(args.device)
        use_fp16 = device == "cuda"
        print(f"Device: {device}{' (fp16 autocast)' if use_fp16 else ''}")
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(classes)).to(device)

        print(f"Tokenising {len(train):,} train / {len(ev):,} eval rows")
        tr_ids, tr_mask = encode(tok, [item_text(r) for r in train.itertuples()], args.max_len)
        ev_ids, ev_mask = encode(tok, [item_text(r) for r in ev.itertuples()], args.max_len)

        tr_loader = DataLoader(
            TensorDataset(tr_ids, tr_mask, torch.tensor(y_train)),
            batch_size=args.batch_size, shuffle=True, drop_last=True,
        )
        ev_loader = DataLoader(
            TensorDataset(ev_ids, ev_mask, torch.zeros(len(ev), dtype=torch.long)),
            batch_size=args.batch_size * 2,
        )

        w = class_weights(y_train, len(classes), args.class_weight, device)
        if w is not None:
            print(f"Class weighting '{args.class_weight}': "
                  f"min {w.min():.3f}  max {w.max():.3f}  ratio {w.max()/w.min():.0f}x")
        print(f"Training {args.epochs} epochs, {len(tr_loader)} steps/epoch")
        train_model(model, tr_loader, device, args.epochs, args.lr, use_fp16, w)

        probs = predict(model, ev_loader, device, use_fp16)
        save_array(out / f"probs_{tag}.npy", probs, model=model,
                   device=device, fp16_autocast=use_fp16)
        (out / f"classes_{tag}.json").write_text(json.dumps(classes))

        # Persist the checkpoint. Predictions are not a model: without this you cannot
        # score new items, point S1 at the Korean corpus, or demo anything without
        # retraining. data/ is gitignored, so ~265 MB stays out of the repo.
        ckpt = out / f"model_{tag}"
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        (ckpt / "run_config.json").write_text(
            json.dumps({**{k: str(v) for k, v in vars(args).items()},
                        # args.device is what was REQUESTED ("auto"); these two are what
                        # it resolved to, which is the part that reproduces a number.
                        "resolved_device": device, "fp16_autocast": use_fp16,
                        "n_classes": len(classes), "n_train": len(train)}, indent=2))
        print(f"Saved checkpoint -> {ckpt}")

    tbl = deferral_table(probs, ev, classes)
    tbl.to_csv(out / f"deferral_{tag}.csv", index=False)
    pc = per_class_f1(probs, ev, classes)
    pc.to_csv(out / f"per_class_{tag}.csv", index=False)

    # Every table below reads VAL. Test appears only in the funnel, once, at the tau
    # val already chose.
    v = tbl[tbl.split == "val"]
    cols = [s for s in STRATA + ["ALL"] if s in set(v.stratum)]
    for title, col in (
        ("accepted accuracy by tau (rows) x stratum (cols)", "accepted_acc"),
        ("DEFERRAL RECALL — the gate. Of what S1 gets wrong, how much reaches S2", "deferral_recall"),
        ("coverage (fraction S1 answers itself)", "coverage"),
    ):
        print(f"\n=== {title} — VAL ===")
        print(v.pivot(index="tau", columns="stratum", values=col)
               .reindex(columns=cols).to_string(float_format=lambda x: f"{x:.3f}"))

    print(f"\nMacro-F1 over {len(pc)} classes: {pc.s1_f1.mean():.3f}  "
          f"(median {pc.s1_f1.median():.3f}, classes with F1=0: {(pc.s1_f1 == 0).sum()})")

    # The funnel. tau is CHOSEN on val and REPORTED on test — the gap between the two
    # rows is the optimism that pooling them would have hidden.
    other_i = classes.index(OTHER) if OTHER in classes else -1
    cidx = {c: i for i, c in enumerate(classes)}
    gold = ev.product_type.map(cidx).fillna(-1).astype(int).to_numpy()
    pred, pmax = probs.argmax(1), probs.max(1)
    wrong = (gold < 0) | (gold == other_i) | (pred != gold)
    is_val = (ev.split == "val").to_numpy()
    is_test = (ev.split == "test").to_numpy()

    print(f"\n=== funnel: tau fitted on val (n={is_val.sum():,}), "
          f"reported on test (n={is_test.sum():,}) ===")
    print(f"{'target':>7s} {'tau':>6s} {'split':>6s} {'S1 answers':>11s} {'S1 acc':>8s} "
          f"{'-> S2':>7s} {'defer recall':>13s}")
    for target in (0.95, 0.98, 0.99, 0.995):
        tau = fit_tau(pred, pmax, gold, other_i, is_val, target)
        if tau is None:
            print(f"{target:>7.3f} {'--':>6s}   unreachable at any tau on val")
            continue
        for split, m in (("val", is_val), ("test", is_test)):
            acc_m = m & (pred != other_i) & (pmax >= tau)
            deferred = m & ~acc_m
            w = m & wrong
            dr = deferred[w].mean() if w.sum() else float("nan")
            acc = (pred[acc_m] == gold[acc_m]).mean() if acc_m.sum() else float("nan")
            print(f"{target:>7.3f} {tau:>6.3f} {split:>6s} {acc_m.sum() / m.sum():>10.1%} "
                  f"{acc:>8.3f} {deferred.sum() / m.sum():>6.1%} {dr:>13.3f}")
    print("\nRead the test row, not the val row. If test accuracy sits below the target")
    print("that val met, that gap IS the selection optimism — reporting the val row as")
    print("the result is the mistake this table exists to prevent.")
    print("An unnecessary deferral costs a fraction of a cent; a confident wrong label")
    print("is permanent. Read the last column first, then buy coverage with what is left.")
    print(f"Wrote {out}/")


if __name__ == "__main__":
    main()
