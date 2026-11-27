"""Round 7: build a bank of training slices with planted tumours (see Round 7 in PREREGISTRATION.md).

Recipients and donors are training patients only. Each recipient gets --per plants, each from a
random other patient's largest tumour component, placed with the round 6 recipe (scripts/plant.py).
Up to --keep slices touched by each planted tumour are stored; their label is the recipient's
own tumour plus the planted one. The bank is written in the same format as data/slices, as a
"train" split, so seg.data.Slices reads it and applies the usual augmentation.

    python scripts/build_bank.py small data/bank_small     # targets log-uniform 3 to 30 mL
    python scripts/build_bank.py native data/bank_native   # donors at their own size
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from plant import RING_MM, allowed, plant, scale, volume  # noqa: E402
from seg.data import DATA, Slices  # noqa: E402

SEEDS = {"small": 7001, "native": 7002}


def train_donors(ds):
    vox_ml = float(np.prod(ds.spacing)) / 1000
    out = {}
    for case, rows in ds.cases():
        x, y = volume(ds, rows)
        lab, n = ndimage.label(y, ndimage.generate_binary_structure(3, 3))
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        mask = lab == sizes.argmax() + 1
        sl = ndimage.find_objects(mask.astype(int))[0]
        pad = [int(np.ceil(RING_MM / s)) + 2 for s in ds.spacing]  # as in plant.donors
        sl = tuple(slice(max(s.start - p, 0), min(s.stop + p, dim)) for s, p, dim in zip(sl, pad, mask.shape))
        out[case] = {"case": case, "ml": sizes.max() * vox_ml, "mask": mask[sl], "img": x[(slice(None),) + sl]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=list(SEEDS))
    ap.add_argument("out")
    ap.add_argument("--per", type=int, default=8, help="plants per recipient")
    ap.add_argument("--keep", type=int, default=5, help="slices kept per plant")
    ap.add_argument("--tries", type=int, default=60, help="donor draws per recipient before giving up")
    ap.add_argument("--limit", type=int, default=0, help="only the first N recipients (smoke tests)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = Slices("train")
    sp = ds.spacing
    D = train_donors(ds)
    names = sorted(D)
    print(f"{len(D)} donors, median {np.median([d['ml'] for d in D.values()]):.0f} mL", flush=True)
    rng = np.random.default_rng(SEEDS[a.kind])
    xs, ys, idx = [], [], []
    tried = placed = 0
    for k, (case, rows) in enumerate(ds.cases()):
        if a.limit and k >= a.limit:
            break
        x, y = volume(ds, rows)
        ok = allowed(x, y, sp)
        j = attempts = 0
        while j < a.per and attempts < a.tries:  # a donor that fits nowhere is redrawn (round 7 addendum)
            attempts += 1
            donor = case
            while donor == case:
                donor = names[rng.integers(len(names))]
            d = D[donor]
            target = float(np.exp(rng.uniform(np.log(3), np.log(30)))) if a.kind == "small" else d["ml"]
            img, m = (d["img"], d["mask"]) if a.kind == "native" else scale(d, (target / d["ml"]) ** (1 / 3))
            tried += 1
            p = plant(x, ok, img, m, sp, rng)
            if p is None:
                continue
            placed += 1
            j += 1
            nx, planted, _ = p
            zs = np.flatnonzero(planted.any((1, 2)))
            zs = np.sort(rng.choice(zs, min(a.keep, len(zs)), replace=False))
            lab = (y | planted).astype(np.uint8)
            for z in zs:
                xs.append(nx[:, z].astype(np.float16))
                ys.append(lab[z])
                idx.append({"case": f"{case}_p{j - 1}", "z": int(z), "recipient": case, "donor": donor,
                            "target_ml": round(target, 2), "planted_px": int(planted[z].sum())})
        if k % 25 == 0:
            print(f"{k} {case}: {placed}/{tried} placed, {len(xs)} slices", flush=True)
    n, s = len(xs), ds.x.shape[-1]
    X = np.memmap(out / "train_x.bin", dtype=np.float16, mode="w+", shape=(n, 3, s, s))
    X[:] = np.stack(xs)
    X.flush()
    Y = np.memmap(out / "train_y.bin", dtype=np.uint8, mode="w+", shape=(n, s, s))
    Y[:] = np.stack(ys)
    Y.flush()
    pd.DataFrame(idx).to_csv(out / "train_index.csv", index=False)
    meta = json.load(open(DATA / "meta.json"))
    json.dump({**meta, "counts": {"train": n}, "bank": {"kind": a.kind, "plants": placed, "tried": tried,
                                                        "per": a.per, "keep": a.keep, "tries": a.tries, "seed": SEEDS[a.kind]}},
              open(out / "meta.json", "w"), indent=1)
    print(f"bank {a.kind}: {placed} of {tried} plants placed, {n} slices -> {out}", flush=True)


if __name__ == "__main__":
    main()
