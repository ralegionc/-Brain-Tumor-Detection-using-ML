"""Round 5 inference over the three round 4 VGG-16 plain models (H16c, H17, H19).

For every validation and test patient: each seed's probability map with and without a
left-right flipped copy, the three-seed ensemble, and seed disagreement. Writes to
results/round5/ (or --out):
  clean_val.csv        validation Dice per seed, patient and component cut-off (H19b picks on it)
  test_{variant}.csv   full test metrics per patient for flip_s{k}, clean_s{k} and ensemble
  uncertainty.csv      per test patient: disagreement and uncertain volumes, ensemble Dice
  ensemble_{split}_hist.npz  256-bin tumour / non-tumour histograms of the ensemble (H16c)

    python scripts/round5_infer.py runs/r4_vgg16_plain_dice_bce_s0 runs/..._s1 runs/..._s2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from seg.data import Slices  # noqa: E402
from seg.metrics import error_patterns, overlap, volume_metrics  # noqa: E402
from seg.models import SegNet  # noqa: E402
from seg.train import DEVICE  # noqa: E402

CUTS_ML = [0.05, 0.1, 0.25, 0.5, 1, 2]


def load(d):
    cfg = json.load(open(d / "config.json"))
    m = SegNet(cfg["encoder"], cfg["attention"], pretrained=False, in_scale=cfg.get("in_scale", 1))
    m.load_state_dict(torch.load(d / "final.pt", weights_only=True, map_location="cpu"))
    return m.to(DEVICE).eval()


@torch.no_grad()
def predict(model, ds, rows, bs=32):
    """Probabilities without and with a left-right flip (the flip used in training augmentation)."""
    plain, flip = [], []
    for i in range(0, len(rows), bs):
        x, _ = ds.batch(rows[i:i + bs])
        x = x.to(DEVICE)
        plain.append(torch.sigmoid(model(x)).float()[:, 0].cpu().numpy())
        flip.append(torch.sigmoid(model(x.flip(-2))).float().flip(-2)[:, 0].cpu().numpy())
    return np.concatenate(plain), np.concatenate(flip)


def remove_small(mask, min_ml, vox_ml):
    lab, n = ndimage.label(mask, ndimage.generate_binary_structure(3, 3))
    if n == 0:
        return mask
    size = np.bincount(lab.ravel())[1:] * vox_ml
    return np.isin(lab, np.flatnonzero(size >= min_ml) + 1)


def full(pred, gt, spacing):
    return {**volume_metrics(pred, gt, spacing), **error_patterns(pred, gt, spacing)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs=3)
    ap.add_argument("--out", default=str(ROOT / "results" / "round5"))
    ap.add_argument("--limit", type=int, default=0, help="first N patients per split (smoke test)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = [load(Path(r)) for r in a.runs]

    for split in ["val", "test"]:
        ds = Slices(split)
        vox_ml = float(np.prod(ds.spacing)) / 1000.0
        if split == "test":
            cut = pd.read_csv(out / "clean_val.csv").groupby(["seed", "cut_ml"]).dice.mean()
            best = {s: cut[s].idxmax() for s in range(3)}
            json.dump({str(s): float(v) for s, v in best.items()}, open(out / "clean_cutoffs.json", "w"))
            print("component cut-offs picked on validation (mL):", best, flush=True)
        rows_clean, rows_unc, res, hp, hn = [], [], {}, [], []
        for k, (case, rows) in enumerate(ds.cases()):
            if a.limit and k >= a.limit:
                break
            gt = ds.y[np.sort(rows)].astype(bool)
            P = [predict(m, ds, rows) for m in models]
            ens = np.mean([p for p, _ in P], axis=0)
            q = np.round(ens * 255).astype(np.uint8)
            hp.append(np.bincount(q[gt], minlength=256))
            hn.append(np.bincount(q[~gt], minlength=256))
            if split == "val":
                for s, (p, _) in enumerate(P):
                    for c in CUTS_ML:
                        rows_clean.append({"seed": s, "case": case, "cut_ml": c,
                                           "dice": overlap(remove_small(p >= 0.5, c, vox_ml), gt)["dice"]})
                continue
            masks = [p >= 0.5 for p, _ in P]
            union, agree_all = np.any(masks, 0), np.all(masks, 0)
            e = ens >= 0.5
            rows_unc.append({"case": case, "gt_ml": gt.sum() * vox_ml, "pred_union_ml": union.sum() * vox_ml,
                             "disagree_ml": (union & ~agree_all).sum() * vox_ml,
                             "disagree_frac": (union & ~agree_all).sum() / max(union.sum(), 1),
                             "uncertain_ml": ((ens > 0.1) & (ens < 0.9)).sum() * vox_ml,
                             "uncertain_frac": ((ens > 0.1) & (ens < 0.9)).sum() / max(e.sum(), 1),
                             "ensemble_dice": overlap(e, gt)["dice"]})
            res.setdefault("ensemble", []).append({"case": case, **full(e, gt, ds.spacing)})
            for s, (p, pf) in enumerate(P):
                res.setdefault(f"flip_s{s}", []).append({"case": case, **full((p + pf) / 2 >= 0.5, gt, ds.spacing)})
                res.setdefault(f"clean_s{s}", []).append(
                    {"case": case, **full(remove_small(p >= 0.5, best[s], vox_ml), gt, ds.spacing)})
            if k % 20 == 0:
                print(f"  {split} {k} {case} ensemble dice {rows_unc[-1]['ensemble_dice']:.3f}", flush=True)
        np.savez(out / f"ensemble_{split}_hist.npz", pos=np.array(hp), neg=np.array(hn))
        if split == "val":
            pd.DataFrame(rows_clean).to_csv(out / "clean_val.csv", index=False)
        else:
            pd.DataFrame(rows_unc).to_csv(out / "uncertainty.csv", index=False)
            for v, r in res.items():
                pd.DataFrame(r).to_csv(out / f"test_{v}.csv", index=False)
        print(split, "done", flush=True)


if __name__ == "__main__":
    main()
