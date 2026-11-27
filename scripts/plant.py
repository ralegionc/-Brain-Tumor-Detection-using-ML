"""Round 6 (H21): plant real tumours, rescaled, into other patients' scans and score the model on them.

Donors are the largest 3D tumour component of each validation patient; recipients are the test
patients. See Round 6 in PREREGISTRATION.md for the exact recipe. Only the slices a planted
tumour touches (plus a margin) are run through the network: the model is 2D, so other slices
are unchanged.

    python scripts/plant.py [--seeds 0] [--limit N] [--threads 12]
Writes results/round6/plants.csv (one row per planted tumour) and a few example volumes for the
demo in results/round6/examples.npz. Round 7 scores other models on the same plants:
    python scripts/plant.py --seeds 0 1 2 --pattern r7_vgg16_plain_dice_bce_s{}_bank_small --out plants_S.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage, signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from seg.data import Slices  # noqa: E402
from seg.models import SegNet  # noqa: E402

OUT = ROOT / "results" / "round6"
SIZES = [5, 10, 20, 40, 80]
SMALL_Q, LARGE_Q = 25, 75  # donor groups: smallest and largest quarter (Round 6 addendum)
RING_MM, MIN_GAP_MM, SCORE_MM = 3.0, 20.0, 10.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def volume(ds, rows):
    rows = np.sort(rows)
    return np.asarray(ds.x[rows], np.float32).transpose(1, 0, 2, 3), ds.y[rows].astype(bool)  # (C,Z,H,W), (Z,H,W)


def ball(mm, spacing):
    r = [int(np.ceil(mm / s)) for s in spacing]
    g = np.ogrid[tuple(slice(-k, k + 1) for k in r)]
    return sum((gi * s / mm) ** 2 for gi, s in zip(g, spacing)) <= 1


def donors(spacing):
    ds = Slices("val")
    vox_ml = float(np.prod(spacing)) / 1000
    out = []
    for case, rows in ds.cases():
        x, y = volume(ds, rows)
        lab, n = ndimage.label(y, ndimage.generate_binary_structure(3, 3))
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        mask = lab == sizes.argmax() + 1
        sl = ndimage.find_objects(mask.astype(int))[0]
        pad = [int(np.ceil(RING_MM / s)) + 2 for s in spacing]
        sl = tuple(slice(max(s.start - p, 0), min(s.stop + p, dim)) for s, p, dim in zip(sl, pad, mask.shape))
        out.append({"case": case, "ml": sizes.max() * vox_ml, "mask": mask[sl], "img": x[(slice(None),) + sl]})
    return out


def scale(d, factor):
    """Zoom a donor patch by a linear factor on every axis."""
    img = np.stack([ndimage.zoom(c, factor, order=1) for c in d["img"]])
    m = ndimage.zoom(d["mask"].astype(np.float32), factor, order=1) >= 0.5
    return img, m


def allowed(x, y, spacing):
    """Voxels a planted tumour may cover: inside the eroded brain and far from the real tumour."""
    brain = ndimage.binary_erosion((x != 0).any(0), iterations=3)
    far = ndimage.distance_transform_edt(~y, sampling=spacing) >= MIN_GAP_MM if y.any() else np.ones_like(y)
    return brain & far


def plant(x, ok, img, m, spacing, rng, tries=200):
    """Paste (img, m) into volume x at a random spot where all of it fits; returns new x, planted mask,
    centre, or None. Every corner position is checked at once: a correlation of the forbidden voxels
    with the tumour mask is zero exactly where the whole tumour fits."""
    if any(a > b for a, b in zip(m.shape, ok.shape)):
        return None
    bad = signal.fftconvolve((~ok).astype(np.float32), m[::-1, ::-1, ::-1].astype(np.float32), mode="valid")
    cand = np.argwhere(bad < 0.5)
    if len(cand) == 0:
        return None
    half = np.array(m.shape) // 2
    for _ in range(tries):
        lo = cand[rng.integers(len(cand))]
        c = lo + half
        sl = tuple(slice(a, a + k) for a, k in zip(lo, m.shape))
        ring = ndimage.binary_dilation(m, ball(RING_MM, spacing)) & ~m
        target = x[(slice(None),) + sl]
        inside = (target != 0).any(0)
        r = ring & inside
        if r.sum() < 10:
            continue
        dr = ring & (img != 0).any(0)
        if dr.sum() < 10:
            continue
        shift = target[:, r].mean(1) - img[:, dr].mean(1)
        alpha = np.clip(ndimage.gaussian_filter(m.astype(np.float32), 1.0), 0, 1) * inside
        new = x.copy()
        new[(slice(None),) + sl] = target * (1 - alpha) + (img + shift[:, None, None, None]) * alpha
        planted = np.zeros(ok.shape, bool)
        planted[sl] = m
        return new, planted, c
    return None


@torch.no_grad()
def predict(model, x, zs, bs=32):
    """Probabilities for slices zs of volume x (C,Z,H,W)."""
    out = []
    for i in range(0, len(zs), bs):
        b = torch.from_numpy(np.ascontiguousarray(x[:, zs[i:i + bs]].transpose(1, 0, 2, 3)))
        dev = next(model.parameters()).device
        out.append(torch.sigmoid(model(b.to(dev)).float())[:, 0].cpu().numpy())
    return np.concatenate(out)


def score(models, x, planted, spacing):
    region = ndimage.binary_dilation(planted, ball(SCORE_MM, spacing))
    zs = np.flatnonzero(region.any((1, 2)))
    res = {}
    for s, m in models.items():
        p = predict(m, x, zs) >= 0.5
        pr, gt = p & region[zs], planted[zs]
        inter = (pr & gt).sum()
        res[f"dice_s{s}"] = 2 * inter / max(pr.sum() + gt.sum(), 1)
        res[f"sens_s{s}"] = inter / max(gt.sum(), 1)
        res[f"pred_ml_s{s}"] = pr.sum() * float(np.prod(spacing)) / 1000
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--pattern", default="r4_vgg16_plain_dice_bce_s{}", help="run folder, {} is the seed")
    ap.add_argument("--out", default="plants.csv", help="file name inside results/round6/")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    models = {}
    for s in a.seeds:
        m = SegNet("vgg16", False, pretrained=False)
        m.load_state_dict(torch.load(ROOT / "runs" / a.pattern.format(s) / "final.pt", weights_only=True,
                                     map_location="cpu"))
        models[s] = m.to(DEVICE).eval()
    ds = Slices("test")
    sp = ds.spacing
    vox_ml = float(np.prod(sp)) / 1000
    D = donors(sp)
    lo_ml, hi_ml = np.percentile([d["ml"] for d in D], [SMALL_Q, LARGE_Q])
    small = [d for d in D if d["ml"] <= lo_ml]
    large = [d for d in D if d["ml"] >= hi_ml]
    print(f"{len(D)} donors, {len(small)} small (<= {lo_ml:.0f} mL), {len(large)} large (>= {hi_ml:.0f} mL)", flush=True)
    rng = np.random.default_rng(2026)
    f = OUT / a.out
    done = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["recipient"])
    rows, examples = done.to_dict("records"), {}
    for k, (case, rr) in enumerate(ds.cases()):
        if a.limit and k >= a.limit:
            break
        jobs = [("native", D[rng.integers(len(D))], None)]
        for v in SIZES:
            jobs += [("small", small[rng.integers(len(small))], v), ("large", large[rng.integers(len(large))], v)]
        prng = np.random.default_rng([2026, k])
        if case in set(done.recipient):
            continue
        x, y = volume(ds, rr)
        ok = allowed(x, y, sp)
        brain = (x != 0).any(0)
        depth_map = ndimage.distance_transform_edt(brain, sampling=sp)
        bmin, bptp = np.argwhere(brain).min(0), np.maximum(np.ptp(np.argwhere(brain), 0), 1)
        for kind, d, v in jobs:
            factor = 1.0 if v is None else (v / d["ml"]) ** (1 / 3)
            img, m = (d["img"], d["mask"]) if v is None else scale(d, factor)
            p = plant(x, ok, img, m, sp, prng)
            row = {"recipient": case, "donor": d["case"], "kind": kind, "donor_ml": d["ml"],
                   "target_ml": v if v is not None else d["ml"], "factor": factor}
            if p is None:
                rows.append({**row, "placed": False})
                continue
            nx, planted, c = p
            depth = depth_map[tuple(c)]
            rel = (c - bmin) / bptp
            rows.append({**row, "placed": True, "planted_ml": planted.sum() * vox_ml, "z": c[0], "cy": c[1], "cx": c[2],
                         "rel_z": rel[0], "rel_y": rel[1], "rel_x": rel[2], "depth_mm": depth,
                         **score(models, nx, planted, sp)})
            if len(examples) < 6 and kind in ("small", "native") and (v in (10, None)):
                zc = int(c[0])
                examples[f"{case}_{kind}_{v}"] = np.concatenate([nx[:, zc], planted[zc][None]]).astype(np.float16)
        pd.DataFrame(rows).to_csv(f, index=False)
        last = rows[-1]
        print(f"{k} {case} done, last dice {last.get(f'dice_s{a.seeds[0]}', float('nan')):.3f}", flush=True)
    if examples and a.out == "plants.csv":
        np.savez_compressed(OUT / "examples.npz", **examples)


if __name__ == "__main__":
    main()
