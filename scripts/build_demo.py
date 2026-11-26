"""Build the static demo in docs/ (served by GitHub Pages; no server, no model in the browser).

For a handful of test patients, and three planted tumours, it stores the slices around the tumour
as image sprites: FLAIR and T1gd as JPEG, and a PNG whose red channel is the true mask, green the
three-seed mean probability, and blue the number of seeds that call a voxel tumour. docs/data.json
holds the per-case numbers (Dice, volume range, seed disagreement). Run after analyze_round6.py.

    python scripts/build_demo.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from plant import SCORE_MM, allowed, ball, donors, plant, predict, scale, volume  # noqa: E402
from seg.data import Slices  # noqa: E402
from seg.metrics import overlap  # noqa: E402
from seg.models import SegNet  # noqa: E402

DOCS = ROOT / "docs"
COLS = 10
MARGIN = 6


def sprite(frames, mode):
    """Stack (Z,H,W[,C]) frames into a grid image, COLS per row; orientation: anterior up."""
    frames = [np.rot90(f, 1) for f in frames]
    h, w = frames[0].shape[:2]
    rows = -(-len(frames) // COLS)
    grid = np.zeros((rows * h, COLS * w) + frames[0].shape[2:], np.uint8)
    for i, f in enumerate(frames):
        grid[(i // COLS) * h:(i // COLS + 1) * h, (i % COLS) * w:(i % COLS + 1) * w] = f
    return Image.fromarray(grid)


def to_u8(c):
    return np.clip((c + 2.5) / 6.5 * 255, 0, 255).astype(np.uint8)


def write_case(key, x, y, probs, z0, z1):
    out = DOCS / "cases"
    zs = range(z0, z1)
    for ch, name in [(0, "flair"), (1, "t1gd")]:
        sprite([to_u8(x[ch, z]) * (x[:, z] != 0).any(0) for z in zs], "L").save(out / f"{key}_{name}.jpg", quality=88)
    mean = np.mean(probs, 0)
    votes = np.sum([p >= 0.5 for p in probs], 0)
    rgb = np.stack([y[z0:z1] * 255, mean * 255, votes * 85], -1).astype(np.uint8)
    sprite(list(rgb), "RGB").save(out / f"{key}_masks.png", optimize=True)
    return {"n": z1 - z0, "cols": COLS, "size": int(x.shape[-1])}


def main():
    torch.set_num_threads(4)
    (DOCS / "cases").mkdir(parents=True, exist_ok=True)
    models = []
    for s in range(3):
        m = SegNet("vgg16", False, pretrained=False)
        m.load_state_dict(torch.load(ROOT / "runs" / f"r4_vgg16_plain_dice_bce_s{s}" / "final.pt", weights_only=True,
                                     map_location="cpu"))
        models.append(m.eval())
    U = pd.read_csv(ROOT / "results" / "round5" / "uncertainty.csv").set_index("case")
    C = pd.read_csv(ROOT / "results" / "round6_conformal.csv")
    C = C[C.setting.str.startswith("A: VGG")].set_index("case")
    flagged = set(U.disagree_frac.nlargest(12).index)
    order = U.sort_values("ensemble_dice")
    picks = {order.index[0]: "Worst case", order.index[1]: "Second worst",
             U[U.gt_ml < 30].ensemble_dice.idxmax(): "Small tumour, done well",
             order.index[len(order) // 2]: "Typical patient",
             U.gt_ml.idxmax(): "Largest tumour",
             U[~U.index.isin(flagged)].ensemble_dice.idxmin(): "Bad, but not flagged"}
    ds = Slices("test")
    rows = dict(ds.cases())
    cases = []
    for case, label in picks.items():
        x, y = volume(ds, rows[case])
        zs = np.flatnonzero(y.any((1, 2)))
        z0, z1 = max(zs.min() - MARGIN, 0), min(zs.max() + MARGIN + 1, y.shape[0])
        probs = [predict(m, x, np.arange(z0, z1)) for m in models]
        meta = write_case(case, x[:, z0:z1], y[z0:z1], probs, 0, z1 - z0)
        u, c = U.loc[case], C.loc[case]
        cases.append({"key": case, "kind": "patient", "label": label, **meta, "dice": round(float(u.ensemble_dice), 3),
                      "gt_ml": round(float(u.gt_ml), 1), "pred_ml": round(float(c.v_hat), 1),
                      "lo_ml": round(max(float(c.lo_ml), 0), 1), "hi_ml": round(float(c.hi_ml), 1),
                      "covered": bool(c.covered), "disagree": round(float(u.disagree_frac), 3),
                      "flagged": case in flagged})
        print(case, label, flush=True)

    D = donors(ds.spacing)
    small = [d for d in D if d["ml"] <= np.percentile([e["ml"] for e in D], 25)]
    rng = np.random.default_rng(7)
    for case, target in [(order.index[len(order) // 2], 10), (order.index[len(order) // 3], 5),
                         (order.index[2 * len(order) // 3], 40)]:
        x, y = volume(ds, rows[case])
        d = small[rng.integers(len(small))]
        img, m = scale(d, (target / d["ml"]) ** (1 / 3))
        p = plant(x, allowed(x, y, ds.spacing), img, m, ds.spacing, rng)
        if p is None:
            continue
        nx, planted, c = p
        zs = np.flatnonzero(planted.any((1, 2)))
        z0, z1 = max(zs.min() - MARGIN, 0), min(zs.max() + MARGIN + 1, y.shape[0])
        probs = [predict(mm, nx, np.arange(z0, z1)) for mm in models]
        key = f"{case}_planted_{target}"
        meta = write_case(key, nx[:, z0:z1], planted[z0:z1], probs, 0, z1 - z0)
        region = ndimage.binary_dilation(planted, ball(SCORE_MM, ds.spacing))[z0:z1]
        dice = overlap((np.mean(probs, 0) >= 0.5) & region, planted[z0:z1])["dice"]
        cases.append({"key": key, "kind": "planted", "label": f"Planted {target} mL tumour", **meta,
                      "dice": round(float(dice), 3), "gt_ml": target, "donor": d["case"], "recipient": case,
                      "note": "Only the planted tumour is marked as truth; the patient's own tumour is elsewhere."})
        print(key, f"dice {dice:.3f}", flush=True)
    json.dump({"cases": cases}, open(DOCS / "data.json", "w"), indent=1, default=lambda o: o.item())


if __name__ == "__main__":
    main()
