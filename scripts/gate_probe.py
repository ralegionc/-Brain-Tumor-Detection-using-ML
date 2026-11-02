"""H8: what the attention gates do at the tumour edge (see Round 3 in PREREGISTRATION.md).

For each gated whole-tumour MSD model, one pass with the gates intact records, per test
patient, the mean gate value in bins of in-plane distance from the true tumour boundary
(tumour-bearing slices only), and scores the predictions. Further passes re-score the test
set with gates ablated at test time: "mean" replaces every gate map by its per-slice spatial
mean; on seed 0, "mean_L1".."mean_L4" ablate one level (1 = finest) and "ones" opens all gates.
Writes results/gate_profile.csv and runs/<model>/ablate_<variant>_metrics.csv; finished
passes are skipped, so the script can be stopped and restarted.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from seg.data import Slices  # noqa: E402
from seg.metrics import error_patterns, volume_metrics  # noqa: E402
from seg.models import SegNet  # noqa: E402

MODELS = [f"{e}_ag_dice_bce_s{s}" for e in ["inceptionv3", "vgg16"] for s in range(3)]
BINS = ["in_deep", "in_edge", "out_edge", "out_mid", "out_far", "outside_brain"]
PROFILE = ROOT / "results" / "gate_profile.csv"


def bins_of(gt, brain, mm):
    """Bin index per pixel (-1 = unused) by distance from the true boundary."""
    din = ndimage.distance_transform_edt(gt) * mm
    dout = ndimage.distance_transform_edt(~gt) * mm
    b = np.full(gt.shape, -1)
    b[~brain] = 5
    b[brain & ~gt & (dout > 15)] = 4
    b[brain & ~gt & (dout > 5) & (dout <= 15)] = 3
    b[brain & ~gt & (dout <= 5)] = 2
    b[gt & (din <= 5)] = 1
    b[gt & (din > 5)] = 0
    return b


@torch.no_grad()
def run(model, ds, variant, profile=None, bs=32):
    gates = [b.gate for b in reversed(model.blocks)]  # finest first
    for k, g in enumerate(gates, 1):
        g.ablate = "mean" if variant == "mean" or variant == f"mean_L{k}" else "ones" if variant == "ones" else None
    rows_out = []
    for case, rows in ds.cases():
        rows = np.sort(rows)
        probs, sums = [], np.zeros((len(gates), len(BINS)))
        counts = np.zeros_like(sums)
        for i in range(0, len(rows), bs):
            x, y = ds.batch(rows[i:i + bs])
            probs.append(torch.sigmoid(model(x))[:, 0].numpy())
            if profile is None:
                continue
            gt = y[:, 0].numpy().astype(bool)
            brain = (x != 0).any(1).numpy()
            alphas = [F.interpolate(g.last_alpha, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0].numpy()
                      for g in gates]
            for j in np.nonzero(gt.any((1, 2)))[0]:
                b = bins_of(gt[j], brain[j], ds.spacing[-1])
                for lv, a in enumerate(alphas):
                    sums[lv] += np.bincount(b[b >= 0], weights=a[j][b >= 0], minlength=len(BINS))
                    counts[lv] += np.bincount(b[b >= 0], minlength=len(BINS))
        if profile is not None:
            for lv in range(len(gates)):
                for bi, name in enumerate(BINS):
                    profile.append({"case": case, "level": lv + 1, "bin": name, "n": int(counts[lv, bi]),
                                    "mean": sums[lv, bi] / counts[lv, bi] if counts[lv, bi] else np.nan})
        p = np.concatenate(probs)
        pred, gt = p >= 0.5, ds.y[rows].astype(bool)
        rows_out.append({"case": case, **volume_metrics(pred, gt, ds.spacing), **error_patterns(pred, gt, ds.spacing)})
    return pd.DataFrame(rows_out)


def main():
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    ds = Slices("test")
    prof = pd.read_csv(PROFILE) if PROFILE.exists() else pd.DataFrame(columns=["model"])
    for name in MODELS:
        out = ROOT / "runs" / name
        variants = ["intact", "mean"] + (["mean_L1", "mean_L2", "mean_L3", "mean_L4", "ones"] if name.endswith("_s0") else [])
        todo = [v for v in variants if not (out / f"ablate_{v}_metrics.csv").exists()]
        if not todo:
            continue
        model = SegNet(name.split("_")[0], True, pretrained=False)
        model.load_state_dict(torch.load(out / "final.pt", weights_only=True))
        model.eval()
        for v in todo:
            print(name, v, flush=True)
            rec = [] if v == "intact" else None
            df = run(model, ds, v, rec)
            df.to_csv(out / f"ablate_{v}_metrics.csv", index=False)
            if rec is not None:
                prof = pd.concat([prof[prof.model != name], pd.DataFrame(rec).assign(model=name)])
                prof.to_csv(PROFILE, index=False)
            print(f"  dice {df.dice.mean():.4f} boundary FP {df.fp_boundary_ml.mean():.3f} "
                  f"distant FP {df.fp_distant_ml.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
