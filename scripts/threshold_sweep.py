"""H4: is Tversky's extra sensitivity something a lower threshold on Dice+BCE also buys?

For each model, predicts every validation and test volume once and stores, per patient,
histograms of tumour and non-tumour voxel probabilities over 256 bins. Overlap metrics at
any threshold then follow exactly from cumulative sums. t* is picked on validation (the
first threshold, scanning down from 0.5, where Dice+BCE's mean sensitivity reaches
Tversky's at 0.5); test volumes are then re-scored in full at t*, including HD95.
Writes results/threshold_hist.npz, results/threshold_curves.csv and results/h4.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.analyze import boot_ci  # noqa: E402
from seg.data import Slices  # noqa: E402
from seg.metrics import error_patterns, volume_metrics  # noqa: E402
from seg.models import SegNet  # noqa: E402
from seg.train import predict_volume  # noqa: E402

RES = ROOT / "results"
BASE, TV = "inceptionv3_ag_dice_bce_s0", "inceptionv3_ag_tversky_s0"
GRID = np.arange(1, 256)  # threshold k/255: voxel counts as tumour if p*255 >= k


def load(run):
    m = SegNet("inceptionv3", True, pretrained=False)
    m.load_state_dict(torch.load(ROOT / "runs" / run / "final.pt", weights_only=True))
    return m.eval()


def histograms(model, split):
    ds = Slices(split)
    out = {}
    for case, rows in ds.cases():
        p = np.round(predict_volume(model, ds, rows) * 255).astype(np.uint8)
        gt = ds.y[np.sort(rows)].astype(bool)
        out[case] = (np.bincount(p[gt], minlength=256), np.bincount(p[~gt], minlength=256))
    return out


def curves(h):
    """Per-patient sensitivity, precision, Dice at every threshold in GRID."""
    rows = []
    for case, (pos, neg) in h.items():
        tp = np.cumsum(pos[::-1])[::-1][GRID]  # tumour voxels with bin >= k
        fp = np.cumsum(neg[::-1])[::-1][GRID]
        fn = pos.sum() - tp
        with np.errstate(invalid="ignore", divide="ignore"):
            rows.append(pd.DataFrame({"case": case, "k": GRID, "sensitivity": tp / (tp + fn),
                                      "precision": np.where(tp + fp > 0, tp / (tp + fp), np.nan),
                                      "dice": 2 * tp / (2 * tp + fp + fn)}))
    return pd.concat(rows)


def rescore(model, k):
    ds = Slices("test")
    rows = []
    for case, rows_ in ds.cases():
        p = predict_volume(model, ds, rows_)
        pred, gt = np.round(p * 255) >= k, ds.y[np.sort(rows_)].astype(bool)
        rows.append({"case": case, **volume_metrics(pred, gt, ds.spacing), **error_patterns(pred, gt, ds.spacing)})
    return pd.DataFrame(rows).set_index("case").sort_index()


def main():
    f = RES / "threshold_hist.npz"
    if f.exists():
        H = dict(np.load(f, allow_pickle=True))["H"].item()
    else:
        H = {}
        for run in [BASE, TV]:
            m = load(run)
            for split in ["val", "test"]:
                print("predicting", run, split, flush=True)
                H[(run, split)] = histograms(m, split)
        np.savez(f, H=np.array(H, dtype=object))
    C = {key: curves(h) for key, h in H.items()}
    pd.concat([c.assign(run=r, split=s) for (r, s), c in C.items()]).groupby(["run", "split", "k"])[
        ["sensitivity", "precision", "dice"]].mean().reset_index().to_csv(RES / "threshold_curves.csv", index=False)

    half = 128  # p >= 128/255 is the 0.5 threshold used everywhere else
    mean_at = lambda c, k, m: c[c.k == k][m].mean()
    target = mean_at(C[(TV, "val")], half, "sensitivity")
    base_val = C[(BASE, "val")].groupby("k").sensitivity.mean()
    ks = [k for k in range(half, 0, -1) if base_val[k] >= target]
    k_star = ks[0]
    print(f"validation: Tversky sensitivity {target:.4f}; Dice+BCE reaches it at t* = {k_star}/255 = {k_star / 255:.3f}")

    a = rescore(load(BASE), k_star)
    b = pd.read_csv(ROOT / "runs" / TV / "test_metrics.csv").set_index("case").sort_index()
    a.to_csv(RES / "h4_dice_bce_at_tstar.csv")
    d = (a.dice - b.dice).to_numpy()
    est, lo, hi = boot_ci(d)
    sens_gap = a.sensitivity.mean() - b.sensitivity.mean()
    verdict = ("THRESHOLD IS ENOUGH" if lo > -0.005 and abs(sens_gap) <= 0.01 else
               "TVERSKY ADDS SOMETHING" if hi < 0 else "INCONCLUSIVE")
    lines = ["# H4: loss function vs threshold", "",
             f"t* = {k_star}/255 = {k_star / 255:.3f}, picked on validation, where Dice+BCE's mean sensitivity "
             f"reaches Tversky's at 0.5 ({target:.3f}).", "",
             "| Test set (117 patients) | Dice | Sensitivity | Precision | median HD95 (mm) | mean distant FP (mL) |",
             "|---|---|---|---|---|---|"]
    base05 = pd.read_csv(ROOT / "runs" / BASE / "test_metrics.csv")
    for lab, df in [("Dice+BCE at 0.5", base05), (f"Dice+BCE at t* = {k_star / 255:.2f}", a), ("Tversky at 0.5", b)]:
        lines.append(f"| {lab} | {df.dice.mean():.3f} | {df.sensitivity.mean():.3f} | {df.precision.mean():.3f} | "
                     f"{df.hd95.median():.2f} | {df.fp_distant_ml.mean():.2f} |")
    lines += ["", f"Paired Dice, Dice+BCE at t* minus Tversky: {est:+.4f} [{lo:+.4f}, {hi:+.4f}]; "
                  f"sensitivity gap {sens_gap:+.4f}.", "", f"**H4: {verdict}.**"]
    (RES / "h4.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
