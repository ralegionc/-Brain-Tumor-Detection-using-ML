"""H9: score models on the dataset they were not trained on (see Round 3 in PREREGISTRATION.md).

MSD slices have the head's front on the right, LGG slices have it at the top, so LGG inputs
are rotated 90 degrees clockwise for MSD-trained models (and MSD inputs counter-clockwise for
LGG-trained ones); predictions are rotated back and scored against each dataset's own masks
and voxel size. Left-right is ignored: training flips it at random.
  MSD (T1w, FLAIR, T1gd) models -> all 110 LGG patients: runs/<name>/lgg_all_metrics.csv
  LGG models -> the 117 MSD test patients (same channels): runs/<name>/msd_test_metrics.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from seg.data import DATASETS, Slices  # noqa: E402
from seg.metrics import error_patterns, volume_metrics  # noqa: E402
from seg.models import SegNet  # noqa: E402

name = lambda pre, att, s: f"{pre}inceptionv3_{'ag' if att else 'plain'}_dice_bce_s{s}"
MSD3 = [name("msd3-wt_", a, s) for s in range(3) for a in [0, 1]]
LGG = [name("lgg-wt_", a, s) for s in range(3) for a in [0, 1]]


@torch.no_grad()
def score(model, ds, k, split, bs=32):
    out = []
    for case, rows in ds.cases():
        rows = np.sort(rows)
        probs = []
        for i in range(0, len(rows), bs):
            x, _ = ds.batch(rows[i:i + bs])
            logit = model(torch.rot90(x, k, dims=(-2, -1)))
            probs.append(torch.sigmoid(torch.rot90(logit, -k, dims=(-2, -1)))[:, 0].numpy())
        pred, gt = np.concatenate(probs) >= 0.5, ds.y[rows].astype(bool)
        out.append({"case": case, "split": split, **volume_metrics(pred, gt, ds.spacing),
                    **error_patterns(pred, gt, ds.spacing)})
    return out


def main():
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    jobs = [(n, "lgg_all_metrics.csv", [(s, Slices(s, DATASETS["lgg"])) for s in ["train", "val", "test"]], -1)
            for n in MSD3]
    jobs += [(n, "msd_test_metrics.csv", [("test", Slices("test", DATASETS["msd3"], channels="t1"))], 1) for n in LGG]
    for n, fname, sets, k in jobs:
        f = ROOT / "runs" / n / fname
        if f.exists() or not (ROOT / "runs" / n / "final.pt").exists():
            continue
        m = SegNet("inceptionv3", "_ag_" in n, pretrained=False)
        m.load_state_dict(torch.load(ROOT / "runs" / n / "final.pt", weights_only=True))
        m.eval()
        df = pd.DataFrame([r for s, ds in sets for r in score(m, ds, k, s)])
        df.to_csv(f, index=False)
        print(n, fname, f"dice {df.dice.mean():.4f} distant FP {df.fp_distant_ml.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
