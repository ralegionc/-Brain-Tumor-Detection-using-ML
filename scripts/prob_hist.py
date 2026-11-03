"""Per-patient histograms of predicted tumour probability, for scoring any threshold exactly
(H4, H11). For each run directory given, writes {split}_hist.npz with, per patient, counts of
tumour and non-tumour voxels in 256 probability bins (bin k holds p*255 rounded to k).

    python scripts/prob_hist.py runs/r4_vgg16_plain_dice_bce_s0 [more run dirs]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from seg.data import Slices  # noqa: E402
from seg.models import SegNet  # noqa: E402
from seg.train import DEVICE, predict_volume  # noqa: E402


def main():
    for d in map(Path, sys.argv[1:]):
        cfg = json.load(open(d / "config.json"))
        m = SegNet(cfg["encoder"], cfg["attention"], pretrained=False, in_scale=cfg.get("in_scale", 1))
        m.load_state_dict(torch.load(d / "final.pt", weights_only=True, map_location="cpu"))
        m.to(DEVICE).eval()
        for split in ["val", "test"]:
            if (d / f"{split}_hist.npz").exists():
                continue
            ds = Slices(split)
            cases, pos, neg = [], [], []
            for case, rows in ds.cases():
                p = np.round(predict_volume(m, ds, rows) * 255).astype(np.uint8)
                gt = ds.y[np.sort(rows)].astype(bool)
                cases.append(case)
                pos.append(np.bincount(p[gt], minlength=256))
                neg.append(np.bincount(p[~gt], minlength=256))
            np.savez(d / f"{split}_hist.npz", cases=np.array(cases), pos=np.array(pos), neg=np.array(neg))
            print(d.name, split, len(cases), "patients", flush=True)


if __name__ == "__main__":
    main()
