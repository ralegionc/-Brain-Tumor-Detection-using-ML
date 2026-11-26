"""Round 6 (H20): MSD validation metrics, including predicted volumes, for the two model sets
used to calibrate the conformal intervals. Writes runs/<name>/val_metrics.csv.

    python scripts/round6_val.py [threads]
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from seg.models import SegNet  # noqa: E402
from seg.train import evaluate  # noqa: E402

SETS = {"msd": [f"r4_vgg16_plain_dice_bce_s{s}" for s in range(3)],
        "msd3": [f"msd3-wt_inceptionv3_plain_dice_bce_s{s}" for s in range(3)]}


def main():
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    for data, names in SETS.items():
        for n in names:
            d = ROOT / "runs" / n
            if (d / "val_metrics.csv").exists():
                continue
            enc = "vgg16" if "vgg16" in n else "inceptionv3"
            m = SegNet(enc, False, pretrained=False)
            m.load_state_dict(torch.load(d / "final.pt", weights_only=True, map_location="cpu"))
            with torch.no_grad():
                df = evaluate(m, d, split="val", keep_maps=0, data=data)
            (d / "val_probmaps.npz").unlink(missing_ok=True)
            print(n, f"val dice {df.dice.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
