"""Slice arrays for the LGG segmentation set (Buda et al. 2019), read straight from the
Kaggle zip. Unlike BraTS these scans are not skull-stripped.

Each TIF slice is 256 x 256 x 3 (pre-contrast T1, FLAIR, post-contrast T1). Per patient,
each channel is z-scored over the head (not the brain: there is no brain mask), clipped to
[-5, 5] and set to 0 outside the head, then area-resampled to SIZE. Slices with < 5% head
are dropped. The files carry no pixel spacing, so meta.json records a nominal
1.875 x 1.875 x 3 mm voxel (about a 240 mm field of view, typical TCGA slice spacing).
Output has the same layout as prepare_data.py, so seg.data.Slices reads it unchanged.
"""
import argparse
import io
import json
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
NOMINAL_MM = [1.875, 1.875, 3.0]


def split_of(case, seed=0):
    # patient-level hash split: 30% test, 10% val, 60% train (110 patients in all)
    u = (zlib.crc32(f"{seed}:{case}".encode()) % 10000) / 10000
    return "test" if u < 0.3 else "val" if u < 0.4 else "train"


def read(z, name):
    return np.array(Image.open(io.BytesIO(z.read(name))))


def process(img, lab, size):
    img = img.astype(np.float32)                                        # (Z,256,256,3)
    head = img.mean(-1) > 10
    head = np.stack([ndimage.binary_fill_holes(ndimage.binary_opening(h, iterations=2)) for h in head])
    for c in range(3):
        v = img[..., c][head]
        img[..., c] = np.where(head, np.clip((img[..., c] - v.mean()) / (v.std() + 1e-6), -5, 5), 0)
    x = F.interpolate(torch.from_numpy(img).permute(0, 3, 1, 2), size=(size, size), mode="area")
    y = F.interpolate(torch.from_numpy((lab > 0).astype(np.float32))[:, None], size=(size, size), mode="area")
    b = F.interpolate(torch.from_numpy(head.astype(np.float32))[:, None], size=(size, size), mode="area")
    y, b = (y >= 0.5)[:, 0], (b >= 0.5)[:, 0]
    keep = b.flatten(1).float().mean(1) >= 0.05
    zs = torch.nonzero(keep)[:, 0].numpy()
    return x[keep].numpy().astype(np.float16), y[keep].numpy().astype(np.uint8), b[keep].numpy(), zs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(ROOT / "data/raw/lgg/lgg.zip"))
    ap.add_argument("--out", default=str(ROOT / "data/lgg_slices"))
    ap.add_argument("--size", type=int, default=128)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    z = zipfile.ZipFile(a.zip)
    names = [n for n in z.namelist() if n.startswith("kaggle_3m/TCGA") and n.endswith(".tif")]
    cases = sorted({n.split("/")[1] for n in names})
    print(len(cases), "cases")
    files = {s: (open(out / f"{s}_x.bin", "wb"), open(out / f"{s}_y.bin", "wb")) for s in ["train", "val", "test"]}
    rows = {s: [] for s in files}
    for i, case in enumerate(cases):
        sl = sorted((n for n in names if n.startswith(f"kaggle_3m/{case}/") and not n.endswith("_mask.tif")),
                    key=lambda n: int(n.rsplit("_", 1)[1][:-4]))
        img = np.stack([read(z, n) for n in sl])
        lab = np.stack([read(z, n[:-4] + "_mask.tif") for n in sl])
        assert img.shape[1:] == (256, 256, 3) and lab.shape[1:] == (256, 256), (case, img.shape, lab.shape)
        x, y, b, zs = process(img, lab, a.size)
        s = split_of(case)
        files[s][0].write(np.ascontiguousarray(x).tobytes()); files[s][1].write(np.ascontiguousarray(y).tobytes())
        for k, zz in enumerate(zs):
            rows[s].append({"case": case, "z": int(zz), "tumour_px": int(y[k].sum()), "brain_px": int(b[k].sum())})
        if i % 20 == 0:
            print(i, case, s, len(zs), "slices", flush=True)
    counts = {}
    for s, (fx, fy) in files.items():
        fx.close(); fy.close()
        pd.DataFrame(rows[s]).to_csv(out / f"{s}_index.csv", index=False)
        counts[s] = len(rows[s])
        print(s, counts[s], "slices,", len({r["case"] for r in rows[s]}), "cases")
    json.dump({"size": a.size, "spacing_mm": NOMINAL_MM, "spacing_note": "nominal; source TIFs carry no spacing",
               "channels": ["T1 pre-contrast", "FLAIR", "T1 post-contrast"], "counts": counts,
               "x_dtype": "float16", "y_dtype": "uint8"}, open(out / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
