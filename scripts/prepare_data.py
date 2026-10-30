"""Stream MSD Task01_BrainTumour volumes out of the tar (no extraction) and write
2D axial slice arrays per split.

Input channels: FLAIR, T1gd, T2w (T1w dropped so the 3-channel ImageNet stems fit).
Target: whole tumour (any label > 0: oedema, non-enhancing core, enhancing tumour).
Each channel is z-scored over brain voxels, clipped to [-5, 5], centre-cropped
240 -> 224 and area-resampled to SIZE. Slices with < 5% brain are dropped.
"""
import argparse
import gzip
import io
import json
import tarfile
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
CHANNELS = [0, 2, 3]  # FLAIR, T1gd, T2w
CROP = slice(8, 232)


def load_nii(blob):
    raw = gzip.decompress(blob)
    fh = nib.FileHolder(fileobj=io.BytesIO(raw))
    img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
    return np.asarray(img.dataobj), img.header.get_zooms()


def split_of(case, seed=0):
    # stable, order-independent patient-level split: 20% test, 10% val, 70% train
    u = (zlib.crc32(f"{seed}:{case}".encode()) % 10000) / 10000
    return "test" if u < 0.2 else "val" if u < 0.3 else "train"


def process(img, lab, size):
    img = img[..., CHANNELS].astype(np.float32)          # (240,240,155,3)
    brain = (img != 0).any(-1)
    for c in range(img.shape[-1]):
        v = img[..., c][brain]
        img[..., c] = np.where(brain, np.clip((img[..., c] - v.mean()) / (v.std() + 1e-6), -5, 5), 0)
    img = img[CROP, CROP]                                 # (224,224,155,3)
    lab = (lab[CROP, CROP] > 0).astype(np.float32)
    brain = brain[CROP, CROP]
    x = torch.from_numpy(img).permute(2, 3, 0, 1)         # (Z,3,H,W)
    y = torch.from_numpy(lab).permute(2, 0, 1)[:, None]   # (Z,1,H,W)
    b = torch.from_numpy(brain.astype(np.float32)).permute(2, 0, 1)[:, None]
    x = F.interpolate(x, size=(size, size), mode="area")
    y = (F.interpolate(y, size=(size, size), mode="area") >= 0.5)[:, 0]
    b = (F.interpolate(b, size=(size, size), mode="area") >= 0.5)[:, 0]
    keep = b.flatten(1).float().mean(1) >= 0.05
    zs = torch.nonzero(keep)[:, 0].numpy()
    return x[keep].numpy().astype(np.float16), y[keep].numpy().astype(np.uint8), b[keep].numpy(), zs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default=str(ROOT / "data/raw/Task01_BrainTumour.tar"))
    ap.add_argument("--out", default=str(ROOT / "data/slices"))
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N cases (smoke test)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    tf = tarfile.open(a.tar)
    members = {m.name: m for m in tf.getmembers() if m.isfile() and "/._" not in m.name}
    meta = json.loads(tf.extractfile(members["Task01_BrainTumour/dataset.json"]).read())
    print("modalities:", meta["modality"], "labels:", meta["labels"])
    cases = sorted(Path(n).name[:-7] for n in members if "/imagesTr/" in n and n.endswith(".nii.gz"))
    if a.limit:
        cases = cases[:a.limit]
    print(len(cases), "cases")

    # stream each case straight to disk: holding a split in RAM peaks near 8 GB
    files = {s: (open(out / f"{s}_x.bin", "wb"), open(out / f"{s}_y.bin", "wb")) for s in ["train", "val", "test"]}
    rows = {s: [] for s in files}
    for i, case in enumerate(cases):
        img, zooms = load_nii(tf.extractfile(members[f"Task01_BrainTumour/imagesTr/{case}.nii.gz"]).read())
        lab, _ = load_nii(tf.extractfile(members[f"Task01_BrainTumour/labelsTr/{case}.nii.gz"]).read())
        assert img.shape[:3] == lab.shape, (case, img.shape, lab.shape)
        assert np.allclose(zooms[:3], 1.0), (case, zooms)
        x, y, b, zs = process(img, lab, a.size)
        s = split_of(case)
        files[s][0].write(np.ascontiguousarray(x).tobytes()); files[s][1].write(np.ascontiguousarray(y).tobytes())
        for k, z in enumerate(zs):
            rows[s].append({"case": case, "z": int(z), "tumour_px": int(y[k].sum()), "brain_px": int(b[k].sum())})
        if i % 25 == 0:
            print(i, case, s, len(zs), "slices", flush=True)

    spacing = 224 / a.size  # mm per pixel in-plane (source is 1 mm isotropic)
    counts = {}
    for s, (fx, fy) in files.items():
        fx.close(); fy.close()
        pd.DataFrame(rows[s]).to_csv(out / f"{s}_index.csv", index=False)
        counts[s] = len(rows[s])
        print(s, counts[s], "slices,", len(set(r["case"] for r in rows[s])), "cases")
    json.dump({"size": a.size, "spacing_mm": [spacing, spacing, 1.0], "channels": ["FLAIR", "T1gd", "T2w"],
               "counts": counts, "x_dtype": "float16", "y_dtype": "uint8"}, open(out / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
