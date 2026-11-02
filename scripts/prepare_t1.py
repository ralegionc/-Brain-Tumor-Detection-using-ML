"""Add the T1w channel to the MSD slice arrays, for models whose inputs must match the LGG
set (pre-contrast T1, FLAIR, post-contrast T1). The tar is streamed from the MSD mirror
and never stored. Each volume goes through exactly the steps of prepare_data.py (same brain
mask, crop, resampling and kept slices); the rebuilt FLAIR channel is checked against the
stored one, so the new channel lines up row for row. Writes data/slices/{split}_t1.bin.
Resumable: finished cases are listed in data/slices/t1_done.txt and skipped.
"""
import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from prepare_data import CHANNELS, CROP, load_nii, split_of

ROOT = Path(__file__).resolve().parent.parent
URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"


def channel(img, brain, c, size):
    v = img[..., c].astype(np.float32)
    v = np.where(brain, np.clip((v - v[brain].mean()) / (v[brain].std() + 1e-6), -5, 5), 0)[CROP, CROP]
    x = torch.from_numpy(v).permute(2, 0, 1)[:, None]
    return F.interpolate(x, size=(size, size), mode="area")[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default=str(ROOT / "data/slices"))
    a = ap.parse_args()
    out = Path(a.out)
    meta = json.load(open(out / "meta.json"))
    s = meta["size"]
    X, T1, IDX = {}, {}, {}
    for sp, n in meta["counts"].items():
        X[sp] = np.memmap(out / f"{sp}_x.bin", dtype=np.float16, mode="r", shape=(n, 3, s, s))
        f = out / f"{sp}_t1.bin"
        T1[sp] = np.memmap(f, dtype=np.float16, mode="r+" if f.exists() else "w+", shape=(n, s, s))
        IDX[sp] = pd.read_csv(out / f"{sp}_index.csv")
    done_f = out / "t1_done.txt"
    done = set(done_f.read_text().split()) if done_f.exists() else set()
    want = {c for sp in IDX for c in IDX[sp].case.unique()}
    print(len(want), "cases,", len(done), "already done", flush=True)

    worst = 0.0
    tf = tarfile.open(fileobj=urllib.request.urlopen(a.url), mode="r|")
    for m in tf:
        if done >= want:
            break
        if not (m.isfile() and "/imagesTr/" in m.name and m.name.endswith(".nii.gz") and "/._" not in m.name):
            continue
        case = Path(m.name).name[:-7]
        if case in done or case not in want:
            continue
        img, _ = load_nii(tf.extractfile(m).read())
        brain = (img[..., CHANNELS] != 0).any(-1)
        b = torch.from_numpy(brain[CROP, CROP].astype(np.float32)).permute(2, 0, 1)[:, None]
        keep = (F.interpolate(b, size=(s, s), mode="area") >= 0.5)[:, 0].flatten(1).float().mean(1) >= 0.05
        zs = torch.nonzero(keep)[:, 0].numpy()
        sp = split_of(case)
        g = IDX[sp][IDX[sp].case == case].sort_values("z")
        assert (g.z.to_numpy() == zs).all(), (case, "kept slices differ")
        rows = g.index.to_numpy()
        flair = channel(img, brain, CHANNELS[0], s)[keep].numpy().astype(np.float16)
        err = float(np.abs(flair.astype(np.float32) - X[sp][rows, 0].astype(np.float32)).max())
        worst = max(worst, err)
        assert err < 1e-2, (case, "FLAIR mismatch", err)
        T1[sp][rows] = channel(img, brain, 1, s)[keep].numpy().astype(np.float16)
        T1[sp].flush()
        done.add(case)
        with open(done_f, "a") as fh:
            fh.write(case + "\n")
        if len(done) % 25 == 0:
            print(len(done), case, sp, len(rows), "slices, worst FLAIR diff", worst, flush=True)
    print("finished", len(done), "of", len(want), "cases; worst FLAIR diff", worst)
    assert done >= want


if __name__ == "__main__":
    main()
