"""Write enhancing-tumour masks aligned row for row with the existing slice arrays.

prepare_data.py kept only whole tumour (label > 0). This re-reads the label volumes,
applies the same crop, resampling and slice selection, and writes {split}_et.bin
(label 3, enhancing tumour). As a check it rebuilds the whole-tumour mask the same way
and asserts it matches {split}_y.bin exactly, so the rows cannot drift.
"""
import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prepare_data import CROP, ROOT, load_nii, split_of


def resample(mask, size):
    m = torch.from_numpy(mask[CROP, CROP].astype(np.float32)).permute(2, 0, 1)[:, None]
    return (F.interpolate(m, size=(size, size), mode="area") >= 0.5)[:, 0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default=str(ROOT / "data/raw/Task01_BrainTumour.tar"))
    ap.add_argument("--out", default=str(ROOT / "data/slices"))
    a = ap.parse_args()
    out = Path(a.out)
    meta = json.load(open(out / "meta.json"))
    size = meta["size"]

    tf = tarfile.open(a.tar)
    members = {m.name: m for m in tf.getmembers() if m.isfile() and "/._" not in m.name}
    labels = json.loads(tf.extractfile(members["Task01_BrainTumour/dataset.json"]).read())["labels"]
    print("labels:", labels)
    assert labels["3"].lower().startswith("enhancing"), labels

    import pandas as pd
    index = {s: pd.read_csv(out / f"{s}_index.csv") for s in ["train", "val", "test"]}
    ys = {s: np.memmap(out / f"{s}_y.bin", dtype=np.uint8, mode="r", shape=(meta["counts"][s], size, size))
          for s in index}
    files = {s: open(out / f"{s}_et.bin", "wb") for s in index}
    pos = {s: 0 for s in index}
    for s in index:  # cases were written in sorted order within each split
        index[s]["order"] = np.arange(len(index[s]))
    cases = sorted(set().union(*[set(index[s].case) for s in index]))
    for i, case in enumerate(cases):
        s = split_of(case)
        rows = index[s][index[s].case == case]
        assert rows.order.iloc[0] == pos[s], (case, "rows out of order")
        lab, _ = load_nii(tf.extractfile(members[f"Task01_BrainTumour/labelsTr/{case}.nii.gz"]).read())
        zs = rows.z.to_numpy()
        wt, et = resample(lab > 0, size)[zs], resample(lab == 3, size)[zs]
        assert np.array_equal(wt, ys[s][pos[s]:pos[s] + len(zs)].astype(bool)), (case, "whole tumour mismatch")
        files[s].write(np.ascontiguousarray(et.astype(np.uint8)).tobytes())
        pos[s] += len(zs)
        if i % 50 == 0:
            print(i, case, s, flush=True)
    for s, f in files.items():
        f.close()
        assert pos[s] == meta["counts"][s], s
    print("done; every whole-tumour row matched")


if __name__ == "__main__":
    main()
