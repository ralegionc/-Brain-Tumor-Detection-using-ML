"""Kaggle CPU notebook: rebuild the MSD slice arrays from the public tar and check them
against the local arrays (slices.sha256, and rows sampled into the bta-check-r4 dataset).
Output: /kaggle/working/slices."""
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
OUT = Path("/kaggle/working/slices")


def code_dir():
    for root, _, files in os.walk("/kaggle/input"):
        if root.endswith("/scripts") and "prepare_data.py" in files:
            return Path(root).parent
    raise SystemExit("code dataset not attached")


def sha(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main():
    code = code_dir()
    subprocess.run(["df", "-h", "/tmp", "/kaggle/working"])
    tar = Path("/tmp/msd.tar")
    if shutil.disk_usage("/tmp").free < 9e9:
        raise SystemExit("not enough space in /tmp for the 7.6 GB tar")
    print("downloading", URL, flush=True)
    with urllib.request.urlopen(URL) as r, open(tar, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 24)
    print("tar", tar.stat().st_size, "bytes", flush=True)
    subprocess.run([sys.executable, "-u", str(code / "scripts" / "prepare_data.py"), "--tar", str(tar),
                    "--out", str(OUT)], check=True)
    tar.unlink()

    # labels must match byte for byte; index CSVs up to line endings (written on Windows locally);
    # image values up to float rounding, against rows sampled from the local arrays (bta-check-r4)
    want = dict(line.split()[::-1] for line in open(code / "scripts" / "kaggle" / "slices.sha256"))
    bad = []
    for name, h in want.items():
        if name.startswith("test_x_rows"):
            continue
        data = open(OUT / name, "rb").read()
        if name.endswith(".csv"):
            data = data.replace(b"\n", b"\r\n")
        ok = hashlib.sha256(data).hexdigest() == h
        print(name, "OK" if ok else "MISMATCH", flush=True)
        if not ok:
            bad.append(name)
    ref = next((Path(r) for r, _, f in os.walk("/kaggle/input") if "train_rows.npy" in f), None)
    counts = {"train": 39176, "val": 6586, "test": 14529}
    for split, n in counts.items():
        x = np.memmap(OUT / f"{split}_x.bin", dtype=np.float16, mode="r", shape=(n, 3, 128, 128))
        idx = np.load(ref / f"{split}_idx.npy")
        d = np.abs(x[idx].astype(np.float32) - np.load(ref / f"{split}_rows.npy").astype(np.float32))
        print(f"{split}_x: max abs diff {d.max():.5f}, mean {d.mean():.2e}, share differing {(d > 0).mean():.4f}",
              flush=True)
        if d.max() > 0.05:
            bad.append(f"{split}_x")
    if bad:
        raise SystemExit(f"slices differ from the local arrays: {bad}")
    print("slices match the local arrays")


if __name__ == "__main__":
    main()
