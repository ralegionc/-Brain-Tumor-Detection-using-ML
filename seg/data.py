"""Memory-mapped access to the slice arrays written by scripts/prepare_data.py."""
import json
import math

import numpy as np
import pandas as pd
import torch

from . import ROOT

DATA = ROOT / "data" / "slices"
# msd3: MSD with the LGG channel order (T1w, FLAIR, T1gd); needs scripts/prepare_t1.py
DATASETS = {"msd": DATA, "lgg": ROOT / "data" / "lgg_slices", "msd3": DATA}
CHANNELS = {"msd3": "t1"}


def train_patients(cases, frac):
    """Fixed random subset of training patients, the same for every run; smaller fractions are
    prefixes of larger ones, so the 10% patients are among the 25% patients."""
    cases = np.random.default_rng(2026).permutation(sorted(set(cases)))
    return set(cases[:math.ceil(frac * len(cases))])


class Slices:
    def __init__(self, split, root=DATA, target="wt", channels=None, context=0):
        """target: wt = whole tumour ({split}_y.bin), et = enhancing tumour ({split}_et.bin).
        channels="t1": inputs become T1w, FLAIR, T1gd ({split}_t1.bin plus stored channels 0 and 1).
        context=1: 2.5D, the slices above and below are added as channels (9 in all); at the ends of
        a volume the missing neighbour is the centre slice."""
        meta = json.load(open(root / "meta.json"))
        n, s = meta["counts"][split], meta["size"]
        self.x = np.memmap(root / f"{split}_x.bin", dtype=np.float16, mode="r", shape=(n, 3, s, s))
        self.y = np.memmap(root / (f"{split}_y.bin" if target == "wt" else f"{split}_{target}.bin"), dtype=np.uint8, mode="r", shape=(n, s, s))
        self.t1 = (np.memmap(root / f"{split}_t1.bin", dtype=np.float16, mode="r", shape=(n, s, s))
                   if channels == "t1" else None)
        self.index = pd.read_csv(root / f"{split}_index.csv")
        assert len(self.index) == n
        self.spacing = tuple(meta["spacing_mm"][::-1])  # (dz, dy, dx)
        assert not (context and channels), "2.5D is only set up for the default channels"
        self.context = context
        if context:
            row = {(c, z): i for i, (c, z) in enumerate(zip(self.index.case, self.index.z))}
            self.prev = np.array([row.get((c, z - 1), i) for i, (c, z) in enumerate(zip(self.index.case, self.index.z))])
            self.next = np.array([row.get((c, z + 1), i) for i, (c, z) in enumerate(zip(self.index.case, self.index.z))])

    def __len__(self):
        return len(self.index)

    def batch(self, idx, rng=None):
        """Float tensors (B,3,H,W) and (B,1,H,W). With rng, applies training augmentation:
        left-right flip and per-channel intensity scale/shift inside the brain."""
        idx = np.sort(idx)  # sorted reads are much faster on a memmap
        x = torch.from_numpy(self.x[idx].astype(np.float32))
        if self.t1 is not None:
            x = torch.cat([torch.from_numpy(self.t1[idx].astype(np.float32))[:, None], x[:, :2]], 1)
        if self.context:  # neighbour rows are non-decreasing when idx is sorted, so reads stay sequential
            nb = lambda r: torch.from_numpy(self.x[r].astype(np.float32))
            x = torch.cat([nb(self.prev[idx]), x, nb(self.next[idx])], 1)
        y = torch.from_numpy(self.y[idx].astype(np.float32))[:, None]
        if rng is not None:
            flip = torch.from_numpy(rng.random(len(idx)) < 0.5)
            x[flip] = x[flip].flip(-2)
            y[flip] = y[flip].flip(-2)
            brain = (x != 0).any(1, keepdim=True)
            scale = torch.from_numpy(rng.uniform(0.9, 1.1, (len(idx), 3, 1, 1)).astype(np.float32))
            shift = torch.from_numpy(rng.uniform(-0.1, 0.1, (len(idx), 3, 1, 1)).astype(np.float32))
            if self.context:  # the same jitter for a sequence in all three slices
                scale, shift = scale.repeat(1, 3, 1, 1), shift.repeat(1, 3, 1, 1)
            x = torch.where(brain, x * scale + shift, x)
        return x, y

    def cases(self):
        """(case, row indices sorted by z) for each volume."""
        for case, g in self.index.groupby("case", sort=True):
            yield case, g.sort_values("z").index.to_numpy()
