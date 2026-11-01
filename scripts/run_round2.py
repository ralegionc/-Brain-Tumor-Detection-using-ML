"""Round 2 queue (see the Round 2 section of PREREGISTRATION.md). Same rules as
run_grid.py: sequential, finished runs skipped, interrupted runs resume.

H4  threshold sweep on two round 1 models                       (no training)
H5  enhancing tumour: InceptionV3 plain/gates x Dice+BCE/Tversky, plus seed 1 of Dice+BCE
H6  LGG scans with the skull kept: InceptionV3 plain/gates x seeds 0-2
H7  InceptionV3 with 2x upsampled input; VGG-16 plain/gates seeds 1-2
"""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from seg.train import run_name  # noqa: E402


def cfg(enc, att, loss="dice_bce", seed=0, data="msd", target="wt", in_scale=1):
    return SimpleNamespace(encoder=enc, attention=att, loss=loss, seed=seed, data=data, target=target,
                           in_scale=in_scale)


H5 = ([cfg("inceptionv3", att, loss, target="et") for loss in ["dice_bce", "tversky"] for att in [False, True]]
      + [cfg("inceptionv3", att, seed=1, target="et") for att in [False, True]])
H6 = [cfg("inceptionv3", att, seed=s, data="lgg") for s in [0, 1, 2] for att in [False, True]]
H7 = [cfg("inceptionv3", False, in_scale=2)] + [cfg("vgg16", att, seed=s) for s in [1, 2] for att in [False, True]]


def main():
    only = sys.argv[1:]  # optional: e.g. "H5 H6"
    if not only or "H4" in only:
        if not (ROOT / "results" / "h4.md").exists():
            print("[H4] threshold sweep", flush=True)
            with open(ROOT / "results" / "threshold_sweep.log", "a") as log:
                r = subprocess.run([sys.executable, "-u", "scripts/threshold_sweep.py"], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT)
            if r.returncode:
                print(f"  failed with code {r.returncode}", flush=True)
                sys.exit(r.returncode)
    queue = [(p, c) for p, cs in [("H5", H5), ("H6", H6), ("H7", H7)] for c in cs if not only or p in only]
    for phase, c in queue:
        name = run_name(c)
        out = ROOT / "runs" / name
        if (out / "test_metrics.csv").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "seg.train", "--encoder", c.encoder, "--loss", c.loss,
               "--seed", str(c.seed), "--data", c.data, "--target", c.target, "--in-scale", str(c.in_scale)]
        if c.attention:
            cmd.append("--attention")
        print(f"[{phase}] {name}", flush=True)
        with open(out / "stdout.log", "a") as log:
            r = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            print(f"  failed with code {r.returncode}; see {out / 'stdout.log'}", flush=True)
            sys.exit(r.returncode)
    print("queue complete")


if __name__ == "__main__":
    main()
