"""Run the whole study as a sequential queue. Safe to re-run at any time: finished
runs are skipped and an interrupted run resumes from its last checkpoint.

A  encoder x attention, Dice+BCE, seed 0                      (6 runs)
B  loss sweep on InceptionV3 + attention gates, seed 0        (5 runs)
C  seed replicates of InceptionV3 with / without gates        (4 runs)
InceptionV3 carries B and C because it trains ~2.5x faster than VGG on this CPU.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

A = [(e, att, "dice_bce", 0) for e in ["inceptionv3", "vgg16", "vgg19"] for att in [False, True]]
B = [("inceptionv3", True, l, 0) for l in ["bce", "dice", "focal", "tversky", "focal_tversky"]]
C = [("inceptionv3", att, "dice_bce", s) for s in [1, 2] for att in [False, True]]


def main():
    only = sys.argv[1:]  # optional: phase letters to run, e.g. "A"
    queue = [(p, c) for p, cs in zip("ABC", [A, B, C]) for c in cs if not only or p in only]
    for phase, (enc, att, loss, seed) in queue:
        name = f"{enc}_{'ag' if att else 'plain'}_{loss}_s{seed}"
        out = ROOT / "runs" / name
        if (out / "test_metrics.csv").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "seg.train", "--encoder", enc, "--loss", loss, "--seed", str(seed)]
        if att:
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
