"""Round 3 queue (see Round 3 in PREREGISTRATION.md). Needs scripts/prepare_t1.py first.

H9  InceptionV3 plain/gates x seeds 0-2 on MSD with T1w, FLAIR, T1gd inputs, then
    scripts/cross_eval.py (each model on the other dataset)
H8  scripts/gate_probe.py (gate profiles and test-time gate ablations, no training)
Finished runs and passes are skipped, so the queue can be restarted.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.run_round2 import cfg  # noqa: E402
from seg.train import run_name  # noqa: E402

H9 = [cfg("inceptionv3", att, seed=s, data="msd3") for s in [0, 1, 2] for att in [False, True]]


def step(label, cmd, log):
    print(label, flush=True)
    with open(log, "a") as fh:
        r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        print(f"  failed with code {r.returncode}; see {log}", flush=True)
        sys.exit(r.returncode)


def main():
    for c in H9:
        out = ROOT / "runs" / run_name(c)
        if (out / "test_metrics.csv").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "seg.train", "--encoder", c.encoder, "--loss", c.loss,
               "--seed", str(c.seed), "--data", c.data] + (["--attention"] if c.attention else [])
        step(f"[H9] {out.name}", cmd, out / "stdout.log")
    step("[H9] cross_eval", [sys.executable, "-u", "scripts/cross_eval.py"], ROOT / "results" / "cross_eval.log")
    step("[H8] gate_probe", [sys.executable, "-u", "scripts/gate_probe.py"], ROOT / "results" / "gate_probe.log")
    print("queue complete")


if __name__ == "__main__":
    main()
