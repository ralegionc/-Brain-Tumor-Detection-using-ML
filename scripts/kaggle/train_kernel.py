"""Kaggle GPU notebook: the round 4 queue (see Round 4 in PREREGISTRATION.md).

Two lanes, one per T4, take runs from the queue in order. Each run trains with seg.train,
then scripts/prob_hist.py stores per-patient probability histograms for threshold analysis.
Finished runs found in an attached earlier version of this notebook are copied over and
skipped, and no run starts unless the slowest run so far would still fit in the session.
Output: /kaggle/working/runs/r4_*.
"""
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

SMOKE = False  # True: 200 steps, 2 test patients, three runs, to measure speed
STEPS, WARMUP, LIMIT_H = (200, 20, 2) if SMOKE else (24500, 300, 11.5)
T0 = time.time()

QUEUE = ([("vgg16", att, "dice_bce", s, 1) for s in range(3) for att in [False, True]]
         + [("vgg16", False, "tversky", s, 1) for s in range(3)]
         + [("inceptionv3", False, "dice_bce", s, 1) for s in range(3)]
         + [("inceptionv3", False, "dice_bce", s, 2) for s in range(3)]
         + [("inceptionv3", True, "dice_bce", s, 2) for s in range(3)])
if SMOKE:
    QUEUE = [QUEUE[0], QUEUE[1], ("inceptionv3", False, "dice_bce", 0, 2)]

W, BTA = Path("/kaggle/working"), Path("/tmp/bta")
os.environ["TORCH_HOME"] = "/tmp/torch"


def find_dir(test):
    for root, dirs, files in os.walk("/kaggle/input"):
        if test(Path(root), files):
            return Path(root)
    raise SystemExit("input not found")


def name_of(enc, att, loss, seed, scale):
    tag = "smoke_" if SMOKE else "r4_"
    return f"{tag}{enc}_{'ag' if att else 'plain'}_{loss}_s{seed}" + ("" if scale == 1 else f"_x{scale}")


def setup():
    code = find_dir(lambda r, f: r.name == "scripts" and "prepare_data.py" in f).parent
    slices = find_dir(lambda r, f: "meta.json" in f and "test_x.bin" in f)
    shutil.copytree(code, BTA, ignore=shutil.ignore_patterns("__pycache__"))
    # an older code dataset version kept the training arrays memory-mapped (3 s per batch on Kaggle)
    assert "np.array(train.x)" in (BTA / "seg" / "train.py").read_text(), "stale code dataset"
    (BTA / "data").mkdir(exist_ok=True)
    os.symlink(slices, BTA / "data" / "slices")
    (W / "runs").mkdir(exist_ok=True)
    os.symlink(W / "runs", BTA / "runs")
    for root, dirs, files in os.walk("/kaggle/input"):  # resume from an earlier version's output
        r = Path(root)
        if "test_metrics.csv" in files and r.name.startswith("r4_") and not (W / "runs" / r.name).exists():
            shutil.copytree(r, W / "runs" / r.name)
            print("carried over", r.name, flush=True)
    subprocess.run(["nvidia-smi"])


lock, todo, durations = threading.Lock(), list(QUEUE), []


def lane(gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    while True:
        with lock:
            if not todo:
                return
            cfg = todo.pop(0)
            need = max(durations) * 1.1 if durations else 0
        enc, att, loss, seed, scale = cfg
        name = name_of(*cfg)
        d = W / "runs" / name
        if not (d / "test_metrics.csv").exists():
            if (time.time() - T0) + need > LIMIT_H * 3600:
                print(f"[gpu{gpu}] no time left for {name}", flush=True)
                return
            d.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-u", "-m", "seg.train", "--encoder", enc, "--loss", loss, "--seed", str(seed),
                   "--in-scale", str(scale), "--steps", str(STEPS), "--warmup", str(WARMUP), "--val-every", "1000",
                   "--ckpt-every", "1000", "--threads", "2", "--amp", "--ram", "--out", f"runs/{name}"]
            cmd += (["--attention"] if att else []) + (["--eval-limit", "2", "--val-every", "100"] if SMOKE else [])
            t = time.time()
            print(f"[gpu{gpu}] start {name}", flush=True)
            with open(d / "stdout.log", "a") as log:
                r = subprocess.run(cmd, cwd=BTA, stdout=log, stderr=subprocess.STDOUT, env=env)
            tail = (d / "stdout.log").read_text().strip().splitlines()[-3:]
            print(f"[gpu{gpu}] {name} exit {r.returncode} after {(time.time() - t) / 60:.1f} min", *tail, sep="\n  ",
                  flush=True)
            if r.returncode:
                continue
            with lock:
                durations.append(time.time() - t)
        if not SMOKE and not (d / "test_hist.npz").exists():
            subprocess.run([sys.executable, "-u", "scripts/prob_hist.py", str(d)], cwd=BTA, env=env)


def main():
    setup()
    lanes = [threading.Thread(target=lane, args=(g,)) for g in range(2)]
    for t in lanes:
        t.start()
    for t in lanes:
        t.join()
    done = sorted(p.name for p in (W / "runs").iterdir() if (p / "test_metrics.csv").exists())
    print(f"finished {len(done)} runs in {(time.time() - T0) / 3600:.2f} h:", *done, sep="\n  ")


if __name__ == "__main__":
    main()
