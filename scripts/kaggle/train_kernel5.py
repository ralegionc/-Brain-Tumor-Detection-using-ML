"""Kaggle GPU notebook: the round 5 queue (see Round 5 in PREREGISTRATION.md).

Two lanes, one per T4, take jobs from the queue in the pre-registered order: first the
inference pass over the round 4 VGG-16 plain models (scripts/round5_infer.py), then the budget
series, 2.5D, and the 25% and 10% data runs. A job starts only if its estimated time still fits
in the session. Finished runs found in an attached earlier version of this notebook are copied
over and skipped. Output: /kaggle/working/runs/r5_* and /kaggle/working/round5.
"""
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

LIMIT_H = 11.5
T0 = time.time()
W, BTA = Path("/kaggle/working"), Path("/tmp/bta")
os.environ["TORCH_HOME"] = "/tmp/torch"

# (name, extra seg.train arguments, estimated minutes on a T4; round 4 VGG-16 ran 0.11 s/step)
QUEUE = [("infer", None, 40)]
for steps, warm in [(2000, 30), (6000, 100)]:
    QUEUE += [(f"r5_vgg16_{att}_dice_bce_s{s}_n{steps}",
               ["--steps", str(steps), "--warmup", str(warm)] + (["--attention"] if att == "ag" else []),
               steps * 0.13 / 60 + 6) for s in range(3) for att in ["plain", "ag"]]
QUEUE += [(f"r5_vgg16_plain_dice_bce_s{s}_c1", ["--steps", "24500", "--warmup", "300", "--context", "1"], 70)
          for s in range(3)]
for frac in [25, 10]:
    QUEUE += [(f"r5_vgg16_{att}_dice_bce_s{s}_f{frac}",
               ["--steps", "24500", "--warmup", "300", "--train-frac", str(frac / 100)]
               + (["--attention"] if att == "ag" else []), 58) for s in range(3) for att in ["plain", "ag"]]


def find_dir(test):
    for root, dirs, files in os.walk("/kaggle/input"):
        if test(Path(root), files):
            return Path(root)
    raise SystemExit("input not found")


def setup():
    code = find_dir(lambda r, f: r.name == "scripts" and "round5_infer.py" in f).parent
    slices = find_dir(lambda r, f: "meta.json" in f and "test_x.bin" in f)
    shutil.copytree(code, BTA, ignore=shutil.ignore_patterns("__pycache__"))
    assert "train_patients" in (BTA / "seg" / "data.py").read_text(), "stale code dataset"
    (BTA / "data").mkdir(exist_ok=True)
    os.symlink(slices, BTA / "data" / "slices")
    (W / "runs").mkdir(exist_ok=True)
    os.symlink(W / "runs", BTA / "runs")
    for root, dirs, files in os.walk("/kaggle/input"):  # resume from an earlier version's output
        r = Path(root)
        if "test_metrics.csv" in files and r.name.startswith("r5_") and not (W / "runs" / r.name).exists():
            shutil.copytree(r, W / "runs" / r.name)
            print("carried over", r.name, flush=True)
        if r.name == "round5" and "uncertainty.csv" in files and not (W / "round5").exists():
            shutil.copytree(r, W / "round5")
            print("carried over round5 inference", flush=True)
    subprocess.run(["nvidia-smi"])


lock, todo = threading.Lock(), list(QUEUE)


def run(cmd, log, env):
    with open(log, "a") as f:
        r = subprocess.run(cmd, cwd=BTA, stdout=f, stderr=subprocess.STDOUT, env=env)
    tail = Path(log).read_text().strip().splitlines()[-3:]
    return r.returncode, tail


def lane(gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    while True:
        with lock:
            if not todo:
                return
            name, args, est = todo.pop(0)
        if (time.time() - T0) / 60 + est * 1.15 > LIMIT_H * 60:
            print(f"[gpu{gpu}] no time left for {name}", flush=True)
            continue  # a shorter job further down could still fit
        t = time.time()
        if name == "infer":
            if (W / "round5" / "uncertainty.csv").exists():
                continue
            r4 = [find_dir(lambda r, f, s=s: r.name == f"r4_vgg16_plain_dice_bce_s{s}" and "final.pt" in f)
                  for s in range(3)]
            (W / "round5").mkdir(exist_ok=True)
            print(f"[gpu{gpu}] start inference", flush=True)
            code, tail = run([sys.executable, "-u", "scripts/round5_infer.py", *map(str, r4), "--out", str(W / "round5")],
                             W / "round5" / "stdout.log", env)
        else:
            d = W / "runs" / name
            if (d / "test_metrics.csv").exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            print(f"[gpu{gpu}] start {name}", flush=True)
            cmd = [sys.executable, "-u", "-m", "seg.train", "--encoder", "vgg16", "--loss", "dice_bce",
                   "--seed", name.split("_s")[1][0], "--val-every", "500", "--ckpt-every", "1000", "--threads", "2",
                   "--amp", "--ram", "--out", f"runs/{name}", *args]
            code, tail = run(cmd, d / "stdout.log", env)
        print(f"[gpu{gpu}] {name} exit {code} after {(time.time() - t) / 60:.1f} min (estimate {est:.0f})",
              *tail, sep="\n  ", flush=True)


def main():
    setup()
    lanes = [threading.Thread(target=lane, args=(g,)) for g in range(2)]
    for t in lanes:
        t.start()
    for t in lanes:
        t.join()
    done = sorted(p.name for p in (W / "runs").iterdir() if (p / "test_metrics.csv").exists())
    print(f"finished {len(done)} runs in {(time.time() - T0) / 3600:.2f} h:", *done, sep="\n  ")
    print("inference:", "done" if (W / "round5" / "uncertainty.csv").exists() else "MISSING")


if __name__ == "__main__":
    main()
