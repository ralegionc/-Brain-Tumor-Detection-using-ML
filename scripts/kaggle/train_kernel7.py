"""Kaggle GPU notebook: round 7 (see Round 7 in PREREGISTRATION.md).

At the start, two CPU processes build the planted-slice banks (scripts/build_bank.py) from the
training patients. Two lanes, one per T4, then take jobs in order: the round 6 planting grid
for the round 4 baseline models (needs no bank, so it runs while the banks build), the three
small-bank runs, the three native-bank runs, and the planting grid for each new arm. A job that
needs a bank waits for it. Finished jobs found in an attached earlier version are carried over.
Output: /kaggle/working/runs/r7_*, /kaggle/working/plants/plants_{B,S,N}.csv, bank metadata.
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
TRAIN = ["--steps", "24500", "--warmup", "300", "--bank-frac", "0.25"]
RUN = "r7_vgg16_plain_dice_bce_s{}_bank_{}"

# (name, kind, bank needed, extra arguments, estimated minutes on a T4)
QUEUE = [("plants_B", "plant", None, ["--pattern", "r4_vgg16_plain_dice_bce_s{}"], 90)]
QUEUE += [(RUN.format(s, "small"), "train", "small", [], 55) for s in range(3)]
QUEUE += [(RUN.format(s, "native"), "train", "native", [], 55) for s in range(3)]
QUEUE += [("plants_S", "plant", None, ["--pattern", RUN.format("{}", "small")], 90),
          ("plants_N", "plant", None, ["--pattern", RUN.format("{}", "native")], 90)]


def find_dir(test):
    for root, dirs, files in os.walk("/kaggle/input"):
        if test(Path(root), files):
            return Path(root)
    raise SystemExit("input not found")


banks = {}


def setup():
    code = find_dir(lambda r, f: r.name == "scripts" and "build_bank.py" in f).parent
    slices = find_dir(lambda r, f: "meta.json" in f and "test_x.bin" in f)
    shutil.copytree(code, BTA, ignore=shutil.ignore_patterns("__pycache__"))
    assert "--bank" in (BTA / "seg" / "train.py").read_text(), "stale code dataset"
    (BTA / "data").mkdir(exist_ok=True)
    os.symlink(slices, BTA / "data" / "slices")
    (W / "runs").mkdir(exist_ok=True)
    os.symlink(W / "runs", BTA / "runs")
    (W / "plants").mkdir(exist_ok=True)
    (BTA / "results").mkdir(exist_ok=True)
    os.symlink(W / "plants", BTA / "results" / "round6")
    for s in range(3):  # baseline weights from the round 4 notebook output
        r4 = find_dir(lambda r, f, s=s: r.name == f"r4_vgg16_plain_dice_bce_s{s}" and "final.pt" in f)
        (W / "runs" / r4.name).mkdir(exist_ok=True)
        os.symlink(r4 / "final.pt", W / "runs" / r4.name / "final.pt")
    for root, dirs, files in os.walk("/kaggle/input"):  # resume from an earlier version's output
        r = Path(root)
        if "test_metrics.csv" in files and r.name.startswith("r7_") and not (W / "runs" / r.name).exists():
            shutil.copytree(r, W / "runs" / r.name)
            print("carried over", r.name, flush=True)
        for f in files:
            if f.startswith("plants_") and f.endswith(".csv") and r.name == "plants":
                shutil.copy(r / f, W / "plants" / f)  # plant.py resumes from a partial file
                print("carried over", f, flush=True)
    for kind in ["small", "native"]:
        log = open(W / f"bank_{kind}.log", "w")
        banks[kind] = subprocess.Popen([sys.executable, "-u", "scripts/build_bank.py", kind, f"/tmp/bank_{kind}"],
                                       cwd=BTA, stdout=log, stderr=subprocess.STDOUT)
    subprocess.run(["nvidia-smi"])


def bank_ready(kind):
    """Wait until the bank is built; False if its build failed."""
    while banks[kind].poll() is None:
        time.sleep(30)
    if banks[kind].returncode != 0:
        return False
    shutil.copy(f"/tmp/bank_{kind}/meta.json", W / f"bank_{kind}_meta.json")
    shutil.copy(f"/tmp/bank_{kind}/train_index.csv", W / f"bank_{kind}_index.csv")
    return True


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
            # a plant job for an arm waits until that arm's three runs are finished or failed
            name, kind, bank, args, est = todo.pop(0)
        if (time.time() - T0) / 60 + est * 1.15 > LIMIT_H * 60:
            print(f"[gpu{gpu}] no time left for {name}", flush=True)
            continue
        t = time.time()
        if kind == "plant":
            arm = name[-1]
            if arm != "B":
                runs = [W / "runs" / args[1].format(s) for s in range(3)]
                while any(not (r / "final.pt").exists() and not (r / "FAILED").exists() for r in runs):
                    time.sleep(60)
                if any((r / "FAILED").exists() for r in runs):
                    print(f"[gpu{gpu}] skip {name}: a run failed", flush=True)
                    continue
            print(f"[gpu{gpu}] start {name}", flush=True)
            code, tail = run([sys.executable, "-u", "scripts/plant.py", "--seeds", "0", "1", "2", "--threads", "2",
                              "--out", f"{name}.csv", *args], W / "plants" / f"{name}.log", env)
        else:
            d = W / "runs" / name
            if (d / "test_metrics.csv").exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            if not bank_ready(bank):
                (d / "FAILED").touch()
                print(f"[gpu{gpu}] bank {bank} failed, skip {name}", flush=True)
                continue
            print(f"[gpu{gpu}] start {name}", flush=True)
            cmd = [sys.executable, "-u", "-m", "seg.train", "--encoder", "vgg16", "--loss", "dice_bce",
                   "--seed", name.split("_s")[1][0], "--val-every", "500", "--ckpt-every", "1000", "--threads", "2",
                   "--amp", "--ram", "--out", f"runs/{name}", "--bank", f"/tmp/bank_{bank}", *TRAIN, *args]
            code, tail = run(cmd, d / "stdout.log", env)
            if code != 0:
                (d / "FAILED").touch()
        print(f"[gpu{gpu}] {name} exit {code} after {(time.time() - t) / 60:.1f} min (estimate {est:.0f})",
              *tail, sep="\n  ", flush=True)


def main():
    setup()
    lanes = [threading.Thread(target=lane, args=(g,)) for g in range(2)]
    for t in lanes:
        t.start()
    for t in lanes:
        t.join()
    for s in range(3):  # the baseline weights were only linked in; keep them out of the output
        shutil.rmtree(W / "runs" / f"r4_vgg16_plain_dice_bce_s{s}", ignore_errors=True)
    for kind in banks:
        print(f"bank {kind}:", (W / f"bank_{kind}.log").read_text().strip().splitlines()[-1])
    done = sorted(p.name for p in (W / "runs").iterdir() if (p / "test_metrics.csv").exists())
    print(f"finished {len(done)} runs in {(time.time() - T0) / 3600:.2f} h:", *done, sep="\n  ")
    print("plants:", *sorted(p.name for p in (W / "plants").glob("plants_*.csv")))


if __name__ == "__main__":
    main()
