"""Score the round 4 hypotheses H10-H12 against PREREGISTRATION.md (Round 4), the full-budget
rerun on a Kaggle GPU. Writes results/verdicts_round4.md, results/round4_summary.csv,
results/round4_gains.csv and results/h11_curves.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.analyze import boot_ci, load  # noqa: E402
from scripts.analyze_round2 import fmt, pooled  # noqa: E402

RES = ROOT / "results"
SEEDS = range(3)
HALF = 128  # bin >= 128 is the 0.5 threshold, as in H4


def r4(enc, att, loss="dice_bce", s=0, x2=False):
    return f"r4_{enc}_{'ag' if att else 'plain'}_{loss}_s{s}" + ("_x2" if x2 else "")


def hist_curves(run, split):
    """Per-patient sensitivity and Dice at every threshold k/255, k = 1..255, from prob_hist.py output."""
    h = np.load(ROOT / "runs" / run / f"{split}_hist.npz")
    pos, neg = h["pos"].astype(np.int64), h["neg"].astype(np.int64)
    tp = np.cumsum(pos[:, ::-1], 1)[:, ::-1][:, 1:]  # column k-1 holds voxels with bin >= k
    fp = np.cumsum(neg[:, ::-1], 1)[:, ::-1][:, 1:]
    fn = pos.sum(1, keepdims=True) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        sens, dice = tp / (tp + fn), 2 * tp / (2 * tp + fp + fn)
    idx = pd.Index(h["cases"], name="case")
    cols = pd.RangeIndex(1, 256, name="k")
    return pd.DataFrame(sens, idx, cols).sort_index(), pd.DataFrame(dice, idx, cols).sort_index()


def h10(runs, L):
    for enc, x2, label in [("vgg16", False, "VGG-16"), ("inceptionv3", True, "InceptionV3 at 2x")]:
        pairs = [(r4(enc, 1, s=s, x2=x2), r4(enc, 0, s=s, x2=x2)) for s in SEEDS]
        if not all(a in runs and b in runs for a, b in pairs):
            L.append(f"**H10 {label}: NOT RUN** (runs missing).\n")
            continue
        dc = boot_ci(pooled(runs, pairs, "dice"))
        far = boot_ci(pooled(runs, pairs, "fp_distant_ml"))
        bnd = boot_ci(pooled(runs, pairs, "fp_boundary_ml"))
        per = [(runs[a].fp_distant_ml - runs[b].fp_distant_ml).mean() for a, b in pairs]
        ok_b = far[2] < 0 and all(p < 0 for p in per)
        L.append(f"**H10a {label} {'SUPPORTED' if dc[0] >= 0.01 else 'NOT SUPPORTED'}.** Dice, gates minus plain, "
                 f"seed-pooled: {fmt(dc)}.\n")
        L.append(f"**H10b {label} {'SUPPORTED' if ok_b else 'NOT SUPPORTED'}.** Distant FP, gates minus plain: "
                 f"{fmt(far, 2)} mL; per seed " + ", ".join(f"{p:+.2f}" for p in per) + ".\n")
        L.append(f"**H10c {label} {'SUPPORTED' if bnd[2] < 0 else 'NOT SUPPORTED'}.** Boundary FP (within 5 mm), "
                 f"gates minus plain: {fmt(bnd, 2)} mL.\n")


def h11(L):
    base = [r4("vgg16", 0, s=s) for s in SEEDS]
    tv = [r4("vgg16", 0, "tversky", s) for s in SEEDS]
    if not all((ROOT / "runs" / r / "test_hist.npz").exists() for r in base + tv):
        L.append("**H11: NOT RUN** (histograms missing).\n")
        return
    diffs, gaps, rows, curves = [], [], [], []
    for s, (b, t) in enumerate(zip(base, tv)):
        target = hist_curves(t, "val")[0][HALF].mean()
        bsens = hist_curves(b, "val")[0].mean()
        k = int(max(k for k in bsens.index if bsens[k] >= target))  # highest threshold reaching it
        (bs, bd), (ts, td) = hist_curves(b, "test"), hist_curves(t, "test")
        diffs.append((bd[k] - td[HALF]).to_numpy())
        gaps.append(bs[k].mean() - ts[HALF].mean())
        rows.append({"seed": s, "t_star": k / 255, "val_target_sens": target, "test_dice_bce_at_0.5": bd[HALF].mean(),
                     "test_dice_bce_at_t*": bd[k].mean(), "test_tversky_at_0.5": td[HALF].mean(),
                     "test_sens_dice_bce_at_0.5": bs[HALF].mean(), "test_sens_dice_bce_at_t*": bs[k].mean(),
                     "test_sens_tversky_at_0.5": ts[HALF].mean()})
        curves.append(pd.DataFrame({"seed": s, "k": bd.columns, "dice_bce_sens": bs.mean().values,
                                    "dice_bce_dice": bd.mean().values, "tversky_sens": ts.mean().values,
                                    "tversky_dice": td.mean().values}))
    pd.concat(curves).to_csv(RES / "h11_curves.csv", index=False)
    est, lo, hi = boot_ci(np.mean(diffs, axis=0))
    gap = float(np.mean(gaps))
    verdict = ("THRESHOLD IS ENOUGH" if lo > -0.005 and abs(gap) <= 0.01 else
               "TVERSKY ADDS SOMETHING" if hi < 0 else "INCONCLUSIVE")
    L.append(f"**H11: {verdict}.** VGG-16 plain, Dice+BCE at a per-seed t* minus Tversky at 0.5, seed-pooled "
             f"test Dice {est:+.4f} [{lo:+.4f}, {hi:+.4f}]; mean sensitivity gap {gap:+.4f}.\n")
    L.append(pd.DataFrame(rows).round(3).to_markdown(index=False))
    L.append("")


def h12(runs, L):
    vgg = [r4("vgg16", 0, s=s) for s in SEEDS]
    inc1 = [r4("inceptionv3", 0, s=s) for s in SEEDS]
    inc2 = [r4("inceptionv3", 0, s=s, x2=True) for s in SEEDS]
    if not all(r in runs for r in vgg + inc1):
        L.append("**H12: NOT RUN** (runs missing).\n")
        return
    d = boot_ci(np.mean([(runs[a].dice - runs[b].dice).to_numpy() for a, b in zip(vgg, inc1)], axis=0))
    L.append(f"**H12a {'SUPPORTED' if d[1] > 0 or d[2] < 0 else 'NOT SUPPORTED'}.** Dice, VGG-16 minus InceptionV3 "
             f"at 1x (plain, seed-averaged per patient): {fmt(d)}.\n")
    if not all(r in runs for r in inc2):
        L.append("**H12b: NOT RUN** (runs missing).\n")
        return
    m = lambda rs: np.mean([runs[r].dice.mean() for r in rs])
    gap, closed = m(vgg) - m(inc1), m(inc2) - m(inc1)
    share = closed / gap if gap > 0 else np.nan
    v = "NOT APPLICABLE (no gap)" if gap <= 0 else "SUPPORTED" if share >= 0.5 else "NOT SUPPORTED"
    x = boot_ci(np.mean([(runs[a].dice - runs[b].dice).to_numpy() for a, b in zip(inc2, vgg)], axis=0))
    L.append(f"**H12b {v}.** Seed-mean Dice: VGG-16 {m(vgg):.4f}, InceptionV3 1x {m(inc1):.4f}, 2x {m(inc2):.4f}; "
             f"2x closes {share:.0%} of the {gap:+.4f} gap. InceptionV3 2x minus VGG-16: {fmt(x)}.\n")


def gains(runs, L):
    rows = []
    for name in sorted(runs):
        if not name.startswith("r4_"):
            continue
        df = runs[name]
        old = name[3:]
        r = {"run": name, "dice": df.dice.mean(), "sensitivity": df.sensitivity.mean(),
             "precision": df.precision.mean(), "hd95_median": df.hd95.median(),
             "fp_distant_ml": df.fp_distant_ml.mean(), "fp_boundary_ml": df.fp_boundary_ml.mean()}
        if old in runs:
            r["dice_600_steps"] = runs[old].dice.mean()
            r["dice_gain"], r["gain_lo"], r["gain_hi"] = boot_ci((df.dice - runs[old].dice).to_numpy())
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(RES / "round4_summary.csv", index=False)
    S[["run", "dice_600_steps", "dice", "dice_gain", "gain_lo", "gain_hi"]].dropna().to_csv(
        RES / "round4_gains.csv", index=False)
    L.append("Every round 4 run on the 117 MSD test patients, with its 600-step counterpart where one exists "
             "(descriptive):\n")
    L.append(S.round(3).to_markdown(index=False))
    L.append("")


def main():
    runs = load()
    L = ["# Round 4 verdicts", "", "Scored by scripts/analyze_round4.py against PREREGISTRATION.md (Round 4).", ""]
    h10(runs, L)
    h11(L)
    h12(runs, L)
    gains(runs, L)
    (RES / "verdicts_round4.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
