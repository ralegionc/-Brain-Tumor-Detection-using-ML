"""Score the round 2 hypotheses H5-H7 (H4 is scored by threshold_sweep.py).
Writes results/round2_summary.csv and results/verdicts_round2.md."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.analyze import boot_ci, load  # noqa: E402

RES = ROOT / "results"


def d(runs, a, b, m="dice"):
    return (runs[a][m] - runs[b][m]).to_numpy()


def pooled(runs, pairs, m):
    """Per-patient a - b, averaged over seed pairs."""
    return np.mean([d(runs, a, b, m) for a, b in pairs], axis=0)


def fmt(ci, p=3):
    return f"{ci[0]:+.{p}f} [{ci[1]:+.{p}f}, {ci[2]:+.{p}f}]"


def main():
    runs = load()
    have = lambda *n: all(k in runs for k in n)
    rows = []
    for name, df in runs.items():
        if name.startswith(("msd-et_", "lgg-wt_")) or name.endswith("_x2") or name.startswith("vgg16") and "_s0" not in name:
            rows.append({"run": name, "n": len(df), "dice": df.dice.mean(), "sensitivity": df.sensitivity.mean(),
                         "precision": df.precision.mean(), "hd95_median": df.hd95.median(),
                         "fp_distant_ml_mean": df.fp_distant_ml.mean(), "fp_boundary_ml_mean": df.fp_boundary_ml.mean(),
                         "empty_gt": int((df.gt_ml == 0).sum())})
    S = pd.DataFrame(rows, columns=["run"] + (list(rows[0])[1:] if rows else []))
    S.to_csv(RES / "round2_summary.csv", index=False)
    print(S.round(3).to_string(index=False))
    L = ["# Round 2 verdicts", "", "Scored by scripts/analyze_round2.py against PREREGISTRATION.md (Round 2).", ""]

    # H5: enhancing tumour
    e = lambda att, loss="dice_bce", s=0: f"msd-et_inceptionv3_{'ag' if att else 'plain'}_{loss}_s{s}"
    if have(*(e(a, l) for a in [0, 1] for l in ["dice_bce", "tversky"]), e(0, s=1), e(1, s=1)):
        gate = {l: boot_ci(d(runs, e(1, l), e(0, l))) for l in ["dice_bce", "tversky"]}
        gpool = boot_ci(pooled(runs, [(e(1, s=s), e(0, s=s)) for s in [0, 1]], "dice"))
        ok = all(g[0] >= 0.01 for g in gate.values()) and gpool[0] >= 0.01
        L.append(f"**H5a {'SUPPORTED' if ok else 'NOT SUPPORTED'}.** Enhancing-tumour Dice, gates minus plain: "
                 f"Dice+BCE {fmt(gate['dice_bce'])}; Tversky {fmt(gate['tversky'])}; Dice+BCE seed-pooled {fmt(gpool)}.")
        tv = {a: boot_ci(d(runs, e(a, "tversky"), e(a))) for a in [0, 1]}
        ok = all(t[1] > 0 for t in tv.values())
        sens = {a: runs[e(a, "tversky")].sensitivity.mean() - runs[e(a)].sensitivity.mean() for a in [0, 1]}
        L.append(f"\n**H5b {'SUPPORTED' if ok else 'NOT SUPPORTED'}.** Enhancing-tumour Dice, Tversky minus Dice+BCE: "
                 f"plain {fmt(tv[0])}; gated {fmt(tv[1])}. Sensitivity change: plain {sens[0]:+.3f}, gated {sens[1]:+.3f}.")
        seed = abs(runs[e(0)].dice.mean() - runs[e(0, s=1)].dice.mean()), abs(runs[e(1)].dice.mean() - runs[e(1, s=1)].dice.mean())
        L.append(f"\nSeed spread (seed 0 vs 1, mean Dice): plain {seed[0]:.3f}, gated {seed[1]:.3f}. "
                 f"Test patients with no enhancing tumour: {int((runs[e(0)].gt_ml == 0).sum())} of {len(runs[e(0)])}.")
        L.append("")

    # H6: skull kept
    g = lambda att, s: f"lgg-wt_inceptionv3_{'ag' if att else 'plain'}_dice_bce_s{s}"
    if have(*(g(a, s) for a in [0, 1] for s in [0, 1, 2])):
        pairs = [(g(1, s), g(0, s)) for s in [0, 1, 2]]
        fp = boot_ci(pooled(runs, pairs, "fp_distant_ml"))
        per = [d(runs, a, b, "fp_distant_ml").mean() for a, b in pairs]
        ok = fp[2] < 0 and all(p < 0 for p in per)
        L.append(f"**H6a {'SUPPORTED' if ok else 'NOT SUPPORTED'}.** LGG, mean change in distant FP volume "
                 f"(nominal mL), gates minus plain, seed-pooled: {fmt(fp, 2)}; per seed "
                 + ", ".join(f"{p:+.2f}" for p in per) + ".")
        dc = boot_ci(pooled(runs, pairs, "dice"))
        L.append(f"\n**H6b {'SUPPORTED' if dc[0] >= 0.01 else 'NOT SUPPORTED'}.** LGG mean Dice change, seed-pooled: {fmt(dc)}.")
        base = np.mean([runs[g(0, s)].fp_distant_ml.mean() for s in [0, 1, 2]])
        bnd = boot_ci(pooled(runs, pairs, "fp_boundary_ml"))
        L.append(f"\nExploratory: plain-model mean distant FP {base:.2f} mL per patient; boundary FP change {fmt(bnd, 2)}.")
        L.append("")

    # H7: encoders replicated, and InceptionV3 at 2x input
    v = lambda enc, att, s: f"{enc}_{'ag' if att else 'plain'}_dice_bce_s{s}"
    if have(*(v("vgg16", a, s) for a in [0, 1] for s in [0, 1, 2])):
        res = {}
        for a in [0, 1]:
            vg = np.mean([runs[v("vgg16", a, s)].dice for s in range(3)], axis=0)
            ic = np.mean([runs[v("inceptionv3", a, s)].dice for s in range(3)], axis=0)
            res[a] = boot_ci(vg - ic)
        ok = all(r[1] > 0 for r in res.values())
        L.append(f"**H7a {'SUPPORTED' if ok else 'NOT SUPPORTED'}.** VGG-16 minus InceptionV3 Dice, 3 seeds each: "
                 f"plain {fmt(res[0])}; gated {fmt(res[1])}.")
        h1 = boot_ci(pooled(runs, [(v("vgg16", 1, s), v("vgg16", 0, s)) for s in range(3)], "dice"))
        L.append(f"\nVGG-16 gates minus plain, seed-pooled Dice (H1 recheck, descriptive): {fmt(h1)}.")
    x2 = "inceptionv3_plain_dice_bce_s0_x2"
    if x2 in runs:
        dx = runs[x2].dice.mean()
        L.append(f"\n**H7b {'SUPPORTED' if dx >= 0.877 else 'NOT SUPPORTED'}.** InceptionV3 at 2x input: Dice {dx:.3f} "
                 f"(threshold 0.877; 1x was 0.868, VGG-16 0.885). Paired vs 1x: "
                 f"{fmt(boot_ci(d(runs, x2, 'inceptionv3_plain_dice_bce_s0')))}.")
    (RES / "verdicts_round2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
