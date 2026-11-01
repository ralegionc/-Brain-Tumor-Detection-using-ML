"""Summarise every finished run and score the pre-registered hypotheses.
Writes results/summary.csv, results/paired.csv and results/verdicts.md."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
B = 5000


def boot_ci(v, stat=np.mean, seed=0):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    s = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(B)])
    return stat(v), *np.percentile(s, [2.5, 97.5])


def load():
    runs = {}
    for d in sorted((ROOT / "runs").iterdir()):
        f = d / "test_metrics.csv"
        if f.exists():
            runs[d.name] = pd.read_csv(f).set_index("case").sort_index()
    return runs


def summary(runs):
    rows = []
    for name, df in runs.items():
        enc, att, *loss, seed = name.split("_")
        r = {"run": name, "encoder": enc, "attention": att == "ag", "loss": "_".join(loss), "seed": int(seed[1:]),
             "n": len(df)}
        for m in ["dice", "iou", "sensitivity", "precision"]:
            r[m], r[m + "_lo"], r[m + "_hi"] = boot_ci(df[m])
        r["hd95_median"], r["hd95_lo"], r["hd95_hi"] = boot_ci(df["hd95"], np.median)
        r["hd95_nan"] = int(df["hd95"].isna().sum())  # empty predictions
        for m in ["fp_distant_ml", "fp_boundary_ml", "fn_boundary_ml"]:
            r[m + "_median"] = df[m].median()
            r[m + "_mean"] = df[m].mean()
        # lesion counts use components >= 0.1 mL; smaller ones are resampling specks
        big = "missed_lesions_big" in df
        r["fp_components_mean"] = df["fp_components_big" if big else "fp_components"].mean()
        r["missed_lesion_rate"] = (df["missed_lesions_big"].sum() / df["gt_lesions_big"].sum() if big
                                   else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def paired(runs, a, b, label):
    """Per-patient differences a - b."""
    x, y = runs[a], runs[b]
    assert (x.index == y.index).all()
    out = {"comparison": label, "a": a, "b": b}
    for m, stat in [("dice", np.mean), ("hd95", np.median), ("fp_distant_ml", np.median),
                    ("fp_components_big", np.mean), ("sensitivity", np.mean), ("precision", np.mean)]:
        if m not in x or m not in y:
            continue
        d = (x[m] - y[m]).to_numpy()
        est, lo, hi = boot_ci(d, stat)
        dd = d[~np.isnan(d)]
        p = wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
        out.update({f"d_{m}": est, f"d_{m}_lo": lo, f"d_{m}_hi": hi, f"p_{m}": p})
    return out


def round1():
    """Names of the 15 round 1 runs, so later rounds never change these results."""
    from run_grid import A, B, C
    return {f"{e}_{'ag' if att else 'plain'}_{l}_s{s}" for e, att, l, s in A + B + C}


def main():
    keep = round1()
    runs = {k: v for k, v in load().items() if k in keep}
    print(len(runs), "finished runs")
    if not runs:
        return
    S = summary(runs)
    S.to_csv(RES / "summary.csv", index=False)
    show = ["run", "dice", "iou", "sensitivity", "precision", "hd95_median", "fp_distant_ml_median",
            "fp_components_mean", "missed_lesion_rate"]
    print(S[show].round(3).to_string(index=False))

    P, lines = [], ["# Verdicts", "", "Scored by scripts/analyze.py against PREREGISTRATION.md.", ""]
    have = lambda *n: all(k in runs for k in n)

    # H1: attention vs plain, matched encoder and seed
    att_pairs = [(f"{e}_ag_dice_bce_s{s}", f"{e}_plain_dice_bce_s{s}", f"{e} s{s}")
                 for e in ["inceptionv3", "vgg16", "vgg19"] for s in [0, 1, 2]]
    att_pairs = [p for p in att_pairs if have(p[0], p[1])]
    for a, b, lab in att_pairs:
        P.append(paired(runs, a, b, "attention " + lab))

    # pooled InceptionV3 seeds: average each patient's difference over seeds
    inc = [(a, b) for a, b, lab in att_pairs if lab.startswith("inceptionv3")]
    pooled = None
    if len(inc) == 3:
        d_fp = np.mean([runs[a]["fp_distant_ml"] - runs[b]["fp_distant_ml"] for a, b in inc], axis=0)
        d_dice = np.mean([runs[a]["dice"] - runs[b]["dice"] for a, b in inc], axis=0)
        pooled = (boot_ci(d_fp, np.median), boot_ci(d_dice))

    # H2: losses vs Dice+BCE on InceptionV3 + gates
    base = "inceptionv3_ag_dice_bce_s0"
    for l in ["bce", "dice", "focal", "tversky", "focal_tversky"]:
        n = f"inceptionv3_ag_{l}_s0"
        if have(n, base):
            P.append(paired(runs, n, base, f"loss {l} vs dice_bce"))
    Pd = pd.DataFrame(P)
    if len(Pd):
        Pd.to_csv(RES / "paired.csv", index=False)

    # seed noise floor
    seeds = S[(S.encoder == "inceptionv3") & (S.loss == "dice_bce")]
    for att in [False, True]:
        g = seeds[seeds.attention == att]
        if len(g) > 1:
            lines.append(f"- Seed spread, InceptionV3 {'with' if att else 'without'} gates: mean Dice "
                         f"{g.dice.min():.3f} to {g.dice.max():.3f} (range {g.dice.max() - g.dice.min():.3f}), "
                         f"median distant FP {g.fp_distant_ml_median.min():.2f} to {g.fp_distant_ml_median.max():.2f} mL")
    lines.append("")

    enc0 = Pd[Pd.comparison.str.match(r"attention .* s0$")] if len(Pd) else Pd
    if len(enc0) == 3:
        neg = (enc0.d_fp_distant_ml < 0).sum()
        ci_ex = (enc0.d_fp_distant_ml_hi < 0).sum()
        ok = neg == 3 and ci_ex >= 2 and pooled is not None and pooled[0][2] < 0
        lines.append(f"**H1a {'SUPPORTED' if ok else 'NOT SUPPORTED'}.** Median per-patient change in distant FP "
                     f"volume (gates minus plain): " + "; ".join(
                         f"{r.comparison[10:]} {r.d_fp_distant_ml:+.2f} mL [{r.d_fp_distant_ml_lo:+.2f}, {r.d_fp_distant_ml_hi:+.2f}]"
                         for r in enc0.itertuples())
                     + (f"; pooled InceptionV3 seeds {pooled[0][0]:+.2f} [{pooled[0][1]:+.2f}, {pooled[0][2]:+.2f}]"
                        if pooled else "; pooled seeds not yet available"))
        gain = (enc0.d_dice >= 0.01).sum()
        ok = gain >= 2 and pooled is not None and pooled[1][0] >= 0.01
        lines.append(f"\n**H1b {'SUPPORTED' if ok else 'NOT SUPPORTED'}.** Mean Dice change: " + "; ".join(
            f"{r.comparison[10:]} {r.d_dice:+.3f} [{r.d_dice_lo:+.3f}, {r.d_dice_hi:+.3f}]" for r in enc0.itertuples())
            + (f"; pooled InceptionV3 seeds {pooled[1][0]:+.3f} [{pooled[1][1]:+.3f}, {pooled[1][2]:+.3f}]"
               if pooled else ""))
    lines.append("")

    L = S[(S.encoder == "inceptionv3") & S.attention & (S.seed == 0)].set_index("loss")
    if {"dice_bce", "tversky", "focal_tversky"} <= set(L.index):
        ok = all(L.loc[l, "sensitivity"] > L.loc["dice_bce", "sensitivity"]
                 and L.loc[l, "precision"] < L.loc["dice_bce", "precision"] for l in ["tversky", "focal_tversky"])
        lines.append(f"**H2a {'SUPPORTED' if ok else 'NOT SUPPORTED'}.** Sensitivity / precision: " + "; ".join(
            f"{l} {L.loc[l, 'sensitivity']:.3f} / {L.loc[l, 'precision']:.3f}" for l in ["dice_bce", "tversky", "focal_tversky"]))
    if len(L) == 6:
        low = L.sensitivity.idxmin()
        lines.append(f"\n**H2b {'SUPPORTED' if low == 'bce' else 'NOT SUPPORTED'}.** Lowest sensitivity: {low} "
                     f"({L.sensitivity.min():.3f}); BCE {L.loc['bce', 'sensitivity']:.3f}.")
    # Not pre-registered: the median distant-FP change is pinned at 0 because a third of
    # patients have no distant false positives at all, so also report means, and split
    # false-positive volume into boundary and distant parts
    if att_pairs:
        lines += ["", "## Exploratory (not pre-registered)", "",
                  "Mean per-patient change, gates minus plain, mL [95% bootstrap CI]:", ""]
        E = []
        for a, b, lab in att_pairs:
            r = {"pair": lab}
            for m in ["fp_distant_ml", "fp_boundary_ml", "fn_boundary_ml"]:
                r[m], r[m + "_lo"], r[m + "_hi"] = boot_ci((runs[a][m] - runs[b][m]).to_numpy())
            E.append(r)
            lines.append(f"- {lab}: distant FP {r['fp_distant_ml']:+.2f} [{r['fp_distant_ml_lo']:+.2f}, "
                         f"{r['fp_distant_ml_hi']:+.2f}]; boundary FP {r['fp_boundary_ml']:+.2f} "
                         f"[{r['fp_boundary_ml_lo']:+.2f}, {r['fp_boundary_ml_hi']:+.2f}]; boundary FN "
                         f"{r['fn_boundary_ml']:+.2f} [{r['fn_boundary_ml_lo']:+.2f}, {r['fn_boundary_ml_hi']:+.2f}]")
        pd.DataFrame(E).to_csv(RES / "exploratory_fp.csv", index=False)
    (RES / "verdicts.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
