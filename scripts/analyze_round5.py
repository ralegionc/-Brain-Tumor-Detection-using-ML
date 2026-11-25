"""Score the round 5 hypotheses H13-H19 against PREREGISTRATION.md (Round 5). Parts whose
inputs are missing (Kaggle runs not downloaded yet) are reported as not run.
Writes results/verdicts_round5.md, results/round5_equivalence.csv, results/round5_budget.csv,
results/round5_calibration.csv and results/round5_failures.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.analyze import boot_ci, load  # noqa: E402
from scripts.analyze_round2 import fmt  # noqa: E402

RES, R5 = ROOT / "results", ROOT / "results" / "round5"
SEEDS = range(3)
MARGIN = 0.005


def pooled(runs, pairs, m):
    return np.mean([(runs[a][m] - runs[b][m]).to_numpy() for a, b in pairs], axis=0)


def ci90(v, seed=0, n=5000):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    s = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return v.mean(), *np.percentile(s, [5, 95])


def gate_pairs(pre, enc="vgg16", post="", n=3):
    return [(f"{pre}{enc}_ag_dice_bce_s{s}{post}", f"{pre}{enc}_plain_dice_bce_s{s}{post}") for s in range(n)]


def h13(runs, L):
    for d in (ROOT / "runs").glob("msd3-wt_*"):
        if (d / "lgg_all_metrics.csv").exists():
            runs["lgg-all_" + d.name] = pd.read_csv(d / "lgg_all_metrics.csv").set_index("case").sort_index()
    settings = [("MSD, InceptionV3, 600 steps", gate_pairs("", "inceptionv3"), "restatement"),
                ("MSD, VGG-16, 600 steps", gate_pairs(""), "restatement"),
                ("Enhancing tumour, InceptionV3, 600 steps", gate_pairs("msd-et_", "inceptionv3", n=2), "restatement"),
                ("Skull kept (LGG), InceptionV3, 600 steps", gate_pairs("lgg-wt_", "inceptionv3"), "restatement"),
                ("MSD-trained on LGG, InceptionV3, 600 steps", gate_pairs("lgg-all_msd3-wt_", "inceptionv3"),
                 "restatement"),
                ("MSD, VGG-16, 2,000 steps", gate_pairs("r5_", post="_n2000"), "test"),
                ("MSD, VGG-16, 6,000 steps", gate_pairs("r5_", post="_n6000"), "test"),
                ("MSD, VGG-16, 24,500 steps", gate_pairs("r4_"), "restatement"),
                ("MSD, InceptionV3 at 2x, 24,500 steps", gate_pairs("r4_", "inceptionv3", "_x2"), "restatement"),
                ("MSD, VGG-16, 25% of patients", gate_pairs("r5_", post="_f25"), "test"),
                ("MSD, VGG-16, 10% of patients", gate_pairs("r5_", post="_f10"), "test")]
    rows = []
    for lab, pairs, kind in settings:
        if not all(a in runs and b in runs for a, b in pairs):
            rows.append({"setting": lab, "kind": kind, "verdict": "not run"})
            continue
        est, lo, hi = ci90(pooled(runs, pairs, "dice"))
        eq = -MARGIN < lo and hi < MARGIN
        rows.append({"setting": lab, "kind": kind, "dice_diff": est, "ci90_lo": lo, "ci90_hi": hi,
                     "verdict": "equivalent" if eq else "not shown equivalent"})
    E = pd.DataFrame(rows)
    E.to_csv(RES / "round5_equivalence.csv", index=False)
    L.append(f"**H13.** Gates minus plain Dice, seed-pooled, 90% CI against a margin of ±{MARGIN} (rows marked "
             "'restatement' were seen before the margin was set):\n")
    L.append(E.round(4).to_markdown(index=False))
    L.append("")


def h14(runs, L):
    budgets = [(600, ""), (2000, "_n2000"), (6000, "_n6000"), (24500, "")]
    pre = {600: "", 2000: "r5_", 6000: "r5_", 24500: "r4_"}
    rows = []
    for steps, post in budgets:
        pairs = gate_pairs(pre[steps], post=post)
        if not all(a in runs and b in runs for a, b in pairs):
            continue
        r = {"steps": steps, "plain_dice": np.mean([runs[b].dice.mean() for _, b in pairs])}
        for m in ["dice", "fp_boundary_ml", "fp_distant_ml"]:
            r[m], r[m + "_lo"], r[m + "_hi"] = boot_ci(pooled(runs, pairs, m))
        rows.append(r)
    B = pd.DataFrame(rows)
    B.to_csv(RES / "round5_budget.csv", index=False)
    if len(B) < 4:
        L.append("**H14: NOT RUN** (budget runs missing).\n")
        return
    at2k = B.set_index("steps").loc[2000]
    mags = B.fp_boundary_ml.abs().to_numpy()
    ok = at2k.fp_boundary_ml_hi < 0 and all(np.diff(mags) <= 0)
    L.append(f"**H14 {'SUPPORTED' if ok else 'NOT SUPPORTED'}.** VGG-16, gates minus plain edge false alarms by "
             "training budget (mL): " + "; ".join(f"{int(r.steps):,} steps {r.fp_boundary_ml:+.2f} "
                                                   f"[{r.fp_boundary_ml_lo:+.2f}, {r.fp_boundary_ml_hi:+.2f}]"
                                                   for r in B.itertuples()) + ".\n")
    L.append(B.round(3).to_markdown(index=False))
    L.append("")


def h15(runs, L):
    for frac in [25, 10]:
        pairs = gate_pairs("r5_", post=f"_f{frac}")
        if not all(a in runs and b in runs for a, b in pairs):
            L.append(f"**H15 {frac}%: NOT RUN** (runs missing).\n")
            continue
        dc = boot_ci(pooled(runs, pairs, "dice"))
        bnd, far = boot_ci(pooled(runs, pairs, "fp_boundary_ml")), boot_ci(pooled(runs, pairs, "fp_distant_ml"))
        plain = np.mean([runs[b].dice.mean() for _, b in pairs])
        full = np.mean([runs[f"r4_vgg16_plain_dice_bce_s{s}"].dice.mean() for s in SEEDS])
        L.append(f"**H15 {frac}% {'SUPPORTED' if dc[0] >= 0.01 else 'NOT SUPPORTED'}.** Gates minus plain, "
                 f"{frac}% of training patients: Dice {fmt(dc)}, edge false alarms {fmt(bnd, 2)} mL, far "
                 f"{fmt(far, 2)} mL. Plain Dice {plain:.3f} against {full:.3f} with all patients.\n")


def calib(pos, neg, probs=None, lo_bin=3):
    """ECE over voxels in histogram bins >= lo_bin (p >= 0.01), 10 equal-width bins of p."""
    k = np.arange(256)
    p = k / 255 if probs is None else probs
    keep = k >= lo_bin
    n, t, p = (pos + neg)[keep], pos[keep], p[keep]
    b = np.minimum((p * 10).astype(int), 9)
    tot = n.sum()
    ece = sum(abs(t[b == j].sum() - (p[b == j] * n[b == j]).sum()) for j in range(10)) / tot
    return ece


def logits():
    p = np.clip(np.arange(256) / 255, 0.5 / 255, 1 - 0.5 / 255)
    return np.log(p / (1 - p))


def fit_T(pos, neg):
    lg = logits()
    def nll(logT):
        q = 1 / (1 + np.exp(-lg / np.exp(logT)))
        return -(pos * np.log(q) + neg * np.log(1 - q)).sum()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def h16(L):
    rows = []
    for s in SEEDS:
        d = ROOT / "runs" / f"r4_vgg16_plain_dice_bce_s{s}"
        v, t = np.load(d / "val_hist.npz"), np.load(d / "test_hist.npz")
        vp, vn, tp, tn = v["pos"].sum(0), v["neg"].sum(0), t["pos"].sum(0), t["neg"].sum(0)
        T = fit_T(vp, vn)
        scaled = 1 / (1 + np.exp(-logits() / T))
        rows.append({"model": f"seed {s}", "T": T, "test_ece": calib(tp, tn), "test_ece_scaled": calib(tp, tn, scaled)})
    if (R5 / "ensemble_test_hist.npz").exists():
        e = np.load(R5 / "ensemble_test_hist.npz")
        ev = np.load(R5 / "ensemble_val_hist.npz")
        T = fit_T(ev["pos"].sum(0), ev["neg"].sum(0))
        rows.append({"model": "3-seed ensemble", "T": T, "test_ece": calib(e["pos"].sum(0), e["neg"].sum(0)),
                     "test_ece_scaled": calib(e["pos"].sum(0), e["neg"].sum(0), 1 / (1 + np.exp(-logits() / T)))})
    C = pd.DataFrame(rows)
    C["ece_cut"] = 1 - C.test_ece_scaled / C.test_ece
    C.to_csv(RES / "round5_calibration.csv", index=False)
    single = C[C.model.str.startswith("seed")]
    L.append(f"**H16a {'SUPPORTED' if (single['T'] > 1).all() else 'NOT SUPPORTED'}.** Temperature fitted on "
             "validation: " + ", ".join(f"{t:.2f}" for t in single["T"]) + " (above 1 means overconfident).\n")
    L.append(f"**H16b {'SUPPORTED' if (single.ece_cut >= 0.5).all() else 'NOT SUPPORTED'}.** Test ECE (voxels with "
             "p >= 0.01) before and after temperature scaling: "
             + "; ".join(f"{r.test_ece:.4f} to {r.test_ece_scaled:.4f}" for r in single.itertuples()) + ".\n")
    L.append(C.round(4).to_markdown(index=False))
    L.append("")
    t = np.load(ROOT / "runs" / "r4_vgg16_plain_dice_bce_s0" / "test_hist.npz")
    pos, neg = t["pos"].sum(0), t["neg"].sum(0)
    p = np.arange(256) / 255
    edges = [0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.01]
    rel = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (p >= lo) & (p < hi)
        n = (pos + neg)[k].sum()
        rel.append({"predicted p": f"{lo:g} to {min(hi, 1):g}", "share of voxels": n / (pos + neg).sum(),
                    "mean predicted": (p[k] * (pos + neg)[k]).sum() / n, "observed tumour share": pos[k].sum() / n,
                    "share of all tumour": pos[k].sum() / pos.sum()})
    L.append("Reliability, seed 0 test set (descriptive):\n")
    L.append(pd.DataFrame(rel).round(4).to_markdown(index=False))
    L.append("")


def h17(L):
    f = R5 / "uncertainty.csv"
    if not f.exists():
        L.append("**H17: NOT RUN** (inference outputs missing).\n")
        return
    U = pd.read_csv(f)
    rng = np.random.default_rng(0)
    for col, tag in [("disagree_frac", "H17a"), ("uncertain_frac", "descriptive")]:
        rho = stats.spearmanr(U[col], U.ensemble_dice)[0]
        bs = [stats.spearmanr(*U.iloc[rng.integers(0, len(U), len(U))][[col, "ensemble_dice"]].T.values)[0]
              for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        flagged = set(U.nlargest(12, col).case)
        worst = set(U.nsmallest(12, "ensemble_dice").case)
        caught = len(flagged & worst)
        if tag == "H17a":
            L.append(f"**H17a {'SUPPORTED' if rho <= -0.4 and hi < 0 else 'NOT SUPPORTED'}.** Spearman, seed "
                     f"disagreement vs ensemble Dice: {rho:+.2f} [{lo:+.2f}, {hi:+.2f}].\n")
            L.append(f"**H17b {'SUPPORTED' if caught >= 6 else 'NOT SUPPORTED'}.** The 12 patients with the most "
                     f"disagreement include {caught} of the 12 with the lowest Dice (chance about 1.2).\n")
        else:
            L.append(f"Descriptive, ensemble uncertain volume instead: Spearman {rho:+.2f} [{lo:+.2f}, {hi:+.2f}], "
                     f"{caught} of the 12 worst caught.\n")
    w = U.nsmallest(12, "ensemble_dice")[["case", "ensemble_dice", "gt_ml", "disagree_frac"]]
    L.append("The 12 lowest-Dice test patients:\n")
    L.append(w.round(3).to_markdown(index=False))
    L.append("")


def h18(runs, L):
    pairs = [(f"r5_vgg16_plain_dice_bce_s{s}_c1", f"r4_vgg16_plain_dice_bce_s{s}") for s in SEEDS]
    if not all(a in runs and b in runs for a, b in pairs):
        L.append("**H18: NOT RUN** (2.5D runs missing).\n")
        return
    dc = boot_ci(pooled(runs, pairs, "dice"))
    L.append(f"**H18 {'SUPPORTED' if dc[0] >= 0.005 and dc[1] > 0 else 'NOT SUPPORTED'}.** 2.5D minus 2D, VGG-16 "
             f"plain, seed-pooled Dice: {fmt(dc)}; 2.5D Dice by seed "
             + ", ".join(f"{runs[a].dice.mean():.4f}" for a, _ in pairs) + ".\n")


def h19(runs, L):
    if not (R5 / "test_ensemble.csv").exists():
        L.append("**H19: NOT RUN** (inference outputs missing).\n")
        return
    rd = lambda n: pd.read_csv(R5 / f"test_{n}.csv").set_index("case").sort_index()
    base = [runs[f"r4_vgg16_plain_dice_bce_s{s}"] for s in SEEDS]
    for tag, name, lab in [("H19a", "flip", "flip averaging"), ("H19b", "clean", "small-component removal")]:
        alt = [rd(f"{name}_s{s}") for s in SEEDS]
        d = np.mean([(a.dice - b.dice).to_numpy() for a, b in zip(alt, base)], axis=0)
        c = boot_ci(d)
        L.append(f"**{tag} {'SUPPORTED' if c[1] > 0 else 'NOT SUPPORTED'}.** {lab} minus plain prediction, "
                 f"seed-pooled Dice: {fmt(c)}.")
    cut = pd.read_json(R5 / "clean_cutoffs.json", typ="series") if (R5 / "clean_cutoffs.json").exists() else None
    if cut is not None:
        L.append(f"Cut-offs picked on validation: {', '.join(f'{v} mL' for v in cut.values)}.")
    ens = rd("ensemble")
    d = ens.dice.to_numpy() - np.mean([b.dice.to_numpy() for b in base], axis=0)
    c = boot_ci(d)
    L.append(f"\n**H19c {'SUPPORTED' if c[1] > 0 else 'NOT SUPPORTED'}.** Three-seed ensemble minus seed-mean single "
             f"model Dice: {fmt(c)} (ensemble {ens.dice.mean():.4f}, median HD95 {ens.hd95.median():.2f} mm).\n")


def failures(runs, L):
    base = [runs[f"r4_vgg16_plain_dice_bce_s{s}"] for s in SEEDS]
    F = base[0][["gt_ml", "gt_lesions_big"]].copy()
    F["dice"] = np.mean([b.dice for b in base], axis=0)
    F["set"] = "MSD test, VGG-16 full budget"
    rows = [F]
    rho_v = stats.spearmanr(F.gt_ml, F.dice)[0]
    rho_l = stats.spearmanr(F.gt_lesions_big, F.dice)[0]
    F["size_q"] = pd.qcut(F.gt_ml, 4, labels=["smallest quarter", "second", "third", "largest quarter"])
    Q = F.groupby("size_q", observed=True).agg(tumour_ml=("gt_ml", "median"), dice=("dice", "mean"),
                                               worst=("dice", "min"))
    multi = F.groupby(F.gt_lesions_big > 1).dice.mean()
    L.append(f"Descriptive, where the full-budget VGG-16 fails (seed-mean Dice per test patient): Spearman with tumour "
             f"volume {rho_v:+.2f}, with number of lesions {rho_l:+.2f}. Single lesion {multi.get(False, np.nan):.3f}, "
             f"several {multi.get(True, np.nan):.3f} mean Dice.\n")
    L.append(Q.round(3).reset_index().to_markdown(index=False))
    L.append("")
    lg = [runs.get(f"lgg-all_msd3-wt_inceptionv3_plain_dice_bce_s{s}") for s in SEEDS]
    if all(x is not None for x in lg):
        G = lg[0][["gt_ml", "gt_lesions_big"]].copy()
        G["dice"] = np.mean([g.dice for g in lg], axis=0)
        G["fp_distant_ml"] = np.mean([g.fp_distant_ml for g in lg], axis=0)
        G["set"] = "LGG, MSD-trained InceptionV3 plain"
        rows.append(G)
        L.append(f"Same for MSD-trained models on the 110 LGG patients: Spearman Dice vs tumour volume "
                 f"{stats.spearmanr(G.gt_ml, G.dice)[0]:+.2f}, vs far false alarms "
                 f"{stats.spearmanr(G.fp_distant_ml, G.dice)[0]:+.2f}; {int((G.dice < 0.3).sum())} patients below "
                 f"Dice 0.3.\n")
    pd.concat(rows).to_csv(RES / "round5_failures.csv")


def main():
    runs = load()
    L = ["# Round 5 verdicts", "", "Scored by scripts/analyze_round5.py against PREREGISTRATION.md (Round 5).", ""]
    h13(runs, L)
    h14(runs, L)
    h15(runs, L)
    h16(L)
    h17(L)
    h18(runs, L)
    h19(runs, L)
    failures(runs, L)
    (RES / "verdicts_round5.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
