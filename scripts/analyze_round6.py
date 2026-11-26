"""Score round 6 (H20 conformal volume ranges, H21 planted tumours) against PREREGISTRATION.md.
Writes results/verdicts_round6.md, results/round6_conformal.csv (per patient intervals) and
results/round6_coverage.csv."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.analyze import boot_ci  # noqa: E402
from scripts.analyze_round2 import fmt  # noqa: E402

RES = ROOT / "results"
ALPHA = 0.1
SETS = {"A": "r4_vgg16_plain_dice_bce_s{}", "B": "msd3-wt_inceptionv3_plain_dice_bce_s{}"}


def per_patient(pattern, fname, split=None):
    """Seed-mean predicted volume, seed spread and true volume per patient."""
    dfs = []
    for s in range(3):
        d = pd.read_csv(ROOT / "runs" / pattern.format(s) / fname)
        if split is not None and "split" in d:
            d = d[d.split.isin(split)]
        dfs.append(d.set_index("case").sort_index())
    pred = np.stack([d.pred_ml.to_numpy() for d in dfs])
    return pd.DataFrame({"gt_ml": dfs[0].gt_ml, "v_hat": pred.mean(0),
                         "sigma": np.log(pred + 1).std(0)}, index=dfs[0].index)


def score(d, norm=False):
    s = np.abs(np.log((d.gt_ml + 1) / (d.v_hat + 1)))
    return s / (d.sigma + 0.05) if norm else s


def qhat(scores):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - ALPHA))
    return np.inf if k > n else np.sort(scores)[k - 1]


def interval(d, q, norm=False):
    h = q * ((d.sigma + 0.05) if norm else 1)
    return (d.v_hat + 1) * np.exp(-h) - 1, (d.v_hat + 1) * np.exp(h) - 1


def covered(d, q, norm=False):
    lo, hi = interval(d, q, norm)
    return (d.gt_ml >= lo) & (d.gt_ml <= hi), hi - lo


def logistic(X, y):
    X = np.column_stack([np.ones(len(X)), X])
    nll = lambda b: np.sum(np.logaddexp(0, X @ b)) - y @ (X @ b) + 1e-3 * b[1:] @ b[1:]
    b = minimize(nll, np.zeros(X.shape[1]), method="BFGS").x
    return lambda Z: 1 / (1 + np.exp(-np.column_stack([np.ones(len(Z)), Z]) @ b))


def weighted_cover(cal, test):
    """Weighted split conformal (Tibshirani et al. 2019) with a domain classifier on unlabelled features."""
    feat = lambda d: np.column_stack([np.log(d.v_hat + 1), d.sigma])
    Xc, Xt = feat(cal), feat(test)
    p = logistic(np.vstack([Xc, Xt]), np.r_[np.zeros(len(Xc)), np.ones(len(Xt))])
    w = lambda X: p(X) / (1 - p(X)) * len(Xc) / len(Xt)
    wc, wt = w(Xc), w(Xt)
    s = score(cal).to_numpy()
    order = np.argsort(s)
    s, wc = s[order], wc[order]
    cov, width = [], []
    for j in range(len(test)):
        cum = np.cumsum(wc) / (wc.sum() + wt[j])
        i = np.searchsorted(cum, 1 - ALPHA)
        q = s[i] if i < len(s) else np.inf
        c, wd = covered(test.iloc[[j]], q)
        cov.append(bool(c.iloc[0]))
        width.append(float(wd.iloc[0]))
    return np.array(cov), np.array(width)


def h20(L):
    A_val, A_test = per_patient(SETS["A"], "val_metrics.csv"), per_patient(SETS["A"], "test_metrics.csv")
    B_val, B_test = per_patient(SETS["B"], "val_metrics.csv"), per_patient(SETS["B"], "test_metrics.csv")
    B_lgg = per_patient(SETS["B"], "lgg_all_metrics.csv")
    rows, out = [], []

    def add(label, cal, test, norm=False):
        q = qhat(score(cal, norm).to_numpy())
        c, w = covered(test, q, norm)
        lo, hi = interval(test, q, norm)
        rows.append({"setting": label, "n_cal": len(cal), "n_test": len(test), "coverage": c.mean(),
                     "q": q, "median_width_ml": np.median(w)})
        out.append(pd.DataFrame({"setting": label, "case": test.index, "gt_ml": test.gt_ml, "v_hat": test.v_hat,
                                 "lo_ml": lo, "hi_ml": hi, "covered": c}))
        return c.mean(), np.median(w)

    a_cov, a_w = add("A: VGG-16 full budget, MSD test", A_val, A_test)
    b_cov, _ = add("B: InceptionV3 600 steps, MSD test", B_val, B_test)
    b_lgg, _ = add("B: InceptionV3 600 steps, LGG (shifted)", B_val, B_lgg)
    an_cov, an_w = add("A, score scaled by seed spread, MSD test", A_val, A_test, norm=True)

    rng = np.random.default_rng(0)
    covs, widths = [], []
    for _ in range(1000):
        idx = rng.permutation(len(B_lgg))
        c, w = covered(B_lgg.iloc[idx[20:]], qhat(score(B_lgg.iloc[idx[:20]]).to_numpy()))
        covs.append(c.mean())
        widths.append(np.median(w))
    rc = float(np.mean(covs))
    rows.append({"setting": "B: LGG, recalibrated on 20 LGG patients (1,000 splits)", "n_cal": 20, "n_test": 90,
                 "coverage": rc, "q": np.nan, "median_width_ml": float(np.median(widths))})
    wcov, wwid = weighted_cover(B_val, B_lgg)
    rows.append({"setting": "B: LGG, weighted conformal (MSD calibration)", "n_cal": len(B_val), "n_test": len(B_lgg),
                 "coverage": wcov.mean(), "q": np.nan, "median_width_ml": float(np.median(wwid))})
    C = pd.DataFrame(rows)
    C.to_csv(RES / "round6_coverage.csv", index=False)
    pd.concat(out).to_csv(RES / "round6_conformal.csv", index=False)

    v = lambda ok: "SUPPORTED" if ok else "NOT SUPPORTED"
    L.append(f"**H20a {v(a_cov >= 0.85)}.** VGG-16 (round 4), calibrated on 53 MSD validation patients: "
             f"{a_cov:.1%} of 117 test patients inside their 90% range (median width {a_w:.1f} mL).\n")
    L.append(f"**H20b {v(b_cov >= 0.85 and b_lgg < 0.80)}.** InceptionV3 (round 3): {b_cov:.1%} on MSD test, "
             f"{b_lgg:.1%} on the 110 LGG patients.\n")
    L.append(f"**H20c {v(rc >= 0.85)}.** Recalibrated on 20 LGG patients: mean coverage {rc:.1%} on the other 90 "
             f"(5th to 95th percentile over splits {np.percentile(covs, 5):.1%} to {np.percentile(covs, 95):.1%}); "
             f"median width {np.median(widths):.1f} mL.\n")
    L.append(f"**H20d {v(wcov.mean() >= 0.85)}.** Weighted conformal with MSD calibration only: {wcov.mean():.1%} on "
             f"LGG (median width {np.median(wwid):.1f} mL).\n")
    L.append(f"**H20e {v(an_cov >= 0.85 and an_w <= 0.9 * a_w)}.** Score scaled by seed spread: coverage {an_cov:.1%}, "
             f"median width {an_w:.1f} mL against {a_w:.1f} mL ({an_w / a_w - 1:+.0%}).\n")
    L.append(C.round(3).to_markdown(index=False))
    L.append("")


def h21(L):
    f = RES / "round6" / "plants.csv"
    if not f.exists():
        L.append("**H21: NOT RUN** (plants.csv missing).\n")
        return
    P = pd.read_csv(f)
    placed = P[P.placed == True].copy()  # noqa: E712
    L.append(f"Planted {len(placed)} of {len(P)} tumours into {P.recipient.nunique()} test patients "
             f"(the rest found no spot that fits).\n")
    real = pd.read_csv(ROOT / "runs" / "r4_vgg16_plain_dice_bce_s0" / "test_metrics.csv")
    edges = np.r_[0, np.percentile(real.gt_ml, [25, 50, 75]), np.inf]
    nat = placed[placed.kind == "native"]
    rb = pd.cut(real.gt_ml, edges)
    nb = pd.cut(nat.planted_ml, edges)
    T = pd.DataFrame({"real_dice": real.groupby(rb, observed=False).dice.mean(),
                      "planted_dice": nat.groupby(nb, observed=False).dice_s0.mean(),
                      "n_planted": nat.groupby(nb, observed=False).size()})
    use = T[T.n_planted >= 5]
    mad = (use.real_dice - use.planted_dice).abs().mean()
    valid = mad <= 0.05
    L.append(f"**H21a {'SUPPORTED' if valid else 'NOT SUPPORTED'}.** Unscaled planted tumours against real ones "
             f"by volume quartile: mean absolute Dice difference {mad:.3f} over {len(use)} bins.\n")
    T.index = T.index.astype(str)
    L.append(T.round(3).to_markdown())
    L.append("")
    tag = "" if valid else " (resting on planted tumours that failed the realism check)"
    sc = placed[placed.kind != "native"]
    per = sc.groupby(["recipient", "target_ml"]).dice_s0.mean().unstack()
    d = (per[80] - per[5]).dropna()
    ci = boot_ci(d)
    L.append(f"**H21b {'SUPPORTED' if ci[0] >= 0.05 and ci[1] > 0 else 'NOT SUPPORTED'}{tag}.** Planted Dice at 80 mL "
             f"minus at 5 mL, per recipient: {fmt(ci)}.\n")
    G = sc.groupby(["target_ml", "kind"]).dice_s0.agg(["mean", "count"]).unstack()
    diffs = []
    for v in sorted(sc.target_ml.unique()):
        p = sc[sc.target_ml == v].pivot_table(index="recipient", columns="kind", values="dice_s0").dropna()
        diffs.append((v, boot_ci(p["small"] - p["large"])))
    ok = all(abs(c[0]) < 0.02 for _, c in diffs)
    L.append(f"**H21c {'SUPPORTED' if ok else 'NOT SUPPORTED'}{tag}.** Natively small minus natively large donors, "
             "same target volume: " + "; ".join(f"{int(v)} mL {fmt(c)}" for v, c in diffs) + ".\n")
    L.append(G.round(3).to_markdown())
    L.append("")
    ten = sc[sc.target_ml == 10]
    rho = stats.spearmanr(ten.depth_mm, ten.dice_s0)[0]
    L.append(f"Descriptive: at 10 mL, Spearman of planted Dice with depth below the brain surface {rho:+.2f} "
             f"(n = {len(ten)}); mean Dice {ten.dice_s0.mean():.3f}, missed entirely (Dice < 0.1) "
             f"{(ten.dice_s0 < 0.1).mean():.1%}.\n")
    extra = [c for c in placed.columns if c.startswith("dice_s") and c != "dice_s0"]
    if extra:
        L.append("Other seeds, mean planted Dice by target volume:\n")
        L.append(sc.groupby("target_ml")[["dice_s0", *extra]].mean().round(3).to_markdown())
        L.append("")


def main():
    L = ["# Round 6 verdicts\n", "Scored by scripts/analyze_round6.py against PREREGISTRATION.md (Round 6).\n"]
    h20(L)
    h21(L)
    text = "\n".join(L)
    (RES / "verdicts_round6.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
