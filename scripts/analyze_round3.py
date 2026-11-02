"""Score the round 3 hypotheses H8 and H9 against PREREGISTRATION.md (Round 3).
Writes results/verdicts_round3.md, results/gate_profile_summary.csv, results/ablation_seed0.csv
and results/round3_summary.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.analyze import boot_ci  # noqa: E402
from scripts.analyze_round2 import fmt  # noqa: E402
from seg.data import DATASETS, Slices  # noqa: E402

RES = ROOT / "results"
ENC = {"inceptionv3": "InceptionV3", "vgg16": "VGG-16"}


def csv(run, f):
    p = ROOT / "runs" / run / f
    return pd.read_csv(p).set_index("case").sort_index() if p.exists() else None


def name(pre, att, s):
    return f"{pre}inceptionv3_{'ag' if att else 'plain'}_dice_bce_s{s}"


def flair_copies():
    """LGG patients whose pre- or post-contrast T1 is a copy of FLAIR in the source data."""
    out = set()
    for s in ["train", "val", "test"]:
        d = Slices(s, DATASETS["lgg"])
        for case, rows in d.cases():
            x = d.x[np.sort(rows)].astype(np.float32)
            if min(np.abs(x[:, 0] - x[:, 1]).mean(), np.abs(x[:, 2] - x[:, 1]).mean()) < 1e-3:
                out.add(case)
    return out


def h8(L):
    prof = RES / "gate_profile.csv"
    if not prof.exists():
        return
    P = pd.read_csv(prof)
    P["encoder"] = P.model.str.split("_").str[0]
    S = P.groupby(["encoder", "level", "bin"])["mean"].mean().unstack("bin")
    S = S[[c for c in ["in_deep", "in_edge", "out_edge", "out_mid", "out_far", "outside_brain"] if c in S]]
    S.to_csv(RES / "gate_profile_summary.csv")
    W = P[P.level == 1].pivot_table(index=["model", "case"], columns="bin", values="mean")
    res = {m: boot_ci((g.out_edge - g.out_far).to_numpy()) for m, g in W.groupby(level="model")}
    if len(res) == 6:
        ok = all(r[2] < 0 for r in res.values())
        L.append(f"**H8a {'SUPPORTED' if ok else 'NOT SUPPORTED'}.** Finest gate, mean value 0-5 mm outside the "
                 "tumour minus >15 mm away, per patient: "
                 + "; ".join(f"{m.replace('_ag_dice_bce', '')} {fmt(r)}" for m, r in res.items()) + ".")
    L.append("\nMean gate value by distance from the true boundary (tumour-bearing slices, averaged over seeds):\n")
    L.append(S.round(3).reset_index().to_markdown(index=False))
    L.append("")

    for enc in ENC:
        seeds = [f"{enc}_ag_dice_bce_s{s}" for s in range(3)]
        intact = [csv(m, "ablate_intact_metrics.csv") for m in seeds]
        mean = [csv(m, "ablate_mean_metrics.csv") for m in seeds]
        if any(d is None for d in intact + mean):
            continue
        d = lambda col: np.mean([(b[col] - a[col]).to_numpy() for a, b in zip(intact, mean)], axis=0)
        bnd, far, dice = boot_ci(d("fp_boundary_ml")), boot_ci(d("fp_distant_ml")), boot_ci(d("dice"))
        gap = boot_ci(np.mean([(csv(m.replace("_ag_", "_plain_"), "test_metrics.csv").fp_boundary_ml
                                - a.fp_boundary_ml).to_numpy() for m, a in zip(seeds, intact)], axis=0))
        verdict = "UNINTERPRETABLE" if dice[0] < -0.05 else "SUPPORTED" if bnd[1] > 0 else "NOT SUPPORTED"
        L.append(f"**H8b {ENC[enc]} {verdict}.** Mean-ablated minus intact, pooled over 3 seeds: boundary FP "
                 f"{fmt(bnd, 2)} mL, distant FP {fmt(far, 2)} mL, Dice {fmt(dice)}. For scale, plain minus "
                 f"gated boundary FP is {fmt(gap, 2)} mL.\n")
    rows = []
    for enc in ENC:
        m = f"{enc}_ag_dice_bce_s0"
        base = csv(m, "ablate_intact_metrics.csv")
        for v in ["mean", "mean_L1", "mean_L2", "mean_L3", "mean_L4", "ones"]:
            a = csv(m, f"ablate_{v}_metrics.csv")
            if base is None or a is None:
                continue
            rows.append({"model": m, "ablation": v, **{f"d_{c}": (a[c] - base[c]).mean() for c in
                                                       ["dice", "fp_boundary_ml", "fp_distant_ml", "fn_boundary_ml"]}})
    if rows:
        A = pd.DataFrame(rows)
        A.to_csv(RES / "ablation_seed0.csv", index=False)
        L.append("Seed 0, change from intact under each ablation (descriptive; L1 = finest gate):\n")
        L.append(A.round(3).to_markdown(index=False))
        L.append("")


def h9(L):
    runs = {(a, s): csv(name("msd3-wt_", a, s), "lgg_all_metrics.csv") for a in [0, 1] for s in range(3)}
    if any(v is None for v in runs.values()):
        return
    copies = flair_copies()
    rows = []
    for (a, s), df in runs.items():
        rows.append({"run": name("msd3-wt_", a, s), "on": "LGG, all 110", "dice": df.dice.mean(),
                     "sensitivity": df.sensitivity.mean(), "precision": df.precision.mean(),
                     "fp_distant_ml": df.fp_distant_ml.mean(), "fp_boundary_ml": df.fp_boundary_ml.mean()})
    for label, drop in [("all 110 LGG patients", None), (f"{110 - len(copies)} without FLAIR copies", copies)]:
        sub = {k: (v if drop is None else v[~v.index.isin(drop)]) for k, v in runs.items()}
        pooled = lambda col: np.mean([(sub[(1, s)][col] - sub[(0, s)][col]).to_numpy() for s in range(3)], axis=0)
        fp, dc = boot_ci(pooled("fp_distant_ml")), boot_ci(pooled("dice"))
        per = [(sub[(1, s)].fp_distant_ml - sub[(0, s)].fp_distant_ml).mean() for s in range(3)]
        ok_a = fp[2] < 0 and all(p < 0 for p in per)
        tag = "" if drop is None else " (sensitivity analysis)"
        L.append(f"**H9a{tag} {'SUPPORTED' if ok_a else 'NOT SUPPORTED'}**, {label}. Distant FP, gates minus plain, "
                 f"seed-pooled: {fmt(fp, 2)} mL; per seed " + ", ".join(f"{p:+.2f}" for p in per) + ".")
        L.append(f"\n**H9b{tag} {'SUPPORTED' if dc[0] >= 0.01 else 'NOT SUPPORTED'}**, {label}. Dice, gates minus "
                 f"plain, seed-pooled: {fmt(dc)}.\n")
    # descriptive: shift cost on the 33 LGG test patients, the reverse direction, in-domain check
    for a in [0, 1]:
        tag = "gates" if a else "plain"
        cross = np.mean([runs[(a, s)][runs[(a, s)].split == "test"].dice.mean() for s in range(3)])
        home = np.mean([csv(name("lgg-wt_", a, s), "test_metrics.csv").dice.mean() for s in range(3)])
        back = [csv(name("lgg-wt_", a, s), "msd_test_metrics.csv") for s in range(3)]
        msd_in = np.mean([csv(name("msd3-wt_", a, s), "test_metrics.csv").dice.mean() for s in range(3)])
        r1 = np.mean([csv(name("", a, s), "test_metrics.csv").dice.mean() for s in range(3)])
        txt = f"- {tag}: on the 33 LGG test patients, MSD-trained Dice {cross:.3f} vs LGG-trained {home:.3f}. "
        if all(b is not None for b in back):
            txt += (f"LGG-trained on the 117 MSD test patients {np.mean([b.dice.mean() for b in back]):.3f} vs "
                    f"MSD-trained {msd_in:.3f}. ")
        L.append(txt + f"MSD in-domain with T1w/FLAIR/T1gd inputs {msd_in:.3f} vs round 1 FLAIR/T1gd/T2w {r1:.3f}.")
        for s in range(3):
            b = back[s]
            if b is not None:
                rows.append({"run": name("lgg-wt_", a, s), "on": "MSD test, 117", "dice": b.dice.mean(),
                             "sensitivity": b.sensitivity.mean(), "precision": b.precision.mean(),
                             "fp_distant_ml": b.fp_distant_ml.mean(), "fp_boundary_ml": b.fp_boundary_ml.mean()})
    L.append(f"\nLGG patients with a FLAIR copy in place of a T1 channel: {len(copies)}.")
    pd.DataFrame(rows).to_csv(RES / "round3_summary.csv", index=False)


def main():
    L = ["# Round 3 verdicts", "", "Scored by scripts/analyze_round3.py against PREREGISTRATION.md (Round 3).", ""]
    h9(L)
    L.append("")
    h8(L)
    (RES / "verdicts_round3.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
