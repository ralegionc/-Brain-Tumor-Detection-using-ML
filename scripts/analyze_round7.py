"""Score round 7 (H22, training on planted small tumours) against PREREGISTRATION.md.
Arms: B = round 4 VGG-16 plain (baseline), S = small bank, N = native bank; three seeds each.
Writes results/verdicts_round7.md and results/round7/summary.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.analyze import boot_ci  # noqa: E402
from scripts.analyze_round2 import fmt  # noqa: E402

RES = ROOT / "results"
R7 = RES / "round7"
ARMS = {"B": "r4_vgg16_plain_dice_bce_s{}", "S": "r7_vgg16_plain_dice_bce_s{}_bank_small",
        "N": "r7_vgg16_plain_dice_bce_s{}_bank_native"}
COLS = ["dice", "fp_distant_ml", "missed_lesions_big", "gt_lesions_big", "fp_boundary_ml", "sensitivity", "precision"]


def boot90(v, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    s = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(10000)])
    return v.mean(), *np.percentile(s, [5, 95])


def real():
    """Per patient, seed-mean metrics for each arm, plus each seed's Dice."""
    out, seeds = {}, {}
    for arm, pat in ARMS.items():
        d = [pd.read_csv(ROOT / "runs" / pat.format(s) / "test_metrics.csv").set_index("case").sort_index()
             for s in range(3)]
        out[arm] = sum(x[COLS] for x in d) / 3
        out[arm]["gt_ml"] = d[0].gt_ml
        seeds[arm] = [x.dice for x in d]
    return out, seeds


def plants():
    P = {}
    for arm in ARMS:
        d = pd.read_csv(R7 / f"plants_{arm}.csv")
        d = d[d.placed == True].copy()  # noqa: E712
        d["dice"] = d[["dice_s0", "dice_s1", "dice_s2"]].mean(1)
        P[arm] = d.set_index(["recipient", "donor", "kind", "target_ml"])
    return P


def main():
    L = ["# Round 7 verdicts\n", "Scored by scripts/analyze_round7.py against PREREGISTRATION.md (Round 7).\n"]
    v = lambda ok: "SUPPORTED" if ok else "NOT SUPPORTED"
    R, seeds = real()
    B, S, N = R["B"], R["S"], R["N"]
    small = B.gt_ml <= 47.3
    L.append(f"Small quarter: {small.sum()} of {len(B)} test patients (true volume at most 47.3 mL).\n")

    a = boot90(S.dice - B.dice)
    L.append(f"**H22a {v(a[1] > -0.005)}.** Overall Dice, S minus B: {a[0]:+.4f}, 90% CI [{a[1]:+.4f}, {a[2]:+.4f}] "
             f"(B {B.dice.mean():.4f}, S {S.dice.mean():.4f}, N {N.dice.mean():.4f}).\n")
    b = boot_ci((S.dice - B.dice)[small])
    L.append(f"**H22b {v(b[0] >= 0.01 and b[1] > 0)}.** Small quarter Dice, S minus B: {fmt(b)} "
             f"(B {B.dice[small].mean():.3f}, S {S.dice[small].mean():.3f}).\n")
    c = boot_ci((S.dice - N.dice)[small])
    nb = boot_ci((N.dice - B.dice)[small])
    L.append(f"**H22c {v(c[1] > 0)}.** Small quarter Dice, S minus N: {fmt(c)}. For reference N minus B: {fmt(nb)}.\n")

    P = plants()
    key = P["B"].index.intersection(P["S"].index).intersection(P["N"].index)
    pb, ps, pn = (P[k].loc[key] for k in "BSN")
    sel = lambda kind: (pb.index.get_level_values("kind") == kind) & pb.index.get_level_values("target_ml").isin([5, 10])
    lg, sm = sel("large"), sel("small")
    gd = (ps.dice - pb.dice)[lg].mean()
    L.append(f"**H22d {v(gd >= 0.10)}.** Planted 5 and 10 mL tumours from naturally large donors, S minus B: "
             f"{gd:+.3f} ({lg.sum()} plants; B {pb.dice[lg].mean():.3f}, S {ps.dice[lg].mean():.3f}).\n")
    per = (ps.dice - pb.dice)[sm].groupby(level="recipient").mean()
    e = boot_ci(per)
    L.append(f"**H22e {v(e[1] > 0 and e[0] >= 0.5 * gd)}.** Same, naturally small donors: {fmt(e)} per recipient "
             f"({sm.sum()} plants; B {pb.dice[sm].mean():.3f}, S {ps.dice[sm].mean():.3f}); half the H22d gain is "
             f"{0.5 * gd:+.3f}.\n")
    f = boot_ci(S.fp_distant_ml - B.fp_distant_ml)
    L.append(f"**H22f {v(f[0] < 0.5)}.** Distant false alarms, S minus B: {fmt(f, 2)} mL "
             f"(B {B.fp_distant_ml.mean():.2f}, S {S.fp_distant_ml.mean():.2f}, N {N.fp_distant_ml.mean():.2f}).\n")

    L.append("## Descriptive\n")
    rows = []
    for arm, d in R.items():
        rows.append({"arm": arm, "dice": d.dice.mean(), "dice_small_quarter": d.dice[small].mean(),
                     "dice_rest": d.dice[~small].mean(), "sensitivity": d.sensitivity.mean(),
                     "precision": d.precision.mean(), "fp_distant_ml": d.fp_distant_ml.mean(),
                     "fp_boundary_ml": d.fp_boundary_ml.mean(),
                     "missed_lesions_big": d.missed_lesions_big.sum(), "lesions_big": d.gt_lesions_big.sum(),
                     **{f"dice_s{s}": seeds[arm][s].mean() for s in range(3)},
                     **{f"small_s{s}": seeds[arm][s][small].mean() for s in range(3)}})
    T = pd.DataFrame(rows)
    T.to_csv(R7 / "summary.csv", index=False)
    L.append("Real test patients (missed lesions are seed means summed over patients):\n")
    L.append(T.round(4).to_markdown(index=False))
    L.append("")
    def by_size(d):
        d = d.reset_index()
        d["size"] = np.where(d.kind == "native", "own size", d.target_ml.astype(int).astype(str) + " mL")
        return d.groupby(["kind", "size"], sort=False).dice.mean()
    G = pd.DataFrame({arm: by_size(P[arm].loc[key]) for arm in "BSN"})
    order = {"small": 0, "large": 1, "native": 2}
    G = G.loc[sorted(G.index, key=lambda k: (order[k[0]], float(k[1].split()[0]) if k[1][0].isdigit() else 0))]
    G["S-B"], G["N-B"] = G.S - G.B, G.N - G.B
    L.append("Planted tumours, seed-mean Dice by donor kind and target volume:\n")
    L.append(G.round(3).to_markdown())
    L.append("")
    miss = {arm: (P[arm].loc[key].dice < 0.1)[lg | sm].mean() for arm in "BSN"}
    L.append("Planted 5 and 10 mL tumours missed entirely (Dice under 0.1): "
             + ", ".join(f"{k} {m:.1%}" for k, m in miss.items()) + ".\n")
    for kind in ["small", "native"]:
        I = pd.read_csv(R7 / f"bank_{kind}_index.csv").drop_duplicates("case")
        q = I.target_ml.quantile([0.1, 0.5, 0.9])
        L.append(f"Bank {kind}: {len(I)} plants, planted volume 10th/50th/90th percentile "
                 f"{q.iloc[0]:.1f} / {q.iloc[1]:.1f} / {q.iloc[2]:.1f} mL.\n")
    text = "\n".join(L)
    (RES / "verdicts_round7.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
