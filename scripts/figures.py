"""Figures for the README, from runs/ and results/ (run scripts/analyze.py first)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from seg.data import Slices  # noqa: E402
from seg.models import SegNet  # noqa: E402

FIG = ROOT / "figures"
BG, PANEL, INK, MUTED, GRID = "#1a1a19", "#232321", "#f0efec", "#c3c2b7", "#3a3936"
PLAIN, GATED = "#3987e5", "#d95926"
TP, FP, FN = "#199e70", "#e66767", "#c98500"
GRAY = matplotlib.colormaps["gray"].with_extremes(bad=BG)
LOSS_ORDER = ["bce", "focal", "dice", "dice_bce", "tversky", "focal_tversky"]
LOSS_NAME = {"bce": "BCE", "focal": "Focal", "dice": "Dice", "dice_bce": "Dice + BCE",
             "tversky": "Tversky", "focal_tversky": "Focal Tversky"}
LABEL_AT = {"dice": (-8, 6)}  # keeps the Dice and Dice + BCE labels apart
ENC_NAME = {"vgg16": "VGG-16", "vgg19": "VGG-19", "inceptionv3": "InceptionV3"}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def show(img):
    """Array (H,W) as stored -> display orientation (anterior up)."""
    return np.rot90(img, 1)


def load_model(run):
    cfg = pd.read_json(ROOT / "runs" / run / "config.json", typ="series")
    m = SegNet(cfg.encoder, bool(cfg.attention), pretrained=False, in_scale=int(cfg.get("in_scale", 1)))
    m.load_state_dict(torch.load(ROOT / "runs" / run / "final.pt", weights_only=True))
    return m.eval()


def hero(plain="inceptionv3_plain_dice_bce_s0", gated="inceptionv3_ag_dice_bce_s0", n=3):
    """For the test cases with stored probability maps, the slice where gates removed the
    most false-positive area: FLAIR + truth, each model's errors, and the finest gate map."""
    ds = Slices("test")
    mp = np.load(ROOT / "runs" / plain / "test_probmaps.npz")
    mg = np.load(ROOT / "runs" / gated / "test_probmaps.npz")
    rows = dict(ds.cases())
    picks = []
    for case in mp.files:
        r = np.sort(rows[case])
        gt = ds.y[r].astype(bool)
        pp, pg = mp[case] >= 128, mg[case] >= 128
        gain = (pp & ~gt).sum((1, 2)) - (pg & ~gt).sum((1, 2))
        z = int(np.argmax(gain + 0.01 * gt.sum((1, 2))))
        picks.append((gain[z], case, r[z], gt[z], pp[z], pg[z]))
    picks = sorted(picks, key=lambda t: -t[0])[:n]

    model = load_model(gated)
    att_cmap = LinearSegmentedColormap.from_list("att", [BG, "#5a2a14", GATED, "#ffd2b8"])
    fig, ax = plt.subplots(n, 4, figsize=(12, 3.2 * n))
    for i, (_, case, row, gt, pp, pg) in enumerate(picks):
        x, _ = ds.batch(np.array([row]))
        with torch.no_grad():
            model(x)
        alpha = model.attention_maps()[0][0, 0].numpy()
        flair = np.ma.masked_where((x[0] == 0).all(0).numpy(), x[0, 0].numpy())  # outside the brain
        ext = (-0.5, flair.shape[1] - 0.5, flair.shape[0] - 0.5, -0.5)  # gate maps are coarser
        panels = [("FLAIR + true tumour", None), ("Plain U-Net", pp), ("With attention gates", pg),
                  ("Finest gate coefficients", "att")]
        for j, (title, p) in enumerate(panels):
            a = ax[i, j]
            if isinstance(p, str):
                a.imshow(show(flair), cmap=GRAY, vmin=-2, vmax=4)
                a.imshow(show(alpha), cmap=att_cmap, alpha=0.75, vmin=0, vmax=1, extent=ext)
            else:
                a.imshow(show(flair), cmap=GRAY, vmin=-2, vmax=4)
                if p is not None:
                    over = np.zeros((*gt.shape, 4))
                    over[p & gt] = matplotlib.colors.to_rgba(TP, 0.55)
                    over[p & ~gt] = matplotlib.colors.to_rgba(FP, 0.9)
                    over[~p & gt] = matplotlib.colors.to_rgba(FN, 0.9)
                    a.imshow(show(over))
                a.contour(show(gt), [0.5], colors=[INK], linewidths=0.9)
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            if i == 0:
                a.set_title(title, color=INK, fontsize=11, pad=8)
        ax[i, 0].set_ylabel(case, color=MUTED, fontsize=9)
    handles = [matplotlib.patches.Patch(color=c, label=l) for c, l in
               [(TP, "correct tumour"), (FP, "false alarm"), (FN, "missed tumour")]]
    handles.append(matplotlib.lines.Line2D([], [], color=INK, lw=1, label="true boundary"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, labelcolor=MUTED,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Best case for the gates: the 3 slices (of 6 stored test patients) where they removed the most "
                 "false alarms", fontsize=12, x=0.02, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(FIG / "hero_false_alarms.png", dpi=130)
    plt.close(fig)


def forest(paired, explo):
    P = paired[paired.comparison.str.startswith("attention")].copy()
    if not len(P):
        return
    P["label"] = P.comparison.str.replace("attention ", "")
    X = explo.set_index("pair")
    for m in ["fp_distant_ml", "fp_boundary_ml"]:
        for suf in ["", "_lo", "_hi"]:
            P["m_" + m + suf] = X.loc[P.label, m + suf].to_numpy()
    for k, v in ENC_NAME.items():
        P["label"] = P.label.str.replace(k, v)
    metrics = [("d_dice", "Mean change in Dice\n(pre-registered)", "{:+.3f}"),
               ("m_fp_distant_ml", "Mean change in false alarms >5 mm\nfrom tumour, mL (exploratory)", "{:+.2f}"),
               ("m_fp_boundary_ml", "Mean change in false alarms <=5 mm\nfrom tumour, mL (exploratory)", "{:+.2f}")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 0.5 * len(P) + 1.8), sharey=True)
    y = np.arange(len(P))[::-1]
    for a, (m, title, fmt) in zip(ax, metrics):
        a.axvline(0, color=GRID, lw=1)
        a.hlines(y, P[m + "_lo"], P[m + "_hi"], color=GATED, lw=2)
        a.plot(P[m], y, "o", color=GATED, ms=8, mec=BG, mew=2)
        for yi, v in zip(y, P[m]):
            a.annotate(fmt.format(v), (v, yi), xytext=(0, 7), textcoords="offset points",
                       ha="center", color=MUTED, fontsize=8)
        a.set_title(title, color=INK, fontsize=10, loc="left")
        a.grid(axis="x", color=GRID, lw=0.5)
        a.tick_params(left=False)
    ax[0].set_yticks(y, P.label)
    fig.suptitle("Attention gates minus plain model, per test patient (95% bootstrap CI)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "attention_effects.png", dpi=130)
    plt.close(fig)


def loss_tradeoff(S):
    L = S[(S.encoder == "inceptionv3") & S.attention & (S.seed == 0)].set_index("loss")
    L = L.loc[[l for l in LOSS_ORDER if l in L.index]]
    if len(L) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for l, r in L.iterrows():
        ax.errorbar(r.precision, r.sensitivity, xerr=[[r.precision - r.precision_lo], [r.precision_hi - r.precision]],
                    yerr=[[r.sensitivity - r.sensitivity_lo], [r.sensitivity_hi - r.sensitivity]],
                    fmt="o", color=GATED, ecolor=GRID, ms=9, mec=BG, mew=2, elinewidth=1.5)
        ax.annotate(f"{LOSS_NAME[l]}  (Dice {r.dice:.3f})", (r.precision, r.sensitivity), xytext=LABEL_AT.get(l, (8, 4)), ha="right" if l == "dice" else "left",
                    textcoords="offset points", color=INK, fontsize=9)
    ax.set_xlabel("Precision (share of predicted tumour that is tumour)")
    ax.set_ylabel("Sensitivity (share of tumour found)")
    ax.grid(color=GRID, lw=0.5)
    ax.set_title("Loss functions trade misses for false alarms\nInceptionV3 + attention gates, 117 test patients",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "loss_tradeoff.png", dpi=130)
    plt.close(fig)


def encoders(S):
    E = S[(S.loss == "dice_bce") & (S.seed == 0)]
    if not len(E):
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    encs = [e for e in ENC_NAME if e in set(E.encoder)]
    for a, (m, lo, hi, title) in zip(ax, [("dice", "dice_lo", "dice_hi", "Mean Dice"),
                                           ("hd95_median", "hd95_lo", "hd95_hi", "Median HD95 (mm), lower is better")]):
        for k, e in enumerate(encs):
            for att, col, dy in [(False, PLAIN, 0.12), (True, GATED, -0.12)]:
                r = E[(E.encoder == e) & (E.attention == att)]
                if not len(r):
                    continue
                r = r.iloc[0]
                a.hlines(k + dy, r[lo], r[hi], color=col, lw=2)
                a.plot(r[m], k + dy, "o", color=col, ms=8, mec=BG, mew=2,
                       label=("with attention gates" if att else "plain") if k == 0 else None)
        a.set_yticks(range(len(encs)), [ENC_NAME[e] for e in encs])
        a.invert_yaxis()
        a.set_title(title, loc="left", color=INK, fontsize=10)
        a.grid(axis="x", color=GRID, lw=0.5)
        a.tick_params(left=False)
    fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, labelcolor=MUTED, loc="upper right", ncol=2)
    fig.suptitle("Encoders, Dice + BCE loss, seed 0 (95% bootstrap CI over 117 patients)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "encoders.png", dpi=130)
    plt.close(fig)


def curves(keep):
    fig, ax = plt.subplots(figsize=(8, 4))
    for d in sorted((ROOT / "runs").iterdir()):
        f = d / "train_log.csv"
        if d.name not in keep or not f.exists():
            continue
        l = pd.read_csv(f).dropna(subset=["val_dice"])
        col = GATED if "_ag_" in d.name else PLAIN
        ax.plot(l.step, l.val_dice, color=col, lw=1.2, alpha=0.8)
    ax.set_xlabel("training step (batch 16)")
    ax.set_ylabel("validation Dice (256 fixed slices)")
    ax.grid(color=GRID, lw=0.5)
    ax.set_title("Validation Dice during training, round 1 runs (orange = gated)", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", dpi=130)
    plt.close(fig)


def threshold_curve():
    """H4: Dice+BCE traced over thresholds in precision/sensitivity space, with Tversky at 0.5."""
    f = ROOT / "results" / "threshold_curves.csv"
    if not f.exists():
        return
    C = pd.read_csv(f)
    C = C[(C.split == "test") & (C.k >= 8)]
    base, tv = C[C.run.str.contains("dice_bce")], C[C.run.str.contains("tversky")]
    txt = (ROOT / "results" / "h4.md").read_text()
    k_star = int(txt.split("t* = ")[1].split("/")[0])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(base.precision, base.sensitivity, color=PLAIN, lw=2, label="Dice + BCE, threshold swept")
    for k, lab in [(128, "0.5"), (k_star, f"t* = {k_star / 255:.2f}")]:
        r = base[base.k == k].iloc[0]
        ax.plot(r.precision, r.sensitivity, "o", color=PLAIN, ms=9, mec=BG, mew=2)
        ax.annotate(f"Dice + BCE at {lab}  (Dice {r.dice:.3f})", (r.precision, r.sensitivity), xytext=(10, -4),
                    textcoords="offset points", color=INK, fontsize=9)
    r = tv[tv.k == 128].iloc[0]
    ax.plot(r.precision, r.sensitivity, "D", color=GATED, ms=9, mec=BG, mew=2, label="Tversky at 0.5")
    ax.annotate(f"Tversky at 0.5  (Dice {r.dice:.3f})", (r.precision, r.sensitivity), xytext=(-10, 6),
                textcoords="offset points", ha="right", color=INK, fontsize=9)
    ax.set_xlim(0.72, 0.92)
    ax.set_ylim(0.80, 0.96)
    ax.set_xlabel("Precision (share of predicted tumour that is tumour)")
    ax.set_ylabel("Sensitivity (share of tumour found)")
    ax.grid(color=GRID, lw=0.5)
    ax.legend(frameon=False, labelcolor=MUTED, loc="lower left")
    ax.set_title("Lowering the threshold buys what Tversky buys\nInceptionV3 + gates, whole tumour, 117 test patients",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "threshold_vs_loss.png", dpi=130)
    plt.close(fig)


def gate_settings():
    """Gates minus plain across settings, per-patient differences averaged over seeds."""
    from scripts.analyze import boot_ci, load
    runs = load()
    name = lambda pre, enc, att, s: f"{pre}{enc}_{'ag' if att else 'plain'}_dice_bce_s{s}"
    settings = [("", "inceptionv3", 3, "Whole tumour, skull-stripped (MSD)\nInceptionV3, 3 seeds, 117 patients"),
                ("", "vgg16", 3, "Whole tumour, skull-stripped (MSD)\nVGG-16, 3 seeds, 117 patients"),
                ("msd-et_", "inceptionv3", 2, "Enhancing tumour (MSD)\nInceptionV3, 2 seeds, 117 patients"),
                ("lgg-wt_", "inceptionv3", 3, "Whole tumour, skull kept (LGG)\nInceptionV3, 3 seeds, 33 patients")]
    rows = []
    for pre, enc, n, lab in settings:
        pairs = [(name(pre, enc, 1, s), name(pre, enc, 0, s)) for s in range(n)]
        if not all(a in runs and b in runs for a, b in pairs):
            continue
        r = {"label": lab}
        for m in ["dice", "fp_distant_ml", "fp_boundary_ml"]:
            d = np.mean([(runs[a][m] - runs[b][m]).to_numpy() for a, b in pairs], axis=0)
            r[m], r[m + "_lo"], r[m + "_hi"] = boot_ci(d)
        rows.append(r)
    if not rows:
        return
    P = pd.DataFrame(rows)
    metrics = [("dice", "Mean change in Dice", "{:+.3f}"),
               ("fp_distant_ml", "Mean change in false alarms >5 mm\nfrom tumour, mL", "{:+.2f}"),
               ("fp_boundary_ml", "Mean change in false alarms <=5 mm\nfrom tumour, mL", "{:+.2f}")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 0.75 * len(P) + 1.8), sharey=True)
    y = np.arange(len(P))[::-1]
    for a, (m, title, fmt) in zip(ax, metrics):
        a.axvline(0, color=GRID, lw=1)
        a.hlines(y, P[m + "_lo"], P[m + "_hi"], color=GATED, lw=2)
        a.plot(P[m], y, "o", color=GATED, ms=8, mec=BG, mew=2)
        for yi, v in zip(y, P[m]):
            a.annotate(fmt.format(v), (v, yi), xytext=(0, 7), textcoords="offset points",
                       ha="center", color=MUTED, fontsize=8)
        a.set_title(title, color=INK, fontsize=10, loc="left")
        a.grid(axis="x", color=GRID, lw=0.5)
        a.tick_params(left=False)
    ax[0].set_yticks(y, P.label, fontsize=9)
    fig.suptitle("Attention gates minus plain model in each setting, Dice + BCE (95% bootstrap CI)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "gates_by_setting.png", dpi=130)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    S = pd.read_csv(ROOT / "results" / "summary.csv")
    curves(set(S.run))
    threshold_curve()
    gate_settings()
    encoders(S)
    loss_tradeoff(S)
    if (ROOT / "results" / "paired.csv").exists():
        forest(pd.read_csv(ROOT / "results" / "paired.csv"), pd.read_csv(ROOT / "results" / "exploratory_fp.csv"))
    runs = set(S.run)
    if {"inceptionv3_plain_dice_bce_s0", "inceptionv3_ag_dice_bce_s0"} <= runs:
        hero()
    print("figures in", FIG)


if __name__ == "__main__":
    main()
