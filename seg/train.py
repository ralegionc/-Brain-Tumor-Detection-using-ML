"""Train one configuration for a fixed step budget, then evaluate on every test volume.

Resumable: a checkpoint is written every --ckpt-every steps and picked up on restart,
so a sleep or crash loses at most that many steps. Fixed budget with cosine decay and
no checkpoint selection, so every configuration is treated identically and the
validation split is only used to watch for divergence.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import ROOT
from .data import CHANNELS, DATASETS, Slices, train_patients
from .losses import LOSSES
from .metrics import error_patterns, volume_metrics
from .models import SegNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_name(a):
    """Round 1 names are unchanged; other datasets, targets and input scales get tags."""
    pre = "" if (a.data, a.target) == ("msd", "wt") else f"{a.data}-{a.target}_"
    post = "" if a.in_scale == 1 else f"_x{a.in_scale}"
    post += f"_c{a.context}" if a.context else ""
    post += f"_f{round(a.train_frac * 100)}" if a.train_frac < 1 else ""
    post += f"_{Path(a.bank).name}" if a.bank else ""
    return f"{pre}{a.encoder}_{'ag' if a.attention else 'plain'}_{a.loss}_s{a.seed}{post}"


@torch.no_grad()
def predict_volume(model, ds, rows, bs=32):
    probs = []
    dev = next(model.parameters()).device
    for i in range(0, len(rows), bs):
        x, _ = ds.batch(rows[i:i + bs])
        probs.append(torch.sigmoid(model(x.to(dev)).float())[:, 0].cpu().numpy())
    return np.concatenate(probs)


def evaluate(model, out, split="test", keep_maps=6, limit=0, data="msd", target="wt", context=0):
    ds = Slices(split, DATASETS[data], target, CHANNELS.get(data), context)
    model.eval()
    rows_out = []
    maps = {}
    for k, (case, rows) in enumerate(ds.cases()):
        if limit and k >= limit:
            break
        p = predict_volume(model, ds, rows)
        pred, gt = p >= 0.5, ds.y[np.sort(rows)].astype(bool)
        r = {"case": case, **volume_metrics(pred, gt, ds.spacing), **error_patterns(pred, gt, ds.spacing)}
        rows_out.append(r)
        if k < keep_maps:  # a few probability volumes for figures
            maps[case] = (p * 255).astype(np.uint8)
        if k % 20 == 0:
            print(f"  eval {k} {case} dice={r['dice']:.3f} hd95={r['hd95']:.1f}", flush=True)
    df = pd.DataFrame(rows_out)
    df.to_csv(out / f"{split}_metrics.csv", index=False)
    np.savez_compressed(out / f"{split}_probmaps.npz", **maps)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="vgg16", choices=["vgg16", "vgg19", "inceptionv3"])
    ap.add_argument("--attention", action="store_true")
    ap.add_argument("--loss", default="dice_bce", choices=list(LOSSES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="msd", choices=list(DATASETS))
    ap.add_argument("--target", default="wt", choices=["wt", "et"])
    ap.add_argument("--in-scale", type=int, default=1, help="upsample inputs inside the network")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--out", default="", help="override the run directory (smoke tests)")
    ap.add_argument("--eval-limit", type=int, default=0, help="evaluate only the first N test cases")
    ap.add_argument("--eval-only", action="store_true", help="re-evaluate final.pt without training")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast on a GPU (loss stays in fp32)")
    ap.add_argument("--ram", action="store_true", help="read the training arrays into memory first")
    ap.add_argument("--context", type=int, default=0, choices=[0, 1], help="1: 2.5D, add the slices above and below")
    ap.add_argument("--train-frac", type=float, default=1.0, help="train on this share of the training patients")
    ap.add_argument("--bank", default="", help="folder of extra training slices (scripts/build_bank.py)")
    ap.add_argument("--bank-frac", type=float, default=0.25, help="chance that a sample is drawn from the bank")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)

    out = ROOT / (a.out or f"runs/{run_name(a)}")
    out.mkdir(parents=True, exist_ok=True)
    if a.eval_only:  # re-score saved weights after a metric change
        torch.set_num_threads(a.threads)
        model = SegNet(a.encoder, a.attention, pretrained=False, in_scale=a.in_scale, in_ch=3 + 6 * a.context)
        model.load_state_dict(torch.load(out / "final.pt", weights_only=True, map_location="cpu"))
        model.to(DEVICE)
        df = evaluate(model, out, limit=a.eval_limit, data=a.data, target=a.target, context=a.context)
        print(out.name, "re-scored, test dice mean", round(df["dice"].mean(), 4))
        return
    if (out / "test_metrics.csv").exists():
        print("done already:", out.name)
        return
    json.dump(vars(a), open(out / "config.json", "w"), indent=1)

    torch.manual_seed(a.seed)
    model = SegNet(a.encoder, a.attention, pretrained=True, in_scale=a.in_scale, in_ch=3 + 6 * a.context).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    lr_at = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * s / a.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    loss_fn = LOSSES[a.loss]
    train, val = (Slices(s, DATASETS[a.data], a.target, CHANNELS.get(a.data), a.context) for s in ["train", "val"])
    pool = (np.flatnonzero(train.index.case.isin(train_patients(train.index.case, a.train_frac)))
            if a.train_frac < 1 else None)  # rows of the kept patients
    val_idx = np.random.default_rng(123).choice(len(val), min(256, len(val)), replace=False)
    bank = Slices("train", Path(a.bank), a.target, CHANNELS.get(a.data), a.context) if a.bank else None
    if a.ram:
        train.x, train.y = np.array(train.x), np.array(train.y)  # a copy; asarray keeps the memmap
        if bank is not None:
            bank.x, bank.y = np.array(bank.x), np.array(bank.y)
    rng = np.random.default_rng(a.seed)
    step, log = 0, []
    amp = a.amp and DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ck = out / "last.pt"
    if ck.exists():
        st = torch.load(ck, weights_only=False, map_location="cpu")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        if "scaler" in st:
            scaler.load_state_dict(st["scaler"])
        step, log = st["step"], st["log"]
        rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"])
        print(f"resumed {out.name} at step {step}")

    t0 = time.time()
    while step < a.steps:
        model.train()
        pick = rng.integers(0, len(train) if pool is None else len(pool), a.batch)
        x, y = train.batch(pick if pool is None else pool[pick], rng)
        if bank is not None:  # replace a random share of the batch with planted-tumour slices
            nb = int((rng.random(a.batch) < a.bank_frac).sum())
            if nb:
                bx, by = bank.batch(rng.integers(0, len(bank), nb), rng)
                x, y = torch.cat([x[nb:], bx]), torch.cat([y[nb:], by])
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
        loss = loss_fn(logits.float(), y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        rec = {"step": step, "loss": loss.item(), "lr": sched.get_last_lr()[0]}
        if step % a.val_every == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = val.batch(val_idx)
                vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    vp = torch.cat([model(vx[i:i + 32]) for i in range(0, len(val_idx), 32)]).float()
                rec["val_loss"] = loss_fn(vp, vy).item()
                pb, tb = vp >= 0, vy.bool()
                rec["val_dice"] = (2 * (pb & tb).sum() / (pb.sum() + tb.sum()).clamp(min=1)).item()
        log.append(rec)
        if step % 10 == 0 or "val_dice" in rec:
            el = time.time() - t0
            print(f"{out.name} step {step}/{a.steps} loss {loss.item():.4f}"
                  + (f" val_dice {rec['val_dice']:.3f}" if "val_dice" in rec else "")
                  + f" ({el:.0f}s this session)", flush=True)
        if step % a.ckpt_every == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "log": log, "rng": rng.bit_generator.state, "scaler": scaler.state_dict(),
                        "torch_rng": torch.get_rng_state()}, out / "tmp.pt")
            (out / "tmp.pt").replace(ck)  # atomic, so a crash mid-save cannot corrupt last.pt

    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    torch.save(model.state_dict(), out / "final.pt")
    df = evaluate(model, out, limit=a.eval_limit, data=a.data, target=a.target, context=a.context)
    print(out.name, "test dice mean", round(df["dice"].mean(), 4), "median hd95", round(df["hd95"].median(), 2))
    ck.unlink()  # optimiser state is 3x the model; final.pt is all that is needed now


if __name__ == "__main__":
    main()
