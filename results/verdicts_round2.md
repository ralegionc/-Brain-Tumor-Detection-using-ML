# Round 2 verdicts

Scored by scripts/analyze_round2.py against PREREGISTRATION.md (Round 2).

**H5a NOT SUPPORTED.** Enhancing-tumour Dice, gates minus plain: Dice+BCE +0.011 [-0.002, +0.031]; Tversky +0.020 [+0.007, +0.039]; Dice+BCE seed-pooled +0.008 [-0.002, +0.023].

**H5b NOT SUPPORTED.** Enhancing-tumour Dice, Tversky minus Dice+BCE: plain -0.029 [-0.049, -0.014]; gated -0.020 [-0.051, +0.009]. Sensitivity change: plain +0.053, gated +0.060.

Seed spread (seed 0 vs 1, mean Dice): plain 0.006, gated 0.001. Test patients with no enhancing tumour: 4 of 117.

**H6a NOT SUPPORTED.** LGG, mean change in distant FP volume (nominal mL), gates minus plain, seed-pooled: +0.00 [-0.12, +0.11]; per seed -0.02, +0.18, -0.15.

**H6b NOT SUPPORTED.** LGG mean Dice change, seed-pooled: +0.004 [-0.001, +0.009].

Exploratory: plain-model mean distant FP 0.80 mL per patient; boundary FP change -0.51 [-0.97, -0.09].

**H7a SUPPORTED.** VGG-16 minus InceptionV3 Dice, 3 seeds each: plain +0.018 [+0.013, +0.023]; gated +0.016 [+0.011, +0.022].

VGG-16 gates minus plain, seed-pooled Dice (H1 recheck, descriptive): -0.001 [-0.002, +0.000].

**H7b SUPPORTED.** InceptionV3 at 2x input: Dice 0.885 (threshold 0.877; 1x was 0.868, VGG-16 0.885). Paired vs 1x: +0.017 [+0.013, +0.020].
