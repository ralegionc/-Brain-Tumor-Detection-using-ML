# Verdicts

Scored by scripts/analyze.py against PREREGISTRATION.md.

- Seed spread, InceptionV3 without gates: mean Dice 0.866 to 0.869 (range 0.003), median distant FP 0.06 to 0.15 mL
- Seed spread, InceptionV3 with gates: mean Dice 0.863 to 0.871 (range 0.008), median distant FP 0.09 to 0.10 mL

**H1a NOT SUPPORTED.** Median per-patient change in distant FP volume (gates minus plain): inceptionv3 s0 +0.00 mL [-0.00, +0.00]; vgg16 s0 +0.00 mL [-0.02, +0.00]; vgg19 s0 +0.00 mL [-0.02, +0.00]; pooled InceptionV3 seeds +0.00 [-0.01, +0.00]

**H1b NOT SUPPORTED.** Mean Dice change: inceptionv3 s0 +0.003 [+0.000, +0.005]; vgg16 s0 -0.003 [-0.007, +0.000]; vgg19 s0 +0.000 [-0.002, +0.003]; pooled InceptionV3 seeds +0.000 [-0.002, +0.002]

**H2a SUPPORTED.** Sensitivity / precision: dice_bce 0.861 / 0.890; tversky 0.910 / 0.826; focal_tversky 0.907 / 0.831

**H2b SUPPORTED.** Lowest sensitivity: bce (0.844); BCE 0.844.

## Exploratory (not pre-registered)

Mean per-patient change, gates minus plain, mL [95% bootstrap CI]:

- inceptionv3 s0: distant FP -0.11 [-0.23, -0.01]; boundary FP -0.69 [-0.95, -0.43]; boundary FN +0.34 [+0.09, +0.59]
- inceptionv3 s1: distant FP -0.11 [-0.25, -0.01]; boundary FP +0.02 [-0.19, +0.23]; boundary FN -0.06 [-0.33, +0.21]
- inceptionv3 s2: distant FP +0.10 [+0.01, +0.20]; boundary FP -0.28 [-0.50, -0.06]; boundary FN -0.02 [-0.25, +0.22]
- vgg16 s0: distant FP +0.08 [-0.06, +0.23]; boundary FP -0.46 [-0.79, -0.12]; boundary FN +0.26 [-0.06, +0.57]
- vgg19 s0: distant FP -0.00 [-0.14, +0.14]; boundary FP -0.51 [-0.75, -0.27]; boundary FN +0.52 [+0.27, +0.80]
