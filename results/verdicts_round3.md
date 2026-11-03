# Round 3 verdicts

Scored by scripts/analyze_round3.py against PREREGISTRATION.md (Round 3).

**H9a SUPPORTED**, all 110 LGG patients. Distant FP, gates minus plain, seed-pooled: -10.40 [-12.58, -8.44] mL; per seed -8.77, -19.58, -2.85.

**H9b NOT SUPPORTED**, all 110 LGG patients. Dice, gates minus plain, seed-pooled: +0.002 [-0.009, +0.012].

**H9a (sensitivity analysis) SUPPORTED**, 100 without FLAIR copies. Distant FP, gates minus plain, seed-pooled: -10.19 [-12.50, -8.15] mL; per seed -8.48, -19.54, -2.55.

**H9b (sensitivity analysis) NOT SUPPORTED**, 100 without FLAIR copies. Dice, gates minus plain, seed-pooled: +0.005 [-0.006, +0.015].

- plain: on the 33 LGG test patients, MSD-trained Dice 0.549 vs LGG-trained 0.802. LGG-trained on the 117 MSD test patients 0.477 vs MSD-trained 0.858. MSD in-domain with T1w/FLAIR/T1gd inputs 0.858 vs round 1 FLAIR/T1gd/T2w 0.868.
- gates: on the 33 LGG test patients, MSD-trained Dice 0.551 vs LGG-trained 0.806. LGG-trained on the 117 MSD test patients 0.497 vs MSD-trained 0.858. MSD in-domain with T1w/FLAIR/T1gd inputs 0.858 vs round 1 FLAIR/T1gd/T2w 0.868.

LGG patients with a FLAIR copy in place of a T1 channel: 10.

**H8a NOT SUPPORTED.** Finest gate, mean value 0-5 mm outside the tumour minus >15 mm away, per patient: inceptionv3_s0 +0.497 [+0.480, +0.513]; inceptionv3_s1 +0.582 [+0.558, +0.605]; inceptionv3_s2 +0.293 [+0.284, +0.303]; vgg16_s0 +0.559 [+0.533, +0.584]; vgg16_s1 +0.525 [+0.502, +0.546]; vgg16_s2 +0.496 [+0.481, +0.510].

Mean gate value by distance from the true boundary (tumour-bearing slices, averaged over seeds):

| encoder     |   level |   in_deep |   in_edge |   out_edge |   out_mid |   out_far |   outside_brain |
|:------------|--------:|----------:|----------:|-----------:|----------:|----------:|----------------:|
| inceptionv3 |       1 |     0.757 |     0.762 |      0.685 |     0.454 |     0.228 |           0.419 |
| inceptionv3 |       2 |     0.794 |     0.76  |      0.714 |     0.61  |     0.455 |           0.584 |
| inceptionv3 |       3 |     0.937 |     0.942 |      0.944 |     0.949 |     0.963 |           0.944 |
| inceptionv3 |       4 |     0.944 |     0.941 |      0.94  |     0.939 |     0.935 |           0.93  |
| vgg16       |       1 |     0.725 |     0.782 |      0.742 |     0.443 |     0.215 |           0.445 |
| vgg16       |       2 |     0.989 |     0.952 |      0.906 |     0.783 |     0.411 |           0.482 |
| vgg16       |       3 |     0.997 |     0.988 |      0.979 |     0.962 |     0.872 |           0.804 |
| vgg16       |       4 |     0.85  |     0.79  |      0.753 |     0.71  |     0.664 |           0.558 |

**H8b InceptionV3 UNINTERPRETABLE.** Mean-ablated minus intact, pooled over 3 seeds: boundary FP -3.58 [-4.06, -3.12] mL, distant FP -0.33 [-0.51, -0.19] mL, Dice -0.101 [-0.112, -0.090]. For scale, plain minus gated boundary FP is +0.32 [+0.18, +0.45] mL.

**H8b VGG-16 NOT SUPPORTED.** Mean-ablated minus intact, pooled over 3 seeds: boundary FP -4.53 [-5.09, -3.98] mL, distant FP -0.52 [-0.77, -0.31] mL, Dice -0.036 [-0.045, -0.029]. For scale, plain minus gated boundary FP is +0.16 [+0.01, +0.34] mL.

Seed 0, change from intact under each ablation (descriptive; L1 = finest gate):

| model                      | ablation   |   d_dice |   d_fp_boundary_ml |   d_fp_distant_ml |   d_fn_boundary_ml |
|:---------------------------|:-----------|---------:|-------------------:|------------------:|-------------------:|
| inceptionv3_ag_dice_bce_s0 | mean       |   -0.076 |             -5.166 |            -0.383 |             12.211 |
| inceptionv3_ag_dice_bce_s0 | mean_L1    |   -0.106 |             -4.306 |            -0.357 |             14.313 |
| inceptionv3_ag_dice_bce_s0 | mean_L2    |   -0.009 |             -0.314 |            -0.081 |              1.029 |
| inceptionv3_ag_dice_bce_s0 | mean_L3    |   -0     |             -0.076 |            -0.006 |              0.061 |
| inceptionv3_ag_dice_bce_s0 | mean_L4    |    0     |              0.002 |             0.001 |             -0.001 |
| inceptionv3_ag_dice_bce_s0 | ones       |   -0.007 |             -1.713 |             0.051 |              2.187 |
| vgg16_ag_dice_bce_s0       | mean       |   -0.055 |             -5.367 |            -0.591 |             10.476 |
| vgg16_ag_dice_bce_s0       | mean_L1    |   -0.015 |             -2.319 |            -0.303 |              3.48  |
| vgg16_ag_dice_bce_s0       | mean_L2    |   -0.004 |             -1.04  |            -0.168 |              1.263 |
| vgg16_ag_dice_bce_s0       | mean_L3    |   -0.001 |             -0.473 |            -0.062 |              0.399 |
| vgg16_ag_dice_bce_s0       | mean_L4    |   -0.002 |             -1.546 |            -0.149 |              1.406 |
| vgg16_ag_dice_bce_s0       | ones       |   -0.006 |              1.961 |             1.939 |             -1.336 |

