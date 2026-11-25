# Round 5 verdicts

Scored by scripts/analyze_round5.py against PREREGISTRATION.md (Round 5).

**H13.** Gates minus plain Dice, seed-pooled, 90% CI against a margin of ±0.005 (rows marked 'restatement' were seen before the margin was set):

| setting                                    | kind        |   dice_diff |   ci90_lo |   ci90_hi | verdict              |
|:-------------------------------------------|:------------|------------:|----------:|----------:|:---------------------|
| MSD, InceptionV3, 600 steps                | restatement |      0.0004 |   -0.0014 |    0.0019 | equivalent           |
| MSD, VGG-16, 600 steps                     | restatement |     -0.001  |   -0.0021 |    0      | equivalent           |
| Enhancing tumour, InceptionV3, 600 steps   | restatement |      0.0083 |   -0.0009 |    0.0202 | not shown equivalent |
| Skull kept (LGG), InceptionV3, 600 steps   | restatement |      0.0041 |   -0.0002 |    0.0085 | not shown equivalent |
| MSD-trained on LGG, InceptionV3, 600 steps | restatement |      0.0019 |   -0.0073 |    0.0104 | not shown equivalent |
| MSD, VGG-16, 2,000 steps                   | test        |     -0.0018 |   -0.0037 |   -0.0002 | equivalent           |
| MSD, VGG-16, 6,000 steps                   | test        |      0.0001 |   -0.0007 |    0.0011 | equivalent           |
| MSD, VGG-16, 24,500 steps                  | restatement |     -0.0006 |   -0.0016 |    0.0003 | equivalent           |
| MSD, InceptionV3 at 2x, 24,500 steps       | restatement |      0.0008 |   -0.0001 |    0.0017 | equivalent           |
| MSD, VGG-16, 25% of patients               | test        |     -0.0006 |   -0.0023 |    0.001  | equivalent           |
| MSD, VGG-16, 10% of patients               | test        |      0.0006 |   -0.0015 |    0.0027 | equivalent           |

**H14 NOT SUPPORTED.** VGG-16, gates minus plain edge false alarms by training budget (mL): 600 steps -0.16 [-0.34, -0.01]; 2,000 steps -0.17 [-0.27, -0.07]; 6,000 steps -0.04 [-0.17, +0.06]; 24,500 steps -0.03 [-0.13, +0.07].

|   steps |   plain_dice |   dice |   dice_lo |   dice_hi |   fp_boundary_ml |   fp_boundary_ml_lo |   fp_boundary_ml_hi |   fp_distant_ml |   fp_distant_ml_lo |   fp_distant_ml_hi |
|--------:|-------------:|-------:|----------:|----------:|-----------------:|--------------------:|--------------------:|----------------:|-------------------:|-------------------:|
|     600 |        0.885 | -0.001 |    -0.002 |     0     |           -0.165 |              -0.336 |              -0.015 |           0.029 |             -0.055 |              0.109 |
|    2000 |        0.895 | -0.002 |    -0.004 |    -0     |           -0.168 |              -0.268 |              -0.069 |          -0.046 |             -0.109 |              0.007 |
|    6000 |        0.899 |  0     |    -0.001 |     0.001 |           -0.043 |              -0.167 |               0.056 |          -0.044 |             -0.111 |              0.008 |
|   24500 |        0.904 | -0.001 |    -0.002 |     0.001 |           -0.032 |              -0.129 |               0.067 |          -0.058 |             -0.132 |              0.008 |

**H15 25% NOT SUPPORTED.** Gates minus plain, 25% of training patients: Dice -0.001 [-0.003, +0.001], edge false alarms -0.04 [-0.15, +0.08] mL, far -0.05 [-0.11, +0.00] mL. Plain Dice 0.886 against 0.904 with all patients.

**H15 10% NOT SUPPORTED.** Gates minus plain, 10% of training patients: Dice +0.001 [-0.002, +0.003], edge false alarms -0.08 [-0.17, +0.02] mL, far -0.07 [-0.17, +0.02] mL. Plain Dice 0.863 against 0.904 with all patients.

**H16a NOT SUPPORTED.** Temperature fitted on validation: 0.97, 0.97, 0.97 (above 1 means overconfident).

**H16b NOT SUPPORTED.** Test ECE (voxels with p >= 0.01) before and after temperature scaling: 0.0776 to 0.0772; 0.0788 to 0.0785; 0.0782 to 0.0777.

| model           |      T |   test_ece |   test_ece_scaled |   ece_cut |
|:----------------|-------:|-----------:|------------------:|----------:|
| seed 0          | 0.9675 |     0.0776 |            0.0772 |    0.0055 |
| seed 1          | 0.9669 |     0.0788 |            0.0785 |    0.004  |
| seed 2          | 0.9717 |     0.0782 |            0.0777 |    0.006  |
| 3-seed ensemble | 0.9399 |     0.0654 |            0.0663 |   -0.0136 |

Reliability, seed 0 test set (descriptive):

| predicted p   |   share of voxels |   mean predicted |   observed tumour share |   share of all tumour |
|:--------------|------------------:|-----------------:|------------------------:|----------------------:|
| 0 to 0.01     |            0.9821 |           0      |                  0.0008 |                0.0498 |
| 0.01 to 0.05  |            0.0009 |           0.0217 |                  0.3212 |                0.0173 |
| 0.05 to 0.1   |            0.0002 |           0.0709 |                  0.3993 |                0.0055 |
| 0.1 to 0.3    |            0.0004 |           0.1852 |                  0.4394 |                0.0093 |
| 0.3 to 0.5    |            0.0002 |           0.3972 |                  0.4797 |                0.0062 |
| 0.5 to 0.7    |            0.0002 |           0.6035 |                  0.5122 |                0.0068 |
| 0.7 to 0.9    |            0.0004 |           0.8164 |                  0.5558 |                0.0128 |
| 0.9 to 0.95   |            0.0003 |           0.9289 |                  0.6032 |                0.0091 |
| 0.95 to 0.99  |            0.0007 |           0.9766 |                  0.6703 |                0.0288 |
| 0.99 to 1     |            0.0147 |           0.9996 |                  0.964  |                0.8544 |

**H17a SUPPORTED.** Spearman, seed disagreement vs ensemble Dice: -0.85 [-0.89, -0.78].

**H17b SUPPORTED.** The 12 patients with the most disagreement include 7 of the 12 with the lowest Dice (chance about 1.2).

Descriptive, ensemble uncertain volume instead: Spearman -0.86 [-0.90, -0.79], 7 of the 12 worst caught.

The 12 lowest-Dice test patients:

| case      |   ensemble_dice |   gt_ml |   disagree_frac |
|:----------|----------------:|--------:|----------------:|
| BRATS_380 |           0.573 |  21.75  |           0.504 |
| BRATS_338 |           0.668 |  20.2   |           0.406 |
| BRATS_096 |           0.715 |  77.417 |           0.084 |
| BRATS_352 |           0.735 |  95.078 |           0.307 |
| BRATS_355 |           0.749 |  21.287 |           0.274 |
| BRATS_302 |           0.77  |  35.014 |           0.351 |
| BRATS_353 |           0.773 |  27.06  |           0.317 |
| BRATS_459 |           0.792 |  13.395 |           0.193 |
| BRATS_430 |           0.793 |  70.811 |           0.119 |
| BRATS_297 |           0.811 |  30.916 |           0.15  |
| BRATS_358 |           0.815 |  97.856 |           0.122 |
| BRATS_462 |           0.82  |  39.629 |           0.129 |

**H18 NOT SUPPORTED.** 2.5D minus 2D, VGG-16 plain, seed-pooled Dice: +0.002 [+0.001, +0.004]; 2.5D Dice by seed 0.9068, 0.9065, 0.9050.

**H19a SUPPORTED.** flip averaging minus plain prediction, seed-pooled Dice: +0.001 [+0.000, +0.002].
**H19b SUPPORTED.** small-component removal minus plain prediction, seed-pooled Dice: +0.002 [+0.001, +0.004].
Cut-offs picked on validation: 2 mL, 2 mL, 2 mL.

**H19c SUPPORTED.** Three-seed ensemble minus seed-mean single model Dice: +0.002 [+0.002, +0.003] (ensemble 0.9062, median HD95 2.66 mm).

Descriptive, where the full-budget VGG-16 fails (seed-mean Dice per test patient): Spearman with tumour volume +0.45, with number of lesions -0.35. Single lesion 0.911, several 0.883 mean Dice.

| size_q           |   tumour_ml |   dice |   worst |
|:-----------------|------------:|-------:|--------:|
| smallest quarter |      28.039 |  0.852 |   0.573 |
| second           |      74.018 |  0.909 |   0.715 |
| third            |     133.602 |  0.93  |   0.734 |
| largest quarter  |     176.492 |  0.926 |   0.869 |

Same for MSD-trained models on the 110 LGG patients: Spearman Dice vs tumour volume +0.48, vs far false alarms -0.34; 24 patients below Dice 0.3.

