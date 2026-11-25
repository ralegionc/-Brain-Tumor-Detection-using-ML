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
| MSD, VGG-16, 2,000 steps                   | test        |    nan      |  nan      |  nan      | not run              |
| MSD, VGG-16, 6,000 steps                   | test        |    nan      |  nan      |  nan      | not run              |
| MSD, VGG-16, 24,500 steps                  | restatement |     -0.0006 |   -0.0016 |    0.0003 | equivalent           |
| MSD, InceptionV3 at 2x, 24,500 steps       | restatement |      0.0008 |   -0.0001 |    0.0017 | equivalent           |
| MSD, VGG-16, 25% of patients               | test        |    nan      |  nan      |  nan      | not run              |
| MSD, VGG-16, 10% of patients               | test        |    nan      |  nan      |  nan      | not run              |

**H14: NOT RUN** (budget runs missing).

**H15 25%: NOT RUN** (runs missing).

**H15 10%: NOT RUN** (runs missing).

**H16a NOT SUPPORTED.** Temperature fitted on validation: 0.97, 0.97, 0.97 (above 1 means overconfident).

**H16b NOT SUPPORTED.** Test ECE (voxels with p >= 0.01) before and after temperature scaling: 0.0776 to 0.0772; 0.0788 to 0.0785; 0.0782 to 0.0777.

| model   |      T |   test_ece |   test_ece_scaled |   ece_cut |
|:--------|-------:|-----------:|------------------:|----------:|
| seed 0  | 0.9675 |     0.0776 |            0.0772 |    0.0055 |
| seed 1  | 0.9669 |     0.0788 |            0.0785 |    0.004  |
| seed 2  | 0.9717 |     0.0782 |            0.0777 |    0.006  |

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

**H17: NOT RUN** (inference outputs missing).

**H18: NOT RUN** (2.5D runs missing).

**H19: NOT RUN** (inference outputs missing).

Descriptive, where the full-budget VGG-16 fails (seed-mean Dice per test patient): Spearman with tumour volume +0.45, with number of lesions -0.35. Single lesion 0.911, several 0.883 mean Dice.

| size_q           |   tumour_ml |   dice |   worst |
|:-----------------|------------:|-------:|--------:|
| smallest quarter |      28.039 |  0.852 |   0.573 |
| second           |      74.018 |  0.909 |   0.715 |
| third            |     133.602 |  0.93  |   0.734 |
| largest quarter  |     176.492 |  0.926 |   0.869 |

Same for MSD-trained models on the 110 LGG patients: Spearman Dice vs tumour volume +0.48, vs far false alarms -0.34; 24 patients below Dice 0.3.

