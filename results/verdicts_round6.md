# Round 6 verdicts

Scored by scripts/analyze_round6.py against PREREGISTRATION.md (Round 6).

**H20a SUPPORTED.** VGG-16 (round 4), calibrated on 53 MSD validation patients: 85.5% of 117 test patients inside their 90% range (median width 30.5 mL).

**H20b SUPPORTED.** InceptionV3 (round 3): 88.9% on MSD test, 26.4% on the 110 LGG patients.

**H20c SUPPORTED.** Recalibrated on 20 LGG patients: mean coverage 90.8% on the other 90 (5th to 95th percentile over splits 77.8% to 98.9%); median width 325.3 mL.

**H20d NOT SUPPORTED.** Weighted conformal with MSD calibration only: 67.3% on LGG (median width 134.2 mL).

**H20e SUPPORTED.** Score scaled by seed spread: coverage 88.0%, median width 27.2 mL against 30.5 mL (-11%).

| setting                                                |   n_cal |   n_test |   coverage |       q |   median_width_ml |
|:-------------------------------------------------------|--------:|---------:|-----------:|--------:|------------------:|
| A: VGG-16 full budget, MSD test                        |      53 |      117 |      0.855 |   0.165 |            30.459 |
| B: InceptionV3 600 steps, MSD test                     |      53 |      117 |      0.889 |   0.27  |            52.322 |
| B: InceptionV3 600 steps, LGG (shifted)                |      53 |      110 |      0.264 |   0.27  |            42.722 |
| A, score scaled by seed spread, MSD test               |      53 |      117 |      0.88  |   2.45  |            27.207 |
| B: LGG, recalibrated on 20 LGG patients (1,000 splits) |      20 |       90 |      0.908 | nan     |           325.29  |
| B: LGG, weighted conformal (MSD calibration)           |      53 |      110 |      0.673 | nan     |           134.161 |

Planted 1017 of 1287 tumours into 117 test patients (the rest found no spot that fits).

**H21a SUPPORTED.** Unscaled planted tumours against real ones by volume quartile: mean absolute Dice difference 0.043 over 3 bins.

|                   |   real_dice |   planted_dice |   n_planted |
|:------------------|------------:|---------------:|------------:|
| (0.0, 47.285]     |       0.855 |          0.79  |          11 |
| (47.285, 92.013]  |       0.908 |          0.861 |          16 |
| (92.013, 155.627] |       0.929 |          0.911 |          14 |
| (155.627, inf]    |       0.927 |          0.925 |           2 |

**H21b SUPPORTED.** Planted Dice at 80 mL minus at 5 mL, per recipient: +0.342 [+0.285, +0.402].

**H21c NOT SUPPORTED.** Natively small minus natively large donors, same target volume: 5 mL -0.147 [-0.226, -0.070]; 10 mL -0.035 [-0.108, +0.040]; 20 mL -0.078 [-0.123, -0.035]; 40 mL -0.058 [-0.092, -0.024]; 80 mL -0.053 [-0.080, -0.023].

|   target_ml |   ('mean', 'large') |   ('mean', 'small') |   ('count', 'large') |   ('count', 'small') |
|------------:|--------------------:|--------------------:|---------------------:|---------------------:|
|           5 |               0.546 |               0.398 |                  117 |                  117 |
|          10 |               0.592 |               0.554 |                  115 |                  115 |
|          20 |               0.784 |               0.708 |                  114 |                  111 |
|          40 |               0.858 |               0.804 |                   99 |                   89 |
|          80 |               0.895 |               0.854 |                   57 |                   40 |

Descriptive: at 10 mL, Spearman of planted Dice with depth below the brain surface -0.21 (n = 230); mean Dice 0.573, missed entirely (Dice < 0.1) 12.2%.
