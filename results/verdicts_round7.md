# Round 7 verdicts

Scored by scripts/analyze_round7.py against PREREGISTRATION.md (Round 7).

Small quarter: 30 of 117 test patients (true volume at most 47.3 mL).

**H22a SUPPORTED.** Overall Dice, S minus B: -0.0011, 90% CI [-0.0027, +0.0006] (B 0.9038, S 0.9027, N 0.9010).

**H22b NOT SUPPORTED.** Small quarter Dice, S minus B: -0.001 [-0.007, +0.007] (B 0.852, S 0.851).

**H22c NOT SUPPORTED.** Small quarter Dice, S minus N: +0.005 [-0.000, +0.012]. For reference N minus B: -0.005 [-0.013, +0.002].

**H22d SUPPORTED.** Planted 5 and 10 mL tumours from naturally large donors, S minus B: +0.351 (232 plants; B 0.562, S 0.914).

**H22e SUPPORTED.** Same, naturally small donors: +0.407 [+0.371, +0.444] per recipient (232 plants; B 0.475, S 0.881); half the H22d gain is +0.176.

**H22f SUPPORTED.** Distant false alarms, S minus B: +0.08 [+0.02, +0.15] mL (B 0.77, S 0.85, N 0.88).

## Descriptive

Real test patients (missed lesions are seed means summed over patients):

| arm   |   dice |   dice_small_quarter |   dice_rest |   sensitivity |   precision |   fp_distant_ml |   fp_boundary_ml |   missed_lesions_big |   lesions_big |   dice_s0 |   dice_s1 |   dice_s2 |   small_s0 |   small_s1 |   small_s2 |
|:------|-------:|---------------------:|------------:|--------------:|------------:|----------------:|-----------------:|---------------------:|--------------:|----------:|----------:|----------:|-----------:|-----------:|-----------:|
| B     | 0.9038 |               0.852  |      0.9217 |        0.9005 |      0.9157 |          0.7654 |           6.3226 |              22      |           167 |    0.9043 |    0.9037 |    0.9034 |     0.8546 |     0.8528 |     0.8485 |
| S     | 0.9027 |               0.8513 |      0.9205 |        0.9014 |      0.9131 |          0.8472 |           6.437  |              21.3333 |           167 |    0.9023 |    0.9027 |    0.9032 |     0.8489 |     0.8519 |     0.8529 |
| N     | 0.901  |               0.8466 |      0.9197 |        0.9012 |      0.9107 |          0.8787 |           6.6755 |              24      |           167 |    0.9006 |    0.9009 |    0.9013 |     0.8469 |     0.8449 |     0.8479 |

Planted tumours, seed-mean Dice by donor kind and target volume:

|                        |     B |     S |     N |   S-B |   N-B |
|:-----------------------|------:|------:|------:|------:|------:|
| ('small', '5 mL')      | 0.4   | 0.877 | 0.821 | 0.477 | 0.42  |
| ('small', '10 mL')     | 0.552 | 0.886 | 0.872 | 0.334 | 0.321 |
| ('small', '20 mL')     | 0.713 | 0.886 | 0.902 | 0.174 | 0.189 |
| ('small', '40 mL')     | 0.801 | 0.899 | 0.924 | 0.098 | 0.124 |
| ('small', '80 mL')     | 0.856 | 0.899 | 0.943 | 0.043 | 0.087 |
| ('large', '5 mL')      | 0.539 | 0.914 | 0.849 | 0.375 | 0.31  |
| ('large', '10 mL')     | 0.586 | 0.913 | 0.869 | 0.328 | 0.283 |
| ('large', '20 mL')     | 0.786 | 0.926 | 0.912 | 0.14  | 0.126 |
| ('large', '40 mL')     | 0.859 | 0.933 | 0.933 | 0.073 | 0.073 |
| ('large', '80 mL')     | 0.894 | 0.934 | 0.951 | 0.04  | 0.057 |
| ('native', 'own size') | 0.859 | 0.921 | 0.941 | 0.062 | 0.082 |

Planted 5 and 10 mL tumours missed entirely (Dice under 0.1): B 13.1%, S 0.0%, N 0.0%.

Bank small: 2512 plants, planted volume 10th/50th/90th percentile 3.8 / 9.6 / 23.4 mL.

Bank native: 2379 plants, planted volume 10th/50th/90th percentile 17.1 / 38.2 / 94.3 mL.
