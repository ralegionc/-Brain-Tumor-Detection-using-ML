# Round 4 verdicts

Scored by scripts/analyze_round4.py against PREREGISTRATION.md (Round 4).

**H10a VGG-16 NOT SUPPORTED.** Dice, gates minus plain, seed-pooled: -0.001 [-0.002, +0.001].

**H10b VGG-16 NOT SUPPORTED.** Distant FP, gates minus plain: -0.06 [-0.13, +0.01] mL; per seed +0.01, -0.10, -0.08.

**H10c VGG-16 NOT SUPPORTED.** Boundary FP (within 5 mm), gates minus plain: -0.03 [-0.13, +0.07] mL.

**H10a InceptionV3 at 2x NOT SUPPORTED.** Dice, gates minus plain, seed-pooled: +0.001 [-0.000, +0.002].

**H10b InceptionV3 at 2x NOT SUPPORTED.** Distant FP, gates minus plain: -0.03 [-0.08, +0.02] mL; per seed +0.08, -0.03, -0.13.

**H10c InceptionV3 at 2x NOT SUPPORTED.** Boundary FP (within 5 mm), gates minus plain: -0.04 [-0.14, +0.05] mL.

**H11: THRESHOLD IS ENOUGH.** VGG-16 plain, Dice+BCE at a per-seed t* minus Tversky at 0.5, seed-pooled test Dice +0.0011 [-0.0011, +0.0040]; mean sensitivity gap +0.0010.

|   seed |   t_star |   val_target_sens |   test_dice_bce_at_0.5 |   test_dice_bce_at_t* |   test_tversky_at_0.5 |   test_sens_dice_bce_at_0.5 |   test_sens_dice_bce_at_t* |   test_sens_tversky_at_0.5 |
|-------:|---------:|------------------:|-----------------------:|----------------------:|----------------------:|----------------------------:|---------------------------:|---------------------------:|
|      0 |    0.027 |             0.95  |                  0.904 |                 0.901 |                 0.898 |                       0.9   |                      0.931 |                      0.935 |
|      1 |    0.027 |             0.949 |                  0.904 |                 0.899 |                 0.897 |                       0.9   |                      0.933 |                      0.932 |
|      2 |    0.02  |             0.948 |                  0.903 |                 0.896 |                 0.898 |                       0.901 |                      0.936 |                      0.93  |

**H12a SUPPORTED.** Dice, VGG-16 minus InceptionV3 at 1x (plain, seed-averaged per patient): +0.012 [+0.008, +0.016].

**H12b SUPPORTED.** Seed-mean Dice: VGG-16 0.9038, InceptionV3 1x 0.8921, 2x 0.9029; 2x closes 93% of the +0.0117 gap. InceptionV3 2x minus VGG-16: -0.001 [-0.004, +0.003].

Every round 4 run on the 117 MSD test patients, with its 600-step counterpart where one exists (descriptive):

| run                                 |   dice |   sensitivity |   precision |   hd95_median |   fp_distant_ml |   fp_boundary_ml |   dice_600_steps |   dice_gain |   gain_lo |   gain_hi |
|:------------------------------------|-------:|--------------:|------------:|--------------:|----------------:|-----------------:|-----------------:|------------:|----------:|----------:|
| r4_inceptionv3_ag_dice_bce_s0_x2    |  0.903 |         0.9   |       0.916 |         2.669 |           0.798 |            6.5   |          nan     |     nan     |   nan     |   nan     |
| r4_inceptionv3_ag_dice_bce_s1_x2    |  0.904 |         0.899 |       0.917 |         2.669 |           0.728 |            6.483 |          nan     |     nan     |   nan     |   nan     |
| r4_inceptionv3_ag_dice_bce_s2_x2    |  0.904 |         0.898 |       0.918 |         2.658 |           0.759 |            6.485 |          nan     |     nan     |   nan     |   nan     |
| r4_inceptionv3_plain_dice_bce_s0    |  0.898 |         0.892 |       0.913 |         3     |           0.663 |            6.726 |            0.868 |       0.029 |     0.023 |     0.036 |
| r4_inceptionv3_plain_dice_bce_s0_x2 |  0.905 |         0.9   |       0.917 |         3     |           0.72  |            6.519 |            0.885 |       0.02  |     0.013 |     0.029 |
| r4_inceptionv3_plain_dice_bce_s1    |  0.896 |         0.886 |       0.917 |         3.182 |           0.589 |            6.58  |            0.866 |       0.03  |     0.023 |     0.038 |
| r4_inceptionv3_plain_dice_bce_s1_x2 |  0.902 |         0.896 |       0.917 |         3     |           0.757 |            6.446 |          nan     |     nan     |   nan     |   nan     |
| r4_inceptionv3_plain_dice_bce_s2    |  0.883 |         0.89  |       0.885 |         4.61  |           3.279 |            6.731 |            0.869 |       0.014 |     0.005 |     0.023 |
| r4_inceptionv3_plain_dice_bce_s2_x2 |  0.902 |         0.898 |       0.915 |         3     |           0.889 |            6.629 |          nan     |     nan     |   nan     |   nan     |
| r4_vgg16_ag_dice_bce_s0             |  0.903 |         0.899 |       0.917 |         2.658 |           0.745 |            6.344 |            0.882 |       0.021 |     0.014 |     0.028 |
| r4_vgg16_ag_dice_bce_s1             |  0.903 |         0.899 |       0.917 |         2.669 |           0.717 |            6.273 |            0.883 |       0.02  |     0.013 |     0.029 |
| r4_vgg16_ag_dice_bce_s2             |  0.903 |         0.898 |       0.917 |         2.669 |           0.66  |            6.254 |            0.887 |       0.016 |     0.012 |     0.021 |
| r4_vgg16_plain_dice_bce_s0          |  0.904 |         0.9   |       0.916 |         2.669 |           0.739 |            6.349 |            0.885 |       0.019 |     0.013 |     0.025 |
| r4_vgg16_plain_dice_bce_s1          |  0.904 |         0.9   |       0.915 |         3     |           0.812 |            6.302 |            0.885 |       0.019 |     0.012 |     0.027 |
| r4_vgg16_plain_dice_bce_s2          |  0.903 |         0.901 |       0.916 |         2.669 |           0.745 |            6.317 |            0.885 |       0.018 |     0.012 |     0.024 |
| r4_vgg16_plain_tversky_s0           |  0.898 |         0.935 |       0.871 |         3.4   |           1.246 |           10.511 |          nan     |     nan     |   nan     |   nan     |
| r4_vgg16_plain_tversky_s1           |  0.897 |         0.932 |       0.874 |         3.473 |           1.228 |           10.25  |          nan     |     nan     |   nan     |   nan     |
| r4_vgg16_plain_tversky_s2           |  0.898 |         0.93  |       0.877 |         3.473 |           1.049 |            9.955 |          nan     |     nan     |   nan     |   nan     |

